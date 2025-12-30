import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count

from .pytorch_selective_scan import SelectiveScanEasy

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

# pytorch cross scan =============
class CrossScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        x_flat = x.flatten(2, 3)
        x_t    = x.transpose(2, 3).flatten(2, 3)
        x_flip = torch.flip(x_flat, dims=[-1])
        x_tflip = torch.flip(x_t, dims=[-1])
        xs = torch.stack([x_flat, x_t, x_flip, x_tflip], dim=1)
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(
            dim0=2, dim1=3
        ).contiguous().view(B, -1, L)
        return y.view(B, -1, H, W)


class CrossMerge(torch.autograd.Function): # Keep Cross_Merge for now
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(
            dim0=2, dim1=3
        ).contiguous().view(B, D, -1)
        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 4, C, L))
        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        xs = xs.view(B, 4, C, H, W)
        return xs


# import selective scan ==============================
try:
    import selective_scan_cuda_oflex
except Exception as e:
    ...
    # print(f"WARNING: can not import selective_scan_cuda_oflex.", flush=True)
    # print(e, flush=True)

try:
    import selective_scan_cuda_core
except Exception as e:
    ...
    # print(f"WARNING: can not import selective_scan_cuda_core.", flush=True)
    # print(e, flush=True)

try:
    import selective_scan_cuda
except Exception as e:
    ...
    # print(f"WARNING: can not import selective_scan_cuda.", flush=True)
    # print(e, flush=True)


def check_nan_inf(tag: str, x: torch.Tensor, enable=True):
    if enable:
        if torch.isinf(x).any() or torch.isnan(x).any():
            print(tag, torch.isinf(x).any(), torch.isnan(x).any(), flush=True)
            import pdb

            pdb.set_trace()


# fvcore flops =======================================
def flops_selective_scan_fn(
    B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_complex=False
):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    assert not with_complex
    # https://github.com/state-spaces/mamba/issues/110
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    return flops


# this is only for selective_scan_ref...
def flops_selective_scan_ref(
    B=1,
    L=256,
    D=768,
    N=16,
    with_D=True,
    with_Z=False,
    with_Group=True,
    with_complex=False,
):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    import numpy as np

    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop

    assert not with_complex

    flops = 0  # below code flops = 0

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum(
            [[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln"
        )
    else:
        flops += get_flops_einsum(
            [[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln"
        )

    in_for_flops = B * D * N
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    return flops


def print_jit_input_names(inputs):
    print("input params: ", end=" ", flush=True)
    try:
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)


# cross selective scan ===============================
class SelectiveScanCore(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(
        ctx,
        u,
        delta,
        A,
        B,
        C,
        D=None,
        delta_bias=None,
        delta_softplus=False,
        nrows=1,
        backnrows=1,
        oflex=True,
    ):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_core.fwd(
            u, delta, A, B, C, D, delta_bias, delta_softplus, 1
        )
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)

def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
    if u.device.type == "cpu":   
        return SelectiveScanEasy.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, False, 64)
    else:
        return SelectiveScanCore.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, -1, -1, True,)
torch.fx.wrap("selective_scan")

def selective_scan_flop_jit(inputs, outputs):
    print_jit_input_names(inputs)
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_selective_scan_fn(B=B, L=L, D=D, N=N, with_D=True, with_Z=False)
    return flops


# =====================================================
# we have this class as linear and conv init differ from each other
# this function enable loading from both conv2d or linear
class Linear2d(nn.Linear):
    def forward(self, x: torch.Tensor):
        # B, C, H, W = x.shape
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        state_dict[prefix + "weight"] = state_dict[prefix + "weight"].view(
            self.weight.shape
        )
        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(
            x, self.normalized_shape, self.weight, self.bias, self.eps
        )
        x = x.permute(0, 3, 1, 2)
        return x


class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(
            4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False
        )
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _patch_merging_pad(x: torch.Tensor):
        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
        return x

    def forward(self, x):
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)

        return x


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0        
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Mlp_Downstream(nn.Module):
    def __init__(self,
                in_features,
                hidden_features = None,                
                act_layer = nn.ReLU,
                drop=0.0,
                last_vss=False):
        super().__init__()
        out_features = in_features * 2 if last_vss else in_features
        hidden_features = hidden_features or in_features
        
        Linear = Linear2d
        
        # Define layer
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        
        if last_vss:
            self.dim_reduce = nn.Conv2d(in_channels = hidden_features,
                                out_channels = out_features,
                                kernel_size = 1,
                                stride = 1,
                                padding = 0,
                                bias = False)
            
            self.SumPool = nn.Conv2d(
                in_channels=hidden_features,
                out_channels=hidden_features,
                kernel_size=2,
                stride=2,
                padding=0,
                groups=hidden_features,
                bias=False
            )
            with torch.no_grad():
                self.SumPool.weight.fill_(1.0)
            for p in self.SumPool.parameters():
                p.requires_grad = False
            
            self.fc2 = nn.Sequential(
                OrderedDict(
                    [
                        ("mlp_sumpool", self.SumPool),
                        ("mlp_dim_redu", self.dim_reduce),
                    ]
                )
            )
            
        else:
            self.fc2 = Linear(hidden_features, out_features)
            
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# =====================================================


class SS2D(nn.Module):
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3,  # < 2 means no conv
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        # ======================
        forward_type="v2",
        channel_first=False,
        # ======================
        **kwargs,
    ):
        kwargs.update(
            d_model=d_model,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            act_layer=act_layer,
            d_conv=d_conv,
            conv_bias=conv_bias,
            dropout=dropout,
            bias=bias,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init=dt_init,
            dt_scale=dt_scale,
            dt_init_floor=dt_init_floor,
            initialize=initialize,
            forward_type=forward_type,
            channel_first=channel_first,
        )
        # only used to run previous version        
        self.__initv2__(**kwargs)

    def __initv2__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3,  # < 2 means no conv
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        # ======================
        forward_type="v2",
        channel_first=False,
        # ======================
        **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_conv = d_conv
        self.channel_first = channel_first
        Linear = Linear2d
        self.forward = self.forwardv2
        
        self.out_norm = LayerNorm2d(d_inner)

        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            v5=partial(
                self.forward_corev2,
                force_fp32=True,
                SelectiveScan=SelectiveScanCore,
                CrossScan=CrossScan,
                CrossMerge=CrossMerge,                
            ),
        )
        self.forward_core = FORWARD_TYPES.get("v5", None)
        k_group = 4        

        # in proj =======================================
        d_proj = d_inner * 2
        self.in_proj = nn.Conv2d(in_channels=d_model, out_channels=d_proj, kernel_size=1, bias=bias)
        self.act: nn.Module = act_layer()

        # conv =======================================
        self.conv2d = nn.Conv2d(in_channels=d_inner, out_channels=d_inner,
            groups=d_inner, bias=conv_bias, kernel_size=d_conv,
            padding=(d_conv - 1) // 2, **factory_kwargs)

        # x proj ============================        
        self.x_proj_conv = nn.Conv1d(in_channels=k_group * d_inner,
                                    out_channels=k_group * (dt_rank + d_state * 2),
                                    kernel_size=1, groups=k_group, bias=False)


        # out proj =======================================
        # self.out_proj = Linear(d_inner, d_model, bias=bias, **factory_kwargs)
        self.out_proj = nn.Conv2d(in_channels=d_inner, out_channels=d_model, kernel_size=1, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        
        # dt proj ============================
        self.dt_projs = [self.dt_init(dt_rank, d_inner, dt_scale, dt_min, dt_max, dt_init_floor, **factory_kwargs,) for _ in range(k_group)]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
        del self.dt_projs

        self.dt_proj_conv = nn.Conv1d(in_channels=k_group * dt_rank, out_channels=k_group * d_inner, kernel_size=1, groups=k_group, bias=False)

        # A, D =======================================
        self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True)  # (K * D, N)
        self.Ds = self.D_init(d_inner, copies=k_group, merge=True)  # (K * D)

        self.K = k_group
        self.D = self.dt_proj_conv.weight.shape[0] // self.K
        self.R = self.dt_proj_conv.weight.shape[1]
        
        with torch.no_grad():
            self.dt_proj_conv.weight.copy_(self.dt_projs_weight.reshape(self.K * self.D, self.R, 1).contiguous())
            del self.dt_projs_weight

    @staticmethod
    def dt_init(
        dt_rank,
        d_inner,
        dt_scale=1.0,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        **factory_kwargs,
    ):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(torch.rand(d_inner, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D
    
    def forward_corev2(
        self,
        x: torch.Tensor = None,        
        dt_projs_bias: torch.Tensor = None,
        A_logs: torch.Tensor = None,
        Ds: torch.Tensor = None,
        delta_softplus=True,
        out_norm: torch.nn.Module = None,
        channel_first=False,
        # ==============================
        to_dtype=True,  # True: final out to dtype
        force_fp32=False,  # True: input fp32
        # ==============================
        ssoflex=True,  # True: out fp32 in SSOflex; else, SSOflex is the same as SSCore
        # ==============================
        SelectiveScan=None,
        CrossScan=CrossScan,
        CrossMerge=CrossMerge,
        no_einsum=True,  # replace einsum with linear or conv1d to raise throughput
        **kwargs,
    ):        
        dt_projs_bias = self.dt_projs_bias
        A_logs = self.A_logs
        Ds = self.Ds
        out_norm = getattr(self, "out_norm", None)
        channel_first = self.channel_first

        B, D, H, W = x.shape
        D, N = A_logs.shape        
        K, D, R = self.K, self.D, self.R        
        L = H * W
        
        xs = CrossScan.apply(x)
        
        x_dbl = self.x_proj_conv(xs.view(B, -1, L)) # x_dbl = F.conv1d
        
        dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
        
        dts = self.dt_proj_conv(dts.contiguous().view(B, -1, L))

        xs = xs.view(B, -1, L)
        dts = dts.contiguous().view(B, -1, L)
        As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
        Bs = Bs.contiguous().view(B, K, N, L)
        Cs = Cs.contiguous().view(B, K, N, L)
        Ds = Ds.to(torch.float)  # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)
        
        xs = xs.to(torch.float)
        dts = dts.to(torch.float)
        Bs = Bs.to(torch.float)
        Cs = Cs.to(torch.float)

        ys = selective_scan(xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus)

        ys = ys.view(B, K, -1, H, W)

        y: torch.Tensor = CrossMerge.apply(ys)
            
        y = y.view(B, -1, H, W)
        
        y = out_norm(y)

        return y.to(x.dtype)
    
    def forwardv2(self, x: torch.Tensor, **kwargs):        
        x = self.in_proj(x)
        x, z = x.chunk(2, dim=1)
        z = self.act(z)
        x = self.conv2d(x)  # (b, d, h, w)
        x = self.act(x)
        y = self.forward_core(x)
        y = y * z
        y = self.out_proj(y)
        out = self.dropout(y)
        return out

class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: nn.Module = nn.LayerNorm,
        channel_first=False,
        # =============================
        ssm_d_state: int = 16,
        ssm_ratio=2.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        ssm_init="v0",
        forward_type="v2",
        # =============================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate: float = 0.0,
        gmlp=False,
        # =============================
        use_checkpoint: bool = False,
        post_norm: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.ss2d = SS2D(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
                channel_first=channel_first,
            )

        self.drop_path = DropPath(drop_path)

        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(
                in_features=hidden_dim,
                hidden_features=mlp_hidden_dim,
                act_layer=mlp_act_layer,
                drop=mlp_drop_rate,                
            )            

    def _forward(self, input: torch.Tensor):
        if self.ssm_branch:
            x = self.norm(input)
            x = self.ss2d(x)
            x = input + self.drop_path(x)

        if self.mlp_branch:            
            mlp_res = x.clone()
            x = self.norm2(x)
            x = self.mlp(x)            
            x = mlp_res + self.drop_path(x)

        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:            
            return self._forward(input)

class VSSBlock_PVM(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: nn.Module = nn.LayerNorm,
        channel_first=False,
        # =============================
        ssm_d_state: int = 16,
        ssm_ratio=2.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        ssm_init="v0",
        forward_type="v2",
        # =============================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate: float = 0.0,
        gmlp=False,
        # =============================
        use_checkpoint: bool = False,
        post_norm: bool = False,
        # =============================
        reduce_factor = 4,
        # =============================
        **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        self.reduce_factor = reduce_factor
        reduct_dim = hidden_dim // self.reduce_factor

        print(f"HIDDEN DIM INIT: {hidden_dim}; REDUCTION FACTOR: {self.reduce_factor}")

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.ss2d = SS2D(
                d_model=reduct_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
                channel_first=channel_first,
            )
            self.skip_scale = nn.Identity()

        self.drop_path = DropPath(drop_path)

        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(
                in_features=hidden_dim,
                hidden_features=mlp_hidden_dim,
                act_layer=mlp_act_layer,
                drop=mlp_drop_rate,                
            )

    def _forward(self, input: torch.Tensor):
        if self.ssm_branch:
            x = self.norm(input)
            
            # xs = torch.chunk(x, self.reduce_factor, dim=1)
            # outs = [self.ss2d(xi) + self.drop_path(self.skip_scale(xi)) for xi in xs]
            # x = torch.cat(outs, dim=1)
            
            # Vectorize version
            B, C, H, W = x.shape
            
            subC = C // self.reduce_factor

            # Fold channels → batch dimension
            x_parallel = x.reshape(B * self.reduce_factor, subC, H, W)

            # Skip connection (per branch)
            skip_parallel = self.skip_scale(x_parallel)

            # Shared SS2D on expanded batch
            y_parallel = self.ss2d(x_parallel)

            # Add drop_path skip
            y_parallel = y_parallel + self.drop_path(skip_parallel)

            # Restore original shape
            x = y_parallel.reshape(B, C, H, W)

        if self.mlp_branch:            
            mlp_res = x.clone()
            x = self.norm2(x)
            x = self.mlp(x)            
            x = mlp_res + self.drop_path(x)

        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:            
            return self._forward(input)

class VSSBlock_Downstream(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: nn.Module = nn.LayerNorm,
        channel_first=False,
        # =============================
        ssm_d_state: int = 16,
        ssm_ratio=2.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        ssm_init="v0",
        forward_type="v2",
        # =============================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate: float = 0.0,
        gmlp=False,
        # =============================
        use_checkpoint: bool = False,
        post_norm: bool = False,
        block_last_vss = False,
        **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        print(f"HIDDEN DIM INIT: {hidden_dim}")

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.ss2d = SS2D(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
                channel_first=channel_first,
            )

        self.drop_path = DropPath(drop_path)

        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            
            if block_last_vss:
                self.skip_reduce = nn.Conv2d(in_channels = hidden_dim,
                                        out_channels = hidden_dim * 2,
                                        kernel_size = 1,
                                        stride = 1,
                                        padding = 0,
                                        bias = False)
                            
                self.skip_SumPool = nn.Conv2d(in_channels=hidden_dim,
                                        out_channels=hidden_dim,
                                        kernel_size=2,
                                        stride=2,
                                        padding=0,
                                        groups=hidden_dim,
                                        bias=False)
                with torch.no_grad():
                    self.skip_SumPool.weight.fill_(1.0)
                    for p in self.skip_SumPool.parameters():
                        p.requires_grad = False
                            
                self.skip_proj = nn.Sequential(
                    OrderedDict(
                        [
                            ("skip_sumpool", self.skip_SumPool),
                            ("skip_dim_redu", self.skip_reduce),
                        ]
                    )
                )
            else:
                self.skip_proj = nn.Identity()
            
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp_Downstream(
                in_features=hidden_dim,
                hidden_features=mlp_hidden_dim,
                act_layer=mlp_act_layer,
                drop=mlp_drop_rate,
                last_vss=block_last_vss,
            )        

    def _forward(self, input: torch.Tensor):        
        if self.ssm_branch:            
            x = self.norm(input)
            x = self.ss2d(x)
            x = input + self.drop_path(x)

        if self.mlp_branch:            
            mlp_res = self.skip_proj(x)
            x = self.norm2(x)
            x = self.mlp(x)
            x = mlp_res + self.drop_path(x)

        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:            
            return self._forward(input)

class VSSBlock_Downstream_PVM(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: nn.Module = nn.LayerNorm,
        channel_first=False,
        # =============================
        ssm_d_state: int = 16,
        ssm_ratio=2.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        ssm_init="v0",
        forward_type="v2",
        # =============================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate: float = 0.0,
        gmlp=False,
        # =============================
        use_checkpoint: bool = False,
        post_norm: bool = False,
        block_last_vss = False,
        # =============================
        reduce_factor = 4,
        # =============================
        **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        self.reduce_factor = reduce_factor
        reduct_dim = hidden_dim // self.reduce_factor
        print(f"HIDDEN DIM INIT: {hidden_dim}; REDUCTION FACTOR: {self.reduce_factor}")
        

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.ss2d = SS2D(
                # d_model=hidden_dim,
                d_model=reduct_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
                channel_first=channel_first,
            )
            self.skip_scale = nn.Identity()

        self.drop_path = DropPath(drop_path)

        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            
            if block_last_vss:
                self.skip_reduce = nn.Conv2d(in_channels = hidden_dim,
                                        out_channels = hidden_dim * 2,
                                        kernel_size = 1,
                                        stride = 1,
                                        padding = 0,
                                        bias = False)
                            
                self.skip_SumPool = nn.Conv2d(in_channels=hidden_dim,
                                        out_channels=hidden_dim,
                                        kernel_size=2,
                                        stride=2,
                                        padding=0,
                                        groups=hidden_dim,
                                        bias=False)
                with torch.no_grad():
                    self.skip_SumPool.weight.fill_(1.0)
                    for p in self.skip_SumPool.parameters():
                        p.requires_grad = False
                            
                self.skip_proj = nn.Sequential(
                    OrderedDict(
                        [
                            ("skip_sumpool", self.skip_SumPool),
                            ("skip_dim_redu", self.skip_reduce),
                        ]
                    )
                )                
            else:
                self.skip_proj = nn.Identity()
            
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp_Downstream(
                in_features=hidden_dim,
                hidden_features=mlp_hidden_dim,
                act_layer=mlp_act_layer,
                drop=mlp_drop_rate,
                last_vss=block_last_vss,
            )   

    def _forward(self, input: torch.Tensor):        
        if self.ssm_branch:            
            x = self.norm(input)
            
            # Vectorize version
            B, C, H, W = x.shape
            
            subC = C // self.reduce_factor
            # Fold channels → batch dimension
            x_parallel = x.reshape(B * self.reduce_factor, subC, H, W)
            # Skip connection (per branch)
            skip_parallel = self.skip_scale(x_parallel)
            # Shared SS2D on expanded batch
            y_parallel = self.ss2d(x_parallel)
            # Add drop_path skip
            y_parallel = y_parallel + self.drop_path(skip_parallel)
            # Restore original shape
            x = y_parallel.reshape(B, C, H, W)            

        if self.mlp_branch:            
            mlp_res = self.skip_proj(x)
            x = self.norm2(x)
            x = self.mlp(x)
            x = mlp_res + self.drop_path(x)

        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:            
            return self._forward(input)
# ==================================================