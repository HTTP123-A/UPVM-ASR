# Modified by Mzero #20240123
# Copyright (C) 2023, Tri Dao, Albert Gu.

import math
from typing import Optional
from functools import partial

import torch
import torch.nn.functional as F

from torch.library import Library, impl

MODE = "v0"

# ==============================================================================================================
# FULL Pytorch VERSION
class SelectiveScanEasy(torch.autograd.Function):
    # for debug, we use it as an orinary object
    DEBUG = (MODE == "fnDEBUG")

    if DEBUG:
        print("DEBUG here...", flush=True)
        saved_tensors = []
        
        @classmethod
        def save_for_backward(ctx, *args):
            ctx.saved_tensors = args

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, us, dts, As, Bs, Cs, Ds, delta_bias=None, delta_softplus=False, return_last_state=False, chunksize=64):
        #ctx shape: {ctx.shape}
        has_D = Ds is not None
        dtype = torch.float32
        
        dts = dts.to(dtype)
        if delta_bias is not None:
            dts = dts + delta_bias.view(1, -1, 1).to(dtype)
        if delta_softplus:
            dts = torch.nn.functional.softplus(dts)

        B_squeeze = (len(Bs.shape) == 3)
        C_squeeze = (len(Cs.shape) == 3)
        if B_squeeze:
            Bs = Bs.unsqueeze(1)
        if C_squeeze:
            Cs = Cs.unsqueeze(1)
        B, G, N, L = Bs.shape
        us = us.view(B, G, -1, L).permute(3, 0, 1, 2).to(dtype)
        dts = dts.view(B, G, -1, L).permute(3, 0, 1, 2).to(dtype)
        As = As.view(G, -1, N).to(dtype)
        Bs = Bs.permute(3, 0, 1, 2).to(dtype)
        Cs = Cs.permute(3, 0, 1, 2).to(dtype)
        Ds = Ds.view(G, -1).to(dtype) if has_D else None
        D = As.shape[1]
        
        ctx.shape = (B, G, D, N, L)
        ctx.delta_softplus = delta_softplus
        ctx.return_last_state = return_last_state
        ctx.chunksize = chunksize
        ctx.BC_squeeze = (B_squeeze, C_squeeze)
        save_for_backward = [us, dts, As, Bs, Cs, Ds, delta_bias]

        chunks = list(range(0, L, chunksize))
        oys = []
        ohs = []        
        hprefix = us.new_zeros((B, G, D, N), dtype=torch.float)
        for i in chunks:
            ts = dts[i:i+chunksize].cumsum(dim=0)
            Ats = torch.einsum("gdn,lbgd->lbgdn", As, ts).exp()
            # scale = Ats[-1:].detach()
            scale = 1
            rAts = Ats / scale
            duts = dts[i:i + chunksize] * us[i:i + chunksize]
            dtBus = torch.einsum("lbgd,lbgn->lbgdn", duts, Bs[i:i + chunksize])
            tmp_dtBus_div_rAts = (dtBus / rAts)
            tmp_dtBus_div_rAts_cumsum = tmp_dtBus_div_rAts.cumsum(dim=0) 
            hs = rAts * tmp_dtBus_div_rAts_cumsum + Ats * hprefix.unsqueeze(0)
            ys = torch.einsum("lbgn,lbgdn->lbgd", Cs[i:i + chunksize], hs) 
            oys.append(ys)
            ohs.append(hs)
            hprefix = hs[-1]

        oys = torch.cat(oys, dim=0)
        ohs = torch.cat(ohs, dim=0)
        if has_D:
            oys = oys + Ds * us

        save_for_backward.extend([ohs])

        ctx.save_for_backward(*save_for_backward)
        
        oys = oys.permute(1, 2, 3, 0).view(B, -1, L)

        if getattr(ctx, "DEBUG", False):
            print("DEBUG here ..............", flush=True)
            oys.backward = partial(ctx.backward, ctx)
        
        #return oys, hprefix.view(B, G * D, N)        
        return oys

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, doys: torch.Tensor, *args):
        DEBUG = getattr(ctx, "DEBUG", False)
        us, dts, As, Bs, Cs, Ds, delta_bias, ohs = ctx.saved_tensors
                
        B, G, D, N, L = ctx.shape
        chunksize = ctx.chunksize
        delta_softplus = ctx.delta_softplus
        doys = doys.view(B, G, D, L).permute(3, 0, 1, 2)

        def rev_comsum(x):
            cum_sum = torch.cumsum(x, dim=0)
            return (x - cum_sum + cum_sum[-1:None])
        
        if DEBUG:
            dtype = torch.float32
            us = us.requires_grad_()
            dts = dts.requires_grad_()
            As = As.requires_grad_()
            Bs = Bs.requires_grad_()
            Cs = Cs.requires_grad_()
            Ds = Ds.requires_grad_() if Ds is not None else None
            delta_bias = delta_bias.requires_grad_() if delta_bias is not None else None
            ohs = ohs.requires_grad_()

        # copy forward again
        if DEBUG:
            has_D = Ds is not None
            
            tmp_fwd_dtBus = []
            tmp_fwd_rAts = []
            tmp_fwd_Ats = []
            tmp_fwd_dtBus_div_rAts_cumsum = []
            tmp_fwd_dtBus_div_rAts = []

            chunks = list(range(0, L, chunksize))
            oys = []
            ohs = []        
            hprefix = us.new_zeros((B, G, D, N), dtype=torch.float)
            for i in chunks:
                ts = dts[i:i+chunksize].cumsum(dim=0)
                Ats = torch.einsum("gdn,lbgd->lbgdn", As, ts).exp()
                # scale = Ats[-1:].detach()
                scale = 1
                rAts = Ats / scale
                duts = dts[i:i + chunksize] * us[i:i + chunksize]
                dtBus = torch.einsum("lbgd,lbgn->lbgdn", duts, Bs[i:i + chunksize])
                tmp_dtBus_div_rAts = (dtBus / rAts)
                tmp_dtBus_div_rAts_cumsum = tmp_dtBus_div_rAts.cumsum(dim=0) 
                hs = rAts * tmp_dtBus_div_rAts_cumsum + Ats * hprefix.unsqueeze(0)
                ys = torch.einsum("lbgn,lbgdn->lbgd", Cs[i:i + chunksize], hs) 
                oys.append(ys)
                ohs.append(hs)
                hprefix = hs[-1]

                tmp_fwd_dtBus_div_rAts_cumsum.append(tmp_dtBus_div_rAts_cumsum)
                tmp_fwd_dtBus_div_rAts.append(tmp_dtBus_div_rAts)
                tmp_fwd_dtBus.append(dtBus)
                tmp_fwd_rAts.append(rAts)
                tmp_fwd_Ats.append(Ats)

            oys = torch.cat(oys, dim=0)
            ohs = torch.cat(ohs, dim=0)
            if has_D:
                oys = oys + Ds * us

        if DEBUG:
            _oys = oys.requires_grad_()

        dus = None
        dDs = None
        if Ds is not None:
            dDs = torch.einsum("lbgd,lbgd->gd", doys, us).view(-1)
            dus = torch.einsum("lbgd,gd->lbgd", doys, Ds)

        chunks = list(range(0, L, chunksize))
        dAs = us.new_zeros((G, D, N), dtype=torch.float)
        dus = us.new_zeros((L, B, G, D), dtype=torch.float) if dus is None else dus
        ddts = us.new_zeros((L, B, G, D), dtype=torch.float)
        dBs = us.new_zeros((L, B, G, N), dtype=torch.float)
        dCs = us.new_zeros((L, B, G, N), dtype=torch.float)
        dhprefix = us.new_zeros((B, G, D, N), dtype=torch.float)
        for i in chunks[::-1]:
            # forward procedure ================
            tmp_dts = dts[i:i+chunksize]
            ts = tmp_dts.cumsum(dim=0)
            Ats = torch.einsum("gdn,lbgd->lbgdn", As, ts).exp()
            scale = Ats[-1].detach()
            scale = 1
            rAts = Ats / scale
            duts = dts[i:i + chunksize] * us[i:i + chunksize]
            dtBus = torch.einsum("lbgd,lbgn->lbgdn", duts, Bs[i:i + chunksize])
            dtBus_div_rAts = (dtBus / rAts)
            hs_minus_prefix_div_rAts = dtBus_div_rAts.cumsum(dim=0)
            
            # hs_minus_prefix = rAts * hs_minus_prefix_div_rAts
            
            # below code is not ok due to precision limitation...
            # use saved hs instead
            if False:
                hprefix = (hsuffix - hs_minus_prefix[-1]) / scale
                hs = hs_minus_prefix + Ats * hprefix.unsqueeze(0)

            # backward procedure ================
            _hs = ohs[i:i+chunksize]
            _hprefix = ohs[i - 1] if i > 0 else None
            dCs[i:i + chunksize] = torch.einsum("lbgd,lbgdn->lbgn", doys[i:i + chunksize], _hs)
            dhs = doys[i:i + chunksize].unsqueeze(4) * Cs[i:i + chunksize].unsqueeze(3) # lbgd,lbgn->lbgdn
            dhs[-1] = dhs[-1] + dhprefix
            dhprefix = torch.einsum("lbgdn,lbgdn -> bgdn", dhs, Ats)
            dAts_hprefix = dhs * _hprefix.unsqueeze_(0) if i > 0 else None # lbgdn,bgdn->lbgdn
            drAts_hs_minus_prefix = dhs * hs_minus_prefix_div_rAts
            dhs_minus_prefix_div_rAts = dhs * rAts
            
            if DEBUG:
                print("1", (torch.autograd.grad(_oys, tmp_fwd_dtBus_div_rAts_cumsum[chunks.index(i)], doys, create_graph=True, allow_unused=True)[0] - dhs_minus_prefix_div_rAts).abs().sum())

            d_dtBus_div_rAts = rev_comsum(dhs_minus_prefix_div_rAts)
            if DEBUG:
                d_dtBus_div_rAts_v1 = torch.autograd.grad(hs_minus_prefix_div_rAts, dtBus_div_rAts, dhs_minus_prefix_div_rAts, create_graph=True, allow_unused=True)[0]
                print("2.0", (torch.autograd.grad(_oys, tmp_fwd_dtBus_div_rAts[chunks.index(i)], doys, create_graph=True, allow_unused=True)[0] - d_dtBus_div_rAts).abs().sum())
                print("2.1", (d_dtBus_div_rAts - d_dtBus_div_rAts_v1).abs().sum())
                d_dtBus_div_rAts = d_dtBus_div_rAts_v1
            
            ddtBus = d_dtBus_div_rAts / rAts
            dBs[i:i + chunksize] = torch.einsum("lbgdn,lbgd->lbgn", ddtBus, duts)
            dduts = torch.einsum("lbgdn,lbgn->lbgd", ddtBus, Bs[i:i + chunksize])
            dus[i:i + chunksize] = dus[i:i + chunksize] + dduts * dts[i:i + chunksize]
            if DEBUG:
                print("3", (torch.autograd.grad(_oys, tmp_fwd_dtBus[chunks.index(i)], doys, create_graph=True, allow_unused=True)[0] - ddtBus).abs().sum())

            if DEBUG:
                tmp_a = torch.randn((L, B, G, D, N)).to(dtype).cuda().requires_grad_()
                tmp_b = torch.cumsum(tmp_a, dim=0)
                tmp_c = torch.randn((L, B, G, D, N)).to(dtype).cuda()
                print("ex.0", (torch.autograd.grad(tmp_b, tmp_a, tmp_c, create_graph=True, allow_unused=True)[0] - rev_comsum(tmp_c)).abs().sum())

            drAts_dtBus_div_rAts = d_dtBus_div_rAts * (-dtBus_div_rAts / rAts)
            if DEBUG:
                drAts_dtBus_div_rAts_v1 = d_dtBus_div_rAts * (dtBus / -(rAts * rAts)) # do not use this!!!
                drAts_dtBus_div_rAts_ref = torch.autograd.grad(dtBus_div_rAts, rAts, d_dtBus_div_rAts, create_graph=True, allow_unused=True)[0]
                print("4.0", (drAts_dtBus_div_rAts - drAts_dtBus_div_rAts_ref).abs().sum())
                print("4.0_v1", (drAts_dtBus_div_rAts - drAts_dtBus_div_rAts_v1).abs().sum())

            ddts[i:i + chunksize] = dduts * us[i:i + chunksize]
            dAts = drAts_dtBus_div_rAts / scale + drAts_hs_minus_prefix / scale + (dAts_hprefix if i > 0 else 0)

            if DEBUG:
                drAts_ref = torch.autograd.grad(_oys, tmp_fwd_rAts[chunks.index(i)], doys, create_graph=True, allow_unused=True)[0]
                dAts_ref = torch.autograd.grad(_oys, tmp_fwd_Ats[chunks.index(i)], doys, create_graph=True, allow_unused=True)[0]
                print("4.1", (drAts_ref - (drAts_dtBus_div_rAts + drAts_hs_minus_prefix)).abs().sum())
                print("4.2", ((drAts_ref - (drAts_dtBus_div_rAts + drAts_hs_minus_prefix)) / scale).abs().sum())
                print("4.3", (dAts_ref - dAts).abs().sum())
                
            dAts_noexp = dAts * Ats # d(e^x) = e^x * dx
            dAs = dAs + torch.einsum("lbgdn,lbgd->gdn", dAts_noexp, ts)
            d_ts = torch.einsum("lbgdn,gdn->lbgd", dAts_noexp, As)
            
            _part_ddts = rev_comsum(d_ts) # the precision is enough
            if DEBUG:
                _part_ddts_v1 = torch.autograd.grad(ts, tmp_dts, d_ts, create_graph=True, allow_unused=True)[0]
                print("5.0", (_part_ddts - _part_ddts_v1).abs().sum())

            ddts[i:i + chunksize] = ddts[i:i + chunksize] + _part_ddts
        
        if DEBUG:
            print("f", (torch.autograd.grad(_oys, dts, doys, create_graph=True, allow_unused=True)[0] - ddts).abs().sum(), flush=True)
        
        if delta_softplus:
            # softplus = log(1 + e^x); dsoftplus = e^x /(1+e^-x) = 1 - 1 / (1+e^x) = 1 - (e^-softplus)
            ddts = ddts - ddts * (-dts).exp()

        ddelta_bias = None 
        if delta_bias is not None:
            ddelta_bias = ddts.sum([0, 1]).view(-1)
        
        if DEBUG:
            print("f", (torch.autograd.grad(_oys, us, doys, create_graph=True, allow_unused=True)[0] - dus).abs().sum(), flush=True)
            print("f", (torch.autograd.grad(_oys, Bs, doys, create_graph=True, allow_unused=True)[0] - dBs).abs().sum(), flush=True)
            print("f", (torch.autograd.grad(_oys, Cs, doys, create_graph=True, allow_unused=True)[0] - dCs).abs().sum(), flush=True)
            print("f", (torch.autograd.grad(_oys, Ds, doys, create_graph=True, allow_unused=True)[0].view(-1) - dDs).abs().sum(), flush=True)
            print("f", (torch.autograd.grad(_oys, As, doys, create_graph=True, allow_unused=True)[0] - dAs).abs().sum(), flush=True)
            # print("f", (torch.autograd.grad(_oys, delta_bias, doys, create_graph=True, allow_unused=True)[0] - ddelta_bias).abs().sum(), flush=True)
            
        dus = dus.permute(1, 2, 3, 0).view(B, -1, L)
        ddts = ddts.permute(1, 2, 3, 0).view(B, -1, L)
        dAs = dAs.view(-1, N)
        dBs = dBs.permute(1, 2, 3, 0)
        dCs = dCs.permute(1, 2, 3, 0)
        if ctx.BC_squeeze[0]:
            dBs = dBs.flatten(1, 2)
        if ctx.BC_squeeze[1]:
            dCs = dCs.flatten(1, 2)

        return dus, ddts, dAs, dBs, dCs, dDs, ddelta_bias, None, None, None