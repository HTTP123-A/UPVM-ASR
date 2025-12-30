# utils/whisper_asr.py

from pathlib import Path
from typing import Tuple, List
import torch
import whisper
from tqdm import tqdm
from jiwer import wer, cer, transforms as tr


# Text normalization (same rules for both GT + Whisper output)
_normalizer = tr.Compose([
    tr.ToLowerCase(),
    tr.RemovePunctuation(),
    tr.RemoveMultipleSpaces(),
    tr.Strip(),
])


def _n(x: str) -> str:
    return _normalizer(x)


def _load_whisper(model_name="large", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_name, device=device)
    fp16 = device.startswith("cuda")
    return model, device, fp16


def _load_gt_text(vctk_root: Path, file_id: str) -> str:
    """
    Example:
        file_id = "p226_001"
        → locate   VCTK/txt/p226/p226_001.txt
    """
    spk = file_id.split("_")[0]
    txt_path = vctk_root / "txt" / spk / f"{file_id}.txt"
    if not txt_path.exists():
        return ""
    return txt_path.read_text().strip()


def compute_folder_whisper_cer_wer(
    folder: str | Path,
    vctk_root: str | Path,
    model_name: str = "large",
    device: str | None = None,
) -> dict:
    """
    Computes: baseline CER/WER and model CER/WER.

    Returns dictionary:
        {
            "cer": value,
            "wer": value,
            "cer_baseline": value,
            "wer_baseline": value
        }
    """
    folder = Path(folder)
    vctk_root = Path(vctk_root)

    # Identify all evaluation pairs
    pairs = []
    for orig_path in folder.glob("*_orig.wav"):
        stem = orig_path.name[:-9]  # remove "_orig.wav"
        up_path = folder / f"{stem}_up.wav"
        if up_path.exists():
            pairs.append((stem, orig_path, up_path))

    if not pairs:
        return {
            "cer": 0.0,
            "wer": 0.0,
            "cer_baseline": 0.0,
            "wer_baseline": 0.0,
        }

    model, device, fp16 = _load_whisper(model_name, device)

    refs = []
    hyps = []
    hyps_clean = []  # baseline

    for stem, orig_wav, up_wav in tqdm(pairs, desc="Whisper CER/WER", ncols=100):

        # GT text from VCTK
        gt_text = _n(_load_gt_text(vctk_root, stem))

        # Whisper transcription of clean audio
        clean_res = model.transcribe(str(orig_wav), fp16=fp16)
        clean_text = _n(clean_res.get("text", ""))

        # Whisper transcription of model output
        up_res = model.transcribe(str(up_wav), fp16=fp16)
        up_text = _n(up_res.get("text", ""))

        refs.append(gt_text)
        hyps_clean.append(clean_text)
        hyps.append(up_text)

    return {
        "cer": float(cer(refs, hyps)),
        "wer": float(wer(refs, hyps)),
        "cer_baseline": float(cer(refs, hyps_clean)),
        "wer_baseline": float(wer(refs, hyps_clean)),
    }
