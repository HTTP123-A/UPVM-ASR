# utils/visqol_cli.py

import subprocess
from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm

import torchaudio  # NEW: for resampling to 48 kHz

# Project root: .../VM-ASR
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Paths to the visqol binary + model inside your project
VISQOL_BIN = PROJECT_ROOT / "visqol" / "bazel-bin" / "visqol"
MODEL_FILE = PROJECT_ROOT / "visqol" / "model" / "libsvm_nu_svr_model.txt"

TARGET_SR = 48000  # ViSQOL audio model expects 48 kHz


def _run_one_pair(args):
    """Worker function for multiprocessing."""
    ref_wav, deg_wav = args

    cmd = [
        str(VISQOL_BIN),
        "--reference_file", str(ref_wav),
        "--degraded_file", str(deg_wav),
        "--similarity_to_quality_model", str(MODEL_FILE),
    ]

    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
        # Debug if you want:
        # print("OUTPUT:", out)
    except subprocess.CalledProcessError:
        # Any failure in ViSQOL => treat as invalid sample
        return None

    for line in out.splitlines():
        if "MOS-LQO" in line:
            try:
                return float(line.split(":")[1].strip())
            except ValueError:
                return None

    return None


def _prepare_48k_pair(orig_path: Path, up_path: Path, tmp_dir: Path) -> tuple[Path, Path]:
    """
    Ensure both files are 48 kHz. If not, resample and save into tmp_dir.

    Returns:
        (ref_path_48k, deg_path_48k)
    """
    info_orig = torchaudio.info(str(orig_path))
    info_up = torchaudio.info(str(up_path))

    # If both already 48k, just use them directly
    if info_orig.sample_rate == TARGET_SR and info_up.sample_rate == TARGET_SR:
        return orig_path, up_path

    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Load audio
    wav_orig, sr_orig = torchaudio.load(str(orig_path))
    wav_up, sr_up = torchaudio.load(str(up_path))

    # Resample each to 48k if needed
    if sr_orig != TARGET_SR:
        resampler_o = torchaudio.transforms.Resample(orig_freq=sr_orig, new_freq=TARGET_SR)
        wav_orig = resampler_o(wav_orig)

    if sr_up != TARGET_SR:
        resampler_u = torchaudio.transforms.Resample(orig_freq=sr_up, new_freq=TARGET_SR)
        wav_up = resampler_u(wav_up)

    # Construct output filenames
    ref_48 = tmp_dir / f"{orig_path.stem}_48k.wav"
    deg_48 = tmp_dir / f"{up_path.stem}_48k.wav"

    # Save as 16-bit PCM WAVs
    torchaudio.save(str(ref_48), wav_orig, TARGET_SR, bits_per_sample=16)
    torchaudio.save(str(deg_48), wav_up, TARGET_SR, bits_per_sample=16)

    return ref_48, deg_48


def compute_folder_average_visqol(folder: str | Path, num_workers=28) -> float:
    """
    Parallel ViSQOL computation over all *_orig.wav / *_up.wav pairs.
    If inputs are not 48kHz, they are resampled to 48kHz before calling ViSQOL.
    Shows tqdm progress bar.
    """
    folder = Path(folder)

    # Temporary dir to hold resampled 48 kHz wavs
    tmp_dir = folder / "visqol_48k"

    # Build task list: [(orig_48k.wav, up_48k.wav), ...]
    tasks = []
    for orig_path in folder.glob("*_orig.wav"):
        stem = orig_path.name[:-9]  # strip '_orig.wav'
        up_path = folder / f"{stem}_up.wav"
        if up_path.exists():
            # Optional debug
            # print(f"Pair found: {orig_path} <> {up_path}")
            ref_48, deg_48 = _prepare_48k_pair(orig_path, up_path, tmp_dir)
            tasks.append((ref_48, deg_48))

    if not tasks:
        print("No *_orig.wav / *_up.wav pairs found for ViSQOL.")
        return 0.0

    # Multiprocessing pool
    with mp.Pool(num_workers) as pool:
        results = list(
            tqdm(
                pool.imap(_run_one_pair, tasks),
                total=len(tasks),
                desc="Computing ViSQOL (parallel)",
                ncols=100,
            )
        )

    # Remove None values
    valid = [r for r in results if r is not None]
    if not valid:
        print("WARNING: All ViSQOL calls failed or returned no MOS-LQO.")
        return 0.0

    return float(sum(valid) / len(valid))
