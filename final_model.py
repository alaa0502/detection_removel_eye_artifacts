# final_model.py
# Lightweight blink-artifact cleaner for EDF using Fp1/Fp2-driven detection.
# - Detection: robust z-score per sliding window, mapped to [0,1] via sigmoid
# - Logic: "either" | "both" | "single"
# - Optional per-recording calibration: raises threshold on noisy files
# - Cleaning: linear inpainting of flagged segments (all channels)
# - Export: EDF via mne.export.export_raw (requires edfio or pyedflib)

from __future__ import annotations
import os
import re
import math
import warnings
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import mne

# ------------------------------
# Config (tweak if you like)
# ------------------------------
DEFAULT_WINDOW_SEC = 1.0      # 1-second windows (used in your experiments)
DEFAULT_OVERLAP    = 0.75     # 75% overlap (denser detection)
BLINK_Z_REF        = 6.0      # z where sigmoid(score)=0.5 (≈ your training label cut)
SIGMOID_SLOPE_K    = 1.2      # steeper -> sharper transition around BLINK_Z_REF
CALIB_FP_RATE      = 0.005    # ~target max fraction flagged by noise (0.5%)

# ------------------------------
# Small utilities
# ------------------------------
def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))

def _zscore_peak_feature(w: np.ndarray) -> float:
    """Robust 'peakiness' in window: (max|x| - mean) / std with floor to avoid div/0."""
    mu = np.mean(w)
    sd = np.std(w)
    if sd <= 1e-9:
        sd = 1e-9
    return float((np.max(np.abs(w)) - mu) / sd)

def _robust_uv(data: np.ndarray) -> np.ndarray:
    """Convert Volts to microvolts if needed. If already looks like µV, no change."""
    # Heuristic: typical EEG ~ tens of µV. If median absolute < 1e-4 V => convert to µV.
    med_abs = np.median(np.abs(data))
    if med_abs < 1e-4:  # likely volts
        return data * 1e6
    return data

def _pick_by_alias(raw: mne.io.BaseRaw, targets: List[str]) -> Optional[int]:
    """
    Return the first matching channel index for any alias in `targets`.
    Matching is case-insensitive and ignores non-alphanumerics (., -, ref).
    """
    def norm(s: str) -> str:
        return re.sub(r'[^a-z0-9]+', '', s.lower())

    names = [norm(n) for n in raw.ch_names]
    for t in targets:
        nt = norm(t)
        for i, n in enumerate(names):
            if nt == n:
                return i
    # fuzzy: try startswith (e.g., fp1ref)
    for t in targets:
        nt = norm(t)
        for i, n in enumerate(names):
            if n.startswith(nt):
                return i
    return None

def _find_fp_indices(raw: mne.io.BaseRaw) -> Tuple[Optional[int], Optional[int]]:
    # Common aliases across datasets (with/without trailing dot / ref)
    fp1_aliases = ["fp1", "fp1.", "fp1-ref", "fp1ref", "fp1f7", "afp1", "fp1a1"]
    fp2_aliases = ["fp2", "fp2.", "fp2-ref", "fp2ref", "fp2f8", "afp2", "fp2a2"]
    i1 = _pick_by_alias(raw, fp1_aliases)
    i2 = _pick_by_alias(raw, fp2_aliases)
    return i1, i2

@dataclass
class WindowGrid:
    starts: np.ndarray  # sample indices for window starts
    size: int           # window size in samples

def _make_window_grid(n_samples: int, sfreq: float,
                      window_sec: float, overlap: float) -> WindowGrid:
    win = int(round(window_sec * sfreq))
    win = max(4, win)
    hop = max(1, int(round(win * (1.0 - overlap))))
    starts = np.arange(0, max(0, n_samples - win + 1), hop, dtype=int)
    return WindowGrid(starts=starts, size=win)

def _combine_logic(z1: float, z2: Optional[float], logic: str) -> float:
    """
    Combine per-channel z features into a single 'blinkiness' z for the window.
    - either: max(z1,z2)
    - both  : min(z1,z2)  (both must be high)
    - single: z1 only (use whichever channel was computed)
    """
    if logic == "either":
        if z2 is None:  # only one chan available
            return z1
        return max(z1, z2)
    elif logic == "both":
        if z2 is None:  # if missing one channel, fall back to single
            return z1
        return min(z1, z2)
    elif logic == "single":
        return z1
    else:
        raise ValueError("logic must be 'either', 'both', or 'single'")

def _z_to_score(z: np.ndarray, z_ref: float = BLINK_Z_REF, k: float = SIGMOID_SLOPE_K) -> np.ndarray:
    """
    Map z (unbounded) to [0,1] so UI thresholds (0.25/0.35/0.5) are meaningful.
    score = sigmoid((z - z_ref)/k) → score=0.5 at z=z_ref.
    """
    return _sigmoid((z - float(z_ref)) / float(k))

def calibrate_tau_from_file(window_scores: np.ndarray,
                            user_tau: float,
                            target_fp_rate: float = CALIB_FP_RATE) -> float:
    """
    Per-recording calibration: raise tau if the file looks noisier than usual.
    tau_file = max(user_tau, percentile_{(1 - target_fp_rate)}(scores))
    """
    if window_scores.size == 0:
        return float(user_tau)
    tau_p = float(np.percentile(window_scores, 100.0 * (1.0 - float(target_fp_rate))))
    return float(max(user_tau, tau_p))

def _labels_from_scores(scores: np.ndarray, tau: float) -> np.ndarray:
    return (scores >= float(tau)).astype(np.int8)

def _merge_windows_to_segments(starts: np.ndarray, size: int, labels: np.ndarray) -> List[Tuple[int, int]]:
    """
    Merge flagged windows (labels==1) into continuous [start,end) sample segments.
    """
    segs: List[Tuple[int, int]] = []
    if labels.size == 0:
        return segs
    current = None
    for s, lab in zip(starts, labels):
        if lab == 1 and current is None:
            current = [int(s), int(s + size)]
        elif lab == 1 and current is not None:
            # extend if overlapping/touching
            current[1] = max(current[1], int(s + size))
        elif lab == 0 and current is not None:
            segs.append((current[start := 0], current[1]))
            current = None
    if current is not None:
        segs.append((current[0], current[1]))
    # coalesce any accidental overlaps
    if not segs:
        return segs
    segs.sort()
    merged = [segs[0]]
    for a, b in segs[1:]:
        la, lb = merged[-1]
        if a <= lb:
            merged[-1] = (la, max(lb, b))
        else:
            merged.append((a, b))
    return merged

def _linear_inpaint(raw: mne.io.BaseRaw, segments: List[Tuple[int, int]], taper_frac: float = 0.1) -> None:
    """
    Inpaint raw._data in-place by linear interpolation over each [s,e) segment (all channels).
    Add a short cosine taper into/out of the segment to avoid sharp edges.
    """
    if not segments:
        return
    data = raw.get_data()  # (n_channels, n_samples), float64
    n_ch, n = data.shape
    for s, e in segments:
        s = max(0, min(int(s), n - 1))
        e = max(s + 1, min(int(e), n))

        left_i  = max(0, s - 1)
        right_i = min(n - 1, e)

        left_val  = data[:, left_i][:, None]      # (n_ch,1)
        right_val = data[:, right_i][:, None]     # (n_ch,1)

        L = e - s
        if L <= 1:
            # single-sample hole -> copy nearest
            data[:, s:e] = left_val
            continue

        # linear ramp
        t = np.linspace(0.0, 1.0, L, endpoint=False)[None, :]  # (1,L)
        interp = (1.0 - t) * left_val + t * right_val          # (n_ch, L)

        # cosine taper at edges
        tf = max(1, int(round(taper_frac * L)))
        if tf > 0 and L >= 2*tf:
            taper_in  = 0.5 * (1 - np.cos(np.linspace(0, np.pi, tf)))
            taper_out = taper_in[::-1]
            # blend with original to be conservative near edges
            interp[:, :tf]  = taper_in  * interp[:, :tf]  + (1 - taper_in)  * data[:, s:s+tf]
            interp[:, -tf:] = taper_out * interp[:, -tf:] + (1 - taper_out) * data[:, e-tf:e]

        data[:, s:e] = interp

    raw._data = data  # assign back (already view)

def _export_edf(raw: mne.io.BaseRaw, out_path: str) -> str:
    """
    Export EDF using MNE. Requires edfio >= 0.4 (preferred) or pyedflib.
    """
    # Ensure directory exists
    d = os.path.dirname(out_path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

    try:
        # mne 1.5+ supports fmt='edf' via edfio (preferred) or pyedflib
        mne.export.export_raw(out_path, raw, fmt='edf', physical_range=(-100e-6, 100e-6))
        return out_path
    except Exception as e:
        raise RuntimeError(
            "EDF export failed. Ensure 'edfio' (preferred) or 'pyedflib' is installed.\n"
            f"Original error: {e}"
        )

# ------------------------------
# Main API
# ------------------------------
def clean_edf_file(in_path: str,
                   out_path: str,
                   logic: str = "either",
                   des_thresh: float = 0.35,
                   window_sec: float = DEFAULT_WINDOW_SEC,
                   overlap: float = DEFAULT_OVERLAP,
                   calibrate: bool = True) -> Tuple[str, int, float]:
    """
    Clean blink artifacts in an EDF file guided by Fp1/Fp2.

    Parameters
    ----------
    in_path : str
        Input EDF filepath.
    out_path : str
        Output EDF filepath.
    logic : {"either","both","single"}
        Channel logic for detection.
    des_thresh : float
        User decision threshold in [0,1] (e.g., 0.25 / 0.35 / 0.50).
    window_sec : float
        Sliding window length (seconds). Default 1.0 s.
    overlap : float
        Fractional overlap [0..1). Default 0.75 (75%).
    calibrate : bool
        If True, raise tau per recording using percentile rule.

    Returns
    -------
    saved_path : str
        Path to cleaned EDF.
    n_segments : int
        Number of merged blink segments removed.
    total_masked_sec : float
        Total seconds inpainted.
    """
    if logic not in {"either", "both", "single"}:
        raise ValueError("logic must be one of: 'either', 'both', 'single'")
    des_thresh = float(des_thresh)

    # 1) Load EDF
    raw = mne.io.read_raw_edf(in_path, preload=True, verbose=False)
    sfreq = float(raw.info["sfreq"])
    n_samples = raw.n_times

    # 2) Find Fp channels (robust aliases)
    i_fp1, i_fp2 = _find_fp_indices(raw)

    if i_fp1 is None and i_fp2 is None:
        raise RuntimeError(
            "Could not find Fp1 or Fp2 channels. Please ensure at least one frontal lead is present "
            "(aliases like 'Fp1', 'Fp1.', 'Fp2', 'Fp2.')."
        )

    # For 'single' logic we only need one; for 'both' we prefer both present.
    picks_for_detection: List[int] = []
    if logic == "single":
        # prefer Fp1 if present, else Fp2
        if i_fp1 is not None:
            picks_for_detection = [i_fp1]
        else:
            picks_for_detection = [i_fp2]
    else:
        # either/both: use whatever is available
        if i_fp1 is not None:
            picks_for_detection.append(i_fp1)
        if i_fp2 is not None:
            picks_for_detection.append(i_fp2)

    # 3) Get data (µV) for detection channels
    det_data = raw.get_data(picks=picks_for_detection, reject_by_annotation="omit")
    det_data = _robust_uv(det_data)  # shape: (n_det, n_samples)

    # 4) Build window grid
    grid = _make_window_grid(n_samples, sfreq, window_sec, overlap)

    # 5) Compute per-window z features and combine via logic
    zs = []
    for s in grid.starts:
        e = s + grid.size
        # handle edges
        if e > n_samples:
            e = n_samples
        # compute per-channel z feature for this window
        if det_data.shape[0] == 1:
            z1 = _zscore_peak_feature(det_data[0, s:e])
            zc = _combine_logic(z1, None, logic)
        else:
            z1 = _zscore_peak_feature(det_data[0, s:e])
            z2 = _zscore_peak_feature(det_data[1, s:e])
            zc = _combine_logic(z1, z2, logic)
        zs.append(zc)

    zs = np.asarray(zs, dtype=float)  # unbounded
    scores = _z_to_score(zs, z_ref=BLINK_Z_REF, k=SIGMOID_SLOPE_K)  # [0,1]

    # 6) Per-recording calibration (optional)
    tau = float(des_thresh)
    if calibrate:
        tau = calibrate_tau_from_file(scores, user_tau=des_thresh, target_fp_rate=CALIB_FP_RATE)

    # 7) Threshold and merge windows into segments (samples)
    labels = _labels_from_scores(scores, tau=tau)
    segments = _merge_windows_to_segments(grid.starts, grid.size, labels)

    # 8) Inpaint across ALL channels for each segment
    _linear_inpaint(raw, segments, taper_frac=0.1)

    # 9) Export EDF
    saved_path = _export_edf(raw, out_path)

    # 10) Stats
    total_masked_sec = 0.0
    if segments:
        total_masked_sec = sum((e - s) / sfreq for s, e in segments)
    n_segments = len(segments)

    return saved_path, n_segments, float(total_masked_sec)

# ------------------------------
# (Optional) Helpers you may want
# ------------------------------
def set_decision_threshold(obj, thr: float):
    """
    Attach a decision threshold attribute to a model-like object (e.g., RF),
    kept for compatibility with earlier experiments.
    """
    setattr(obj, "_decision_threshold", float(thr))
    return obj
