# === Minimal EEG blink-cleaning toolkit for one EDF ===
# drop this in a .py file and import it in your Streamlit app

import os
import numpy as np
import mne

# -------------------------------------------------------
# 1) Windowing identical to training (Fp1/Fp2, same overlap)
# -------------------------------------------------------
def extract_windows_from_raw(raw, channels=('Fp1.', 'Fp2.'), window_sec=1.0, overlap=0.75):
    """Return windows X (n_win, ch, samples), and (starts, win, hop, sfreq)."""
    sfreq = int(raw.info['sfreq'])
    win = int(sfreq * window_sec)
    hop = max(1, int(win * (1 - overlap)))

    picks = mne.pick_channels(raw.ch_names, include=list(channels), ordered=True)
    if len(picks) != len(channels):
        raise ValueError(f"Missing channels: expected {channels}, got {[raw.ch_names[p] for p in picks]}")

    data = raw.get_data(picks=picks) * 1e6  # µV to match training
    n = data.shape[1]
    starts = list(range(0, n - win + 1, hop))

    X = []
    for s in starts:
        X.append(data[:, s:s+win])
    X = np.asarray(X)  # (n_win, ch, samples)
    return X, starts, win, hop, sfreq

# -------------------------------------------------------
# 2) Blink prediction: RF model (preferred) or z-score fallback
# -------------------------------------------------------
def predict_blinks(X, clf=None, z_thresh=6.0, logic='either',des_thresh=0.35):
    """
    If clf (sklearn RF) is provided, uses it. Else uses z-score rule (same idea you trained with).
    Returns y_pred (0/1 per window) and y_proba (if clf given).
    """
    y_pred = None
    y_proba = None

    if clf is not None:
        Xf = X.reshape(len(X), -1)
        y_proba = clf.predict_proba(Xf)[:, 1]
        # Default threshold; you can expose this in the UI or load your F1-optimal one
        thr = getattr(clf, "_decision_threshold", des_thresh)  # set on your clf if you like
        y_pred = (y_proba >= thr).astype(int)
    else:
        # fallback: z-score rule window-wise (max abs amplitude std-normalized)
        y_pred = []
        for w in X:  # (ch, samples)
            z_scores = []
            for ch in w:
                sd = np.std(ch) or 1e-9
                z_scores.append((np.max(np.abs(ch)) - np.mean(ch)) / sd)
            if logic == 'either':
                label = 1 if max(z_scores) > z_thresh else 0
            elif logic == 'both':
                label = 1 if sum(z_scores) > z_thresh else 0
            elif logic == 'single':
                label = 1 if z_scores[0] > z_thresh else 0
            else:
                raise ValueError("logic must be 'either', 'both', or 'single'")
            y_pred.append(label)
        y_pred = np.asarray(y_pred, dtype=int)

    return y_pred, y_proba

# -------------------------------------------------------
# 3) Convert window predictions to merged sample segments
# -------------------------------------------------------
def windows_to_segments(y_pred, starts, win, hop, min_gap_windows=1, pad_sec=0.05, sfreq=256):
    """
    Merge adjacent/nearby positive windows into continuous sample segments.
    min_gap_windows=1 merges windows separated by ≤1 hop of zeros.
    pad_sec expands each segment on both ends to be safe.
    Returns list of (s_start, s_end) sample indices inclusive-exclusive.
    """
    pos_idx = np.where(y_pred == 1)[0]
    if len(pos_idx) == 0:
        return []

    segments = []
    cur_start = pos_idx[0]
    prev = pos_idx[0]
    for i in pos_idx[1:]:
        if i - prev <= (min_gap_windows + 1):  # allow tiny gaps
            prev = i
        else:
            s0 = starts[cur_start]
            s1 = starts[prev] + win
            segments.append((s0, s1))
            cur_start = i
            prev = i
    # last run
    s0 = starts[cur_start]
    s1 = starts[prev] + win
    segments.append((s0, s1))

    # pad
    pad = int(round(pad_sec * sfreq))
    segments = [(max(0, a - pad), b + pad) for (a, b) in segments]
    return segments

# -------------------------------------------------------
# 4) Clean by interpolation inside segments (non-destructive)
# -------------------------------------------------------
def clean_raw_by_interpolation(raw, segments, channels=None):
    """
    Performs linear interpolation over detected segments for specified channels (or all EEG channels).
    Returns a COPY of raw (does not modify the original).
    """
    raw_clean = raw.copy()
    if channels is None:
        picks = mne.pick_types(raw_clean.info, eeg=True)
    else:
        picks = mne.pick_channels(raw_clean.ch_names, include=list(channels), ordered=True)

    data = raw_clean.get_data(picks=picks)
    n = data.shape[1]

    for (a, b) in segments:
        a = max(0, a)
        b = min(n, b)
        if b <= a:
            continue
        left = a - 1
        right = b
        # handle borders: extrapolate using nearest valid value
        for ch in range(data.shape[0]):
            if left < 0 and right >= n:
                # degenerate: entire signal — nothing to interpolate to, skip
                continue
            left_val = data[ch, left] if left >= 0 else data[ch, right]
            right_val = data[ch, right] if right < n else data[ch, left]
            length = b - a
            # linear ramp
            interp = np.linspace(left_val, right_val, num=length, endpoint=False, dtype=float)
            data[ch, a:b] = interp

    raw_clean._data[picks] = data
    return raw_clean

# -------------------------------------------------------
# 5) Save as EDF (uses mne.export.export_raw); falls back to .fif
# -------------------------------------------------------
def save_cleaned(raw_clean, out_path, overwrite=True):
    """Write EDF via MNE+edfio only; error if it can’t."""
    import os, numpy as np, mne
    from mne.export import export_raw

    base, _ = os.path.splitext(out_path)
    out_path = base + ".edf"  # force .edf

    r = raw_clean.copy().load_data()

    # Keep EEG/MISC only (drop stim/MEG/etc.)
    picks = mne.pick_types(r.info, eeg=True, meg=False, stim=False, eog=False, ecg=False, seeg=False, misc=True)
    if len(picks) == 0:
        raise RuntimeError("No EEG/MISC channels to export.")
    r.pick(picks)

    # Sanitize data
    data = r.get_data()
    bad = ~np.isfinite(data)
    if bad.any():
        data[bad] = 0.0
    std = data.std(axis=1)
    const_idx = (std == 0)
    if const_idx.any():
        data[const_idx, 0] += 1e-6
    r._data = data

    # Write EDF (uses edfio under the hood)
    export_raw(out_path, r, fmt="edf", physical_range="auto", overwrite=overwrite)
    return out_path

# -------------------------------------------------------
# 6) High-level wrapper: ONE EDF IN -> CLEAN EDF OUT
# -------------------------------------------------------
def clean_edf_file(
    in_path,
    out_path,
    clf=None,               # your trained RandomForestClassifier (optional)
    channels=('Fp1.', 'Fp2.'),
    window_sec=1.0,
    overlap=0.75,
    z_thresh=6.0,
    logic='either',
    min_gap_windows=1,
    pad_sec=0.05,
    des_thresh=0.35
):

    """
    Full pipeline to clean one EDF:
    - read EDF
    - window -> predict -> segments
    - interpolate segments on selected channels
    - save cleaned file
    Returns (saved_path, n_segments, total_masked_sec)
    """
    raw = mne.io.read_raw_edf(in_path, preload=True, verbose=False)

    # (optional) basic filter you can enable if desired:
    # raw.filter(0.1, 40., fir_design='firwin', verbose=False)

    X, starts, win, hop, sfreq = extract_windows_from_raw(raw, channels, window_sec, overlap)
    y_pred, _ = predict_blinks(X, clf=clf, z_thresh=z_thresh, logic=logic,des_thresh=des_thresh)
    segments = windows_to_segments(y_pred, starts, win, hop, min_gap_windows=min_gap_windows, pad_sec=pad_sec, sfreq=sfreq)

    raw_clean = clean_raw_by_interpolation(raw, segments, channels=channels)
    saved_path = save_cleaned(raw_clean, out_path, overwrite=True)

    total_masked_samples = sum(max(0, b - a) for (a, b) in segments)
    total_masked_sec = total_masked_samples / float(sfreq)
    return saved_path, len(segments), total_masked_sec

# -------------------------------------------------------
# (Optional) tiny helper to attach your best threshold to the clf
# -------------------------------------------------------
def set_decision_threshold(clf, thr=0.35):
    """
    Store your F1-optimal threshold on the classifier so predict_blinks() can use it.
    """
    setattr(clf, "_decision_threshold", float(thr))
    return clf

# saved_path, n_segments, total_masked_sec = clean_edf_file(
#     in_path="C:/Users/shaim/.vscode/alaa/database_1/S001R01.edf",
#     out_path="subject01_clean.edf"
#     # no clf passed → falls back to z-threshold=6.0 rule
# )
# print("Saved:", saved_path)

# raw = mne.io.read_raw_edf("S001R01_clean.edf", preload=True, verbose=False)
# raw.plot(block=True, scalings="auto", title=" downloaded cleand S001R01 EDF")
