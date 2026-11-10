#!/usr/bin/env python3
"""
Window-level feature extraction for the Shimmer3 dehydration dataset
(Sabry et al., 2022 -- zenodo.org/records/7099266).

Usage:
    python extract_features.py path/to/shimmer-raw-dataset-dehydration
    python extract_features.py path/to/data --window 60 --out features_export

Walks every subject folder (S1, S2, ...), reads each raw Shimmer CSV,
selects calibrated channels BY NAME (not by column position, since the
column order can differ between devices/sessions), cuts the recording into
fixed windows, and writes one row of features per window.

Outputs (in --out, default ./features_export):
    features.csv       one row per window, all subjects
    manifest.csv       one row per raw file: rows, duration, sampling rate, channels, errors
    fasting_times/     copy of every fastingTimes.csv (renamed <subject>_fastingTimes.csv)
    summary.txt        dataset-level summary
Zip the output folder and share it -- it is small (a few MB at most).

Requires: numpy, pandas, scipy.
"""
import argparse
import re
import shutil
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

# ---------------------------------------------------------------------------
# Channel selection
# ---------------------------------------------------------------------------
# canonical name -> list of substrings to match against the sensor-name row
# (first calibrated match wins)
CHANNELS = {
    "ppg": ["PPG_A13", "PPG"],
    "acc_x": ["Accel_LN_X", "Accel_WR_X", "Accel_X"],
    "acc_y": ["Accel_LN_Y", "Accel_WR_Y", "Accel_Y"],
    "acc_z": ["Accel_LN_Z", "Accel_WR_Z", "Accel_Z"],
    "gyro_x": ["Gyro_X"],
    "gyro_y": ["Gyro_Y"],
    "gyro_z": ["Gyro_Z"],
    "temp": ["Temperature_BMP280", "Temperature"],
    "pressure": ["Pressure_BMP280", "Pressure"],
    "gsr_cond": ["GSR_Skin_Conductance"],
    "gsr_res": ["GSR_Skin_Resistance"],
    "ts_ms": ["Timestamp"],  # must not match System_Timestamp_Plot_Zeroed (handled below)
}


def read_header(path, max_lines=12):
    """Find the sensor-name / CAL-UNCAL / units rows. Returns (names, cal, units, data_start_row)."""
    raw = pd.read_csv(path, header=None, nrows=max_lines, dtype=str,
                      keep_default_na=False, on_bad_lines="skip", engine="python")
    for i in range(len(raw)):
        row = raw.iloc[i].astype(str).str.strip().str.upper()
        if row.isin(["CAL", "UNCAL"]).sum() >= 3:
            names = raw.iloc[i - 1].astype(str).str.strip().tolist()
            cal = raw.iloc[i].astype(str).str.strip().str.upper().tolist()
            units = raw.iloc[i + 1].astype(str).str.strip().tolist()
            return names, cal, units, i + 2
    raise ValueError("could not find CAL/UNCAL header row")


def pick_columns(names, cal, units):
    chosen = {}
    for key, patterns in CHANNELS.items():
        found = None
        for pat in patterns:
            for idx, (n, c) in enumerate(zip(names, cal)):
                if c != "CAL" or pat.lower() not in n.lower():
                    continue
                if key == "ts_ms" and ("system" in n.lower() or "plot" in n.lower()):
                    continue
                if key == "gsr_cond" and units[idx].lower() not in ("microsiemens", "us", "µs"):
                    continue
                found = idx
                break
            if found is not None:
                break
        if found is not None:
            chosen[key] = found
    return chosen


def parse_start_time(fname):
    m = re.match(r"(\d{14})", fname)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------
def slope_per_min(t_s, x):
    ok = np.isfinite(x)
    if ok.sum() < 10:
        return np.nan
    return np.polyfit(t_s[ok], x[ok], 1)[0] * 60.0


def bandpass(x, fs, lo, hi, order=3):
    nyq = fs / 2.0
    hi = min(hi, nyq * 0.95)
    b, a = signal.butter(order, [lo / nyq, hi / nyq], btype="band")
    return signal.filtfilt(b, a, x)


def lowpass(x, fs, hi, order=3):
    nyq = fs / 2.0
    b, a = signal.butter(order, min(hi / nyq, 0.95), btype="low")
    return signal.filtfilt(b, a, x)


def ppg_features(x, fs):
    out = dict(hr_mean=np.nan, hr_std=np.nan, ibi_sdnn_ms=np.nan, ibi_rmssd_ms=np.nan,
               n_beats=0, ppg_quality=np.nan)
    x = x[np.isfinite(x)]
    if len(x) < fs * 10:
        return out
    try:
        f = bandpass(x - np.mean(x), fs, 0.5, 4.0)
    except ValueError:
        return out
    # Shimmer optical pulse is often inverted; use the polarity with sharper peaks
    best = None
    for sgn in (1, -1):
        pk, _ = signal.find_peaks(sgn * f, distance=int(0.33 * fs),
                                  prominence=0.3 * np.std(f))
        if len(pk) < 3:
            continue
        ibi = np.diff(pk) / fs
        good = (ibi > 0.33) & (ibi < 1.5)
        q = good.mean()
        if best is None or q > best[0]:
            best = (q, ibi[good], len(pk))
    if best is None:
        return out
    q, ibi, n = best
    out["n_beats"] = n
    out["ppg_quality"] = q
    if len(ibi) >= 3:
        hr = 60.0 / ibi
        out["hr_mean"] = hr.mean()
        out["hr_std"] = hr.std()
        out["ibi_sdnn_ms"] = ibi.std() * 1000
        out["ibi_rmssd_ms"] = np.sqrt(np.mean(np.diff(ibi) ** 2)) * 1000
    return out


def gsr_features(cond, res, t_s, fs):
    out = {}
    c = cond[np.isfinite(cond)]
    out["gsr_mean"] = np.mean(c) if len(c) else np.nan
    out["gsr_std"] = np.std(c) if len(c) else np.nan
    out["gsr_min"] = np.min(c) if len(c) else np.nan
    out["gsr_max"] = np.max(c) if len(c) else np.nan
    out["gsr_slope_per_min"] = slope_per_min(t_s, cond)
    # Shimmer3 GSR+ accurate range is ~10 kOhm - 4.7 MOhm; flag samples outside it
    r = res[np.isfinite(res)] if res is not None else np.array([])
    out["gsr_out_of_range_frac"] = float(np.mean((r > 4700) | (r < 10))) if len(r) else np.nan
    # skin conductance responses (phasic peaks)
    out["scr_count"] = np.nan
    out["scr_amp_mean"] = np.nan
    if len(c) > fs * 10:
        try:
            sm = lowpass(c, fs, 1.0)
            tonic = signal.medfilt(sm, kernel_size=int(fs * 8) | 1) if len(sm) > fs * 8 else sm
            phasic = sm - tonic
            pk, props = signal.find_peaks(phasic, prominence=max(0.01, 0.05 * np.std(c) + 1e-9),
                                          distance=int(fs * 1.0))
            out["scr_count"] = len(pk)
            out["scr_amp_mean"] = float(np.mean(props["prominences"])) if len(pk) else 0.0
            out["gsr_tonic_mean"] = float(np.mean(tonic))
        except ValueError:
            pass
    return out


def window_features(w, fs):
    t_s = w["t_s"].to_numpy()
    feats = {"n_samples": len(w)}
    if "ppg" in w:
        p = w["ppg"].to_numpy(float)
        feats["ppg_mean"] = np.nanmean(p)
        feats["ppg_std"] = np.nanstd(p)
        feats.update(ppg_features(p, fs))
    if "gsr_cond" in w:
        feats.update(gsr_features(w["gsr_cond"].to_numpy(float),
                                  w["gsr_res"].to_numpy(float) if "gsr_res" in w else None,
                                  t_s, fs))
    if "temp" in w:
        tt = w["temp"].to_numpy(float)
        feats["temp_mean"] = np.nanmean(tt)
        feats["temp_std"] = np.nanstd(tt)
        feats["temp_slope_per_min"] = slope_per_min(t_s, tt)
    if {"acc_x", "acc_y", "acc_z"} <= set(w.columns):
        a = np.sqrt(w["acc_x"] ** 2 + w["acc_y"] ** 2 + w["acc_z"] ** 2).to_numpy(float)
        feats["acc_mag_mean"] = np.nanmean(a)
        feats["acc_mag_std"] = np.nanstd(a)
        feats["acc_activity"] = np.nanmean(np.abs(a - np.nanmedian(a)))
    if {"gyro_x", "gyro_y", "gyro_z"} <= set(w.columns):
        g = np.sqrt(w["gyro_x"] ** 2 + w["gyro_y"] ** 2 + w["gyro_z"] ** 2).to_numpy(float)
        feats["gyro_mag_mean"] = np.nanmean(g)
        feats["gyro_mag_std"] = np.nanstd(g)
    if "pressure" in w:
        feats["pressure_mean"] = np.nanmean(w["pressure"].to_numpy(float))
    feats["missing_frac"] = float(w.drop(columns=["t_s"]).isna().mean().mean())
    return feats


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------
def process_file(path, subject, condition, window_s, min_frac):
    names, cal, units, start_row = read_header(path)
    chosen = pick_columns(names, cal, units)
    if "ts_ms" not in chosen:
        raise ValueError("no calibrated Timestamp column")
    cols = sorted(chosen.values())
    df = pd.read_csv(path, header=None, skiprows=start_row, usecols=cols,
                     low_memory=False, on_bad_lines="skip")
    inv = {v: k for k, v in chosen.items()}
    df.columns = [inv[c] for c in cols]
    df = df.apply(pd.to_numeric, errors="coerce").dropna(subset=["ts_ms"])
    df = df.sort_values("ts_ms").drop_duplicates("ts_ms")
    if len(df) < 100:
        raise ValueError(f"too few rows ({len(df)})")

    t = (df["ts_ms"].to_numpy() - df["ts_ms"].iloc[0]) / 1000.0
    dt = np.diff(t)
    fs = 1.0 / np.median(dt[dt > 0])
    gaps = int(np.sum(dt > 5 * np.median(dt)))
    df = df.drop(columns=["ts_ms"])
    df["t_s"] = t
    duration_min = t[-1] / 60.0

    start = parse_start_time(path.name)
    rows = []
    df["win"] = (df["t_s"] // window_s).astype(int)
    for win, w in df.groupby("win"):
        if len(w) < min_frac * window_s * fs:
            continue  # drop short trailing windows
        f = window_features(w.drop(columns=["win"]), fs)
        f.update(subject=subject, condition=condition, file=path.name, window=int(win),
                 t_start_s=float(win * window_s),
                 window_start=(start + timedelta(seconds=float(win * window_s))).isoformat()
                 if start else None)
        rows.append(f)

    info = dict(subject=subject, condition=condition, file=path.name,
                rel_path=str(path), rows=len(df), duration_min=round(duration_min, 2),
                fs_hz=round(fs, 2), gaps=gaps, n_windows=len(rows),
                recording_start=start.isoformat() if start else None,
                channels=";".join(f"{k}={names[v]}" for k, v in chosen.items()),
                missing_channels=";".join(k for k in CHANNELS if k not in chosen),
                error="")
    return rows, info


def find_subject(path, root):
    for part in path.relative_to(root).parts:
        if re.fullmatch(r"[Ss]\d+", part):
            return part.upper()
    return path.relative_to(root).parts[0]


def find_condition(path):
    p = str(path).lower()
    if "nonfasting" in p or "notfasting" in p or "non_fasting" in p or "not_fasting" in p:
        return "nonfasting"
    if "fasting" in p:
        return "fasting"
    return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir", type=Path)
    ap.add_argument("--out", type=Path, default=Path("features_export"))
    ap.add_argument("--window", type=float, default=60.0, help="window length in seconds")
    ap.add_argument("--min-frac", type=float, default=0.8,
                    help="drop windows with fewer than this fraction of expected samples")
    args = ap.parse_args()

    root = args.data_dir
    if not root.exists():
        sys.exit(f"{root} does not exist")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "fasting_times").mkdir(exist_ok=True)

    all_csv = sorted(root.rglob("*.csv"))
    fasting_files = [p for p in all_csv if "fastingtime" in p.name.lower()]
    raw_files = [p for p in all_csv if p not in fasting_files]
    print(f"Found {len(raw_files)} recordings and {len(fasting_files)} fastingTimes files")

    for p in fasting_files:
        shutil.copy(p, args.out / "fasting_times" / f"{find_subject(p, root)}_{p.parent.name}_{p.name}")

    feats, manifest = [], []
    for i, p in enumerate(raw_files, 1):
        subj, cond = find_subject(p, root), find_condition(p)
        try:
            rows, info = process_file(p, subj, cond, args.window, args.min_frac)
            feats.extend(rows)
        except Exception as e:  # keep going, record the problem
            info = dict(subject=subj, condition=cond, file=p.name, rel_path=str(p),
                        error=f"{type(e).__name__}: {e}")
            traceback.print_exc(limit=1)
        manifest.append(info)
        print(f"[{i}/{len(raw_files)}] {subj} {cond:10s} {p.name}  "
              f"{info.get('duration_min', '?')} min, {info.get('n_windows', 0)} windows"
              + (f"  ERROR {info['error']}" if info.get("error") else ""))

    fdf = pd.DataFrame(feats)
    lead = ["subject", "condition", "file", "window", "t_start_s", "window_start"]
    fdf = fdf[lead + [c for c in fdf.columns if c not in lead]]
    fdf.to_csv(args.out / "features.csv", index=False)
    mdf = pd.DataFrame(manifest)
    mdf.to_csv(args.out / "manifest.csv", index=False)

    with open(args.out / "summary.txt", "w") as fh:
        fh.write(f"data_dir: {root.resolve()}\n")
        fh.write(f"recordings: {len(raw_files)}, errors: {(mdf['error'] != '').sum()}\n")
        fh.write(f"windows: {len(fdf)} ({args.window:.0f} s each)\n\n")
        if "duration_min" in mdf:
            fh.write("minutes by subject x condition:\n")
            fh.write(mdf.pivot_table(index="subject", columns="condition", values="duration_min",
                                     aggfunc="sum", fill_value=0).round(1).to_string())
            fh.write("\n\nsampling rate (Hz) by subject:\n")
            fh.write(mdf.groupby("subject")["fs_hz"].describe().round(2).to_string())
            fh.write("\n")
    print(f"\nDone. {len(fdf)} windows from {len(raw_files)} files -> {args.out}/")
    print(open(args.out / "summary.txt").read())
    print(f"Zip and share the '{args.out}' folder.")


if __name__ == "__main__":
    main()
