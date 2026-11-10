#!/usr/bin/env python3
"""
Modeling for personalized dehydration detection.

Input : features_export/ produced by extract_features.py
Output: results/ (metrics.json, per-window predictions, figures)

Usage:
    python analysis.py features_export --out results [--threshold-h 8]

Label
-----
hours_since_drink = window_start - fasting start   (fasting recordings; from fastingTimes.csv)
                  = 1.0                            (non-fasting recordings; per dataset description)
Binary target "deficit" = hours_since_drink >= threshold (default 8 h).

Evaluation (no window-level leakage: splits are by subject or by recording)
---------------------------------------------------------------------------
* General model       : leave-one-subject-out (LOSO), raw features.
* Personalized model  : LOSO, but every subject's features are expressed relative to
                        their own hydrated baseline (first non-fasting recording =
                        "calibration session"). Calibration recordings are excluded
                        from scoring for ALL models so the comparison is fair.
* Within-subject model: leave-one-recording-out inside a single subject (S1 has most data).
* Clock baseline      : time-of-day only -- the model has to beat this, because long
                        fasts always happen late in the day.
* Modality ablation   : PPG / GSR / temperature / motion alone.
* False alarms        : false-positive rate on hydrated windows vs. activity level.
"""
import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, f1_score, mean_absolute_error,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
RNG = 0

MODALITIES = {
    "PPG": ["ppg_mean", "ppg_std", "hr_mean", "hr_std", "ibi_sdnn_ms", "ibi_rmssd_ms"],
    "GSR": ["gsr_mean", "gsr_std", "gsr_min", "gsr_max", "gsr_slope_per_min", "scr_count",
            "scr_amp_mean", "gsr_tonic_mean"],
    "Temp": ["temp_mean", "temp_std", "temp_slope_per_min"],
    "Motion": ["acc_mag_mean", "acc_mag_std", "acc_activity", "gyro_mag_mean", "gyro_mag_std"],
}


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
def parse_fasting_times(export_dir):
    """Return DataFrame(subject, key, fast_start: Timestamp).

    fastingTimes.csv layout is not documented; this tries the plausible layouts:
    a column holding a recording file name / date and a column holding the start
    time (time-of-day or full datetime). Adjust here once the real files are seen.
    """
    rows = []
    for p in sorted((export_dir / "fasting_times").glob("*.csv")):
        subj = p.name.split("_")[0]
        df = pd.read_csv(p, dtype=str)
        df.columns = [c.strip() for c in df.columns]
        for _, r in df.iterrows():
            vals = [str(v).strip() for v in r.values if pd.notna(v)]
            rows.append(dict(subject=subj, raw=vals))
    return pd.DataFrame(rows)


def attach_labels(feat, fasting, manual=None):
    """manual: optional DataFrame(file, fast_start) to override automatic matching."""
    feat = feat.copy()
    feat["window_start"] = pd.to_datetime(feat["window_start"])
    feat["fast_start"] = pd.NaT
    if manual is not None:
        m = manual.set_index("file")["fast_start"]
        feat["fast_start"] = pd.to_datetime(feat["file"].map(m))
    hrs = (feat["window_start"] - feat["fast_start"]).dt.total_seconds() / 3600
    feat["hours_since_drink"] = np.where(feat["condition"] == "fasting", hrs, 1.0)
    return feat


# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------
def feature_cols(df):
    cols = [c for m in MODALITIES.values() for c in m if c in df.columns]
    return [c for c in cols if df[c].notna().mean() > 0.5]


def add_clock(df):
    h = df["window_start"].dt.hour + df["window_start"].dt.minute / 60
    df["tod_sin"], df["tod_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    return df


def calibration_files(df, min_windows=10):
    """First non-fasting recording (>= min_windows minutes) of each subject = calibration."""
    nf = df[df["condition"] == "nonfasting"]
    size = nf.groupby(["subject", "file"]).size()
    nf = nf[nf.set_index(["subject", "file"]).index.map(size) >= min_windows]
    nf = nf.sort_values("window_start")
    return nf.groupby("subject")["file"].first().to_dict()


def personalize(df, cols, calib):
    """Express each feature relative to the subject's own hydrated baseline (z-score)."""
    out = df.copy()
    out[cols] = out[cols].astype(float)
    pop_sd = df[cols].std().replace(0, 1.0)  # population spread (no labels used)
    for s, f in calib.items():
        base = df[(df["subject"] == s) & (df["file"] == f)][cols]
        idx = df["subject"] == s
        out.loc[idx, cols] = (df.loc[idx, cols] - base.mean()) / pop_sd
    return out[out["subject"].isin(calib)]


# ---------------------------------------------------------------------------
# CV
# ---------------------------------------------------------------------------
def make_clf(kind):
    if kind == "logreg":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(max_iter=2000, class_weight="balanced"))
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                                class_weight="balanced", random_state=RNG, n_jobs=-1))


def make_reg():
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestRegressor(n_estimators=200, min_samples_leaf=5,
                                               random_state=RNG, n_jobs=-1))


def grouped_cv(df, cols, group_col, kind="rf", eval_mask=None):
    """Out-of-fold probabilities, regression predictions, grouped by group_col."""
    prob = pd.Series(np.nan, index=df.index)
    reg = pd.Series(np.nan, index=df.index)
    X, y, yr = df[cols], df["y"], df["hours_since_drink"]
    for g in df[group_col].unique():
        te = df[group_col] == g
        tr = ~te
        if eval_mask is not None:
            te = te & eval_mask
        if te.sum() == 0 or y[tr].nunique() < 2:
            continue
        m = make_clf(kind).fit(X[tr], y[tr])
        prob[te] = m.predict_proba(X[te])[:, 1]
        if kind == "rf":
            r = make_reg().fit(X[tr], yr[tr])
            reg[te] = r.predict(X[te])
    return prob, reg


def scores(y, p, yr=None, pr=None, thr=0.5):
    ok = p.notna()
    y, p = y[ok], p[ok]
    if len(y) == 0 or y.nunique() < 2:
        return {"n": int(len(y)), "auc": None}
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    out = dict(n=int(len(y)), n_pos=int(y.sum()),
               auc=round(roc_auc_score(y, p), 3),
               f1=round(f1_score(y, yhat), 3),
               precision=round(precision_score(y, yhat, zero_division=0), 3),
               recall=round(recall_score(y, yhat), 3),
               specificity=round(tn / (tn + fp), 3) if tn + fp else None,
               fpr=round(fp / (tn + fp), 3) if tn + fp else None,
               confusion=[[int(tn), int(fp)], [int(fn), int(tp)]])
    if pr is not None:
        okr = pr.notna() & yr.notna()
        if okr.sum():
            out["mae_hours"] = round(mean_absolute_error(yr[okr], pr[okr]), 2)
    return out


def per_subject_auc(df, p):
    res = {}
    for s, g in df.groupby("subject"):
        pp = p[g.index]
        if pp.notna().sum() and g["y"].nunique() == 2:
            res[s] = round(roc_auc_score(g["y"][pp.notna()], pp.dropna()), 3)
    return res


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir", type=Path)
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--threshold-h", type=float, default=8.0)
    ap.add_argument("--labels", type=Path, help="CSV file,fast_start overriding fastingTimes parsing")
    ap.add_argument("--drop", nargs="*", default=[], help="feature columns to exclude")
    ap.add_argument("--labeled-features", type=Path,
                    help="already-labeled features CSV (skips label attachment)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.labeled_features:
        df = pd.read_csv(args.labeled_features, parse_dates=["window_start"])
    else:
        feat = pd.read_csv(args.export_dir / "features.csv")
        manual = pd.read_csv(args.labels) if args.labels else None
        df = attach_labels(feat, parse_fasting_times(args.export_dir), manual)

    df = df.dropna(subset=["hours_since_drink"]).reset_index(drop=True)
    df = df[df["hours_since_drink"] >= 0].reset_index(drop=True)
    df = add_clock(df)
    df["y"] = (df["hours_since_drink"] >= args.threshold_h).astype(int)
    df["rec"] = df["subject"] + "/" + df["file"]
    cols = [c for c in feature_cols(df) if c not in args.drop]

    calib = calibration_files(df)
    df = df[df["subject"].isin(calib)].reset_index(drop=True)  # need a hydrated baseline
    is_calib = df.apply(lambda r: calib.get(r["subject"]) == r["file"], axis=1)
    eval_mask = ~is_calib
    dfp = personalize(df, cols, calib)

    R = {"threshold_h": args.threshold_h, "features": cols,
         "n_windows": int(len(df)), "n_subjects": int(df["subject"].nunique()),
         "n_recordings": int(df["rec"].nunique()),
         "label_dist": df.groupby("subject")["y"].agg(["count", "sum"]).rename(
             columns={"count": "windows", "sum": "deficit_windows"}).to_dict("index"),
         "calibration_files": calib}
    preds = df[["subject", "condition", "file", "window", "window_start", "hours_since_drink", "y",
                "acc_activity"] if "acc_activity" in df else
               ["subject", "condition", "file", "window", "window_start", "hours_since_drink", "y"]].copy()
    ev = df[eval_mask]

    multi_subject = df["subject"].nunique() >= 3
    if multi_subject:
        for kind in ["logreg", "rf"]:
            p, r = grouped_cv(df, cols, "subject", kind, eval_mask)
            R[f"general_{kind}"] = scores(ev["y"], p[eval_mask], ev["hours_since_drink"], r[eval_mask])
            R[f"general_{kind}"]["per_subject_auc"] = per_subject_auc(ev, p[eval_mask])
            preds[f"p_general_{kind}"] = p
            pp, rp = grouped_cv(dfp, cols, "subject", kind, eval_mask)
            R[f"personal_{kind}"] = scores(ev["y"], pp[eval_mask], ev["hours_since_drink"], rp[eval_mask])
            R[f"personal_{kind}"]["per_subject_auc"] = per_subject_auc(ev, pp[eval_mask])
            preds[f"p_personal_{kind}"] = pp
        pc, rc = grouped_cv(df, ["tod_sin", "tod_cos"], "subject", "logreg", eval_mask)
        R["clock_only"] = scores(ev["y"], pc[eval_mask], ev["hours_since_drink"], rc[eval_mask])
        R["ablation_personal_rf"] = {}
        for mod, mc in MODALITIES.items():
            mc = [c for c in mc if c in cols]
            if mc:
                pm, _ = grouped_cv(dfp, mc, "subject", "rf", eval_mask)
                R["ablation_personal_rf"][mod] = scores(ev["y"], pm[eval_mask]).get("auc")

    # within-subject (leave-one-recording-out) for the subject with the most recordings
    top = df.groupby("subject")["rec"].nunique().idxmax()
    sub = df[df["subject"] == top]
    if sub["y"].nunique() == 2 and sub["rec"].nunique() >= 4:
        p, r = grouped_cv(sub, cols, "rec", "rf")
        R["within_subject"] = {"subject": top, **scores(sub["y"], p, sub["hours_since_drink"], r)}
        pc, rc = grouped_cv(sub, ["tod_sin", "tod_cos"], "rec", "logreg")
        R["within_subject_clock_only"] = {"subject": top,
                                          **scores(sub["y"], pc, sub["hours_since_drink"], rc)}
        preds.loc[sub.index, "p_within"] = p
        preds.loc[sub.index, "hours_pred_within"] = r
        # importance + ablation within subject
        m = make_clf("rf").fit(sub[cols], sub["y"])
        imp = m[-1].feature_importances_
        R["within_importance"] = dict(sorted(zip(cols, np.round(imp, 4)), key=lambda x: -x[1]))
        R["within_ablation"] = {}
        for mod, mc in MODALITIES.items():
            mc = [c for c in mc if c in cols]
            if mc:
                pm, _ = grouped_cv(sub, mc, "rec", "rf")
                R["within_ablation"][mod] = scores(sub["y"], pm).get("auc")

    # importance for the personalized model (fit on all)
    if multi_subject:
        m = make_clf("rf").fit(dfp[cols], dfp["y"])
        R["personal_importance"] = dict(sorted(zip(cols, np.round(m[-1].feature_importances_, 4)),
                                               key=lambda x: -x[1]))

    # false alarms vs activity
    pcol = "p_personal_rf" if "p_personal_rf" in preds else "p_within"
    if pcol in preds and "acc_activity" in preds:
        neg = preds[(preds["y"] == 0) & preds[pcol].notna()].copy()
        if len(neg) >= 20:
            neg["activity_tertile"] = pd.qcut(neg["acc_activity"].rank(method="first"), 3,
                                              labels=["low", "mid", "high"])
            R["false_alarm_by_activity"] = {
                "model": pcol,
                **(neg.groupby("activity_tertile", observed=True)[pcol]
                   .apply(lambda s: round(float((s >= 0.5).mean()), 3)).to_dict())}

    preds.to_csv(args.out / "predictions.csv", index=False)
    with open(args.out / "metrics.json", "w") as fh:
        json.dump(R, fh, indent=2, default=str)
    print(json.dumps({k: v for k, v in R.items() if k not in ("features", "label_dist")},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
