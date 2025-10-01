#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Network ML Pipeline (Binary only, progress prints, optimized)
Adds sample-level classification by grouping flows per sample (file or session) and
applying a threshold rule:
    - malicious if (#predicted_malicious_flows >= SAMPLE_THRESHOLD_COUNT)
      OR (ratio_predicted_malicious >= SAMPLE_THRESHOLD_RATIO, if set)
Outputs per-model CSVs for both flow-level and sample-level evaluation.
- Loads CSVs from:
    BENIGN_DIR = "analysis_output_benign_baseline_labelled"
    RANSOMWARE_DIR = "Ransomware" (recursively per family)
- Uses the CSV column 'malicious' (coerced to 0/1) as the ONLY target.
- Drops high-cardinality text features before one-hot to keep things fast.
- Preprocessing: impute -> scale (sparse-friendly) -> one-hot (sparse).
- Models: RandomForest, LogisticRegression(saga), GradientBoosting (SVM optional but disabled).
- Evaluation: stratified 3-fold CV + 80/20 holdout.
- Prints detailed progress (file discovery, row counts, label distribution, feature types, per-model timing).
- Limits files per folder during iteration via LIMIT_PER_FOLDER.

Outputs:
    model_reports/<timestamp>/ with:
      - loaded_files_manifest.csv
      - label_counts.csv
      - binary__summary.csv
      - per-model reports + confusion matrices under binary/
      - README.txt
"""

from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional
import time

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC  

# ---------------------------
# Config
# ---------------------------
BENIGN_DIR = "Benign/analysis_output_benign_baseline_labelled"
RANSOMWARE_DIR = "Ransomware"
EXCLUDE_SUBSTR = "-with-ipinfo-"
OUTPUT_ROOT = "model_reports"
RANDOM_STATE = 42

RANSOMWARE_LIMIT_PER_FOLDER = 5   # cap per ransomware family (0 = no cap)
BENIGN_LIMIT = 0                  # 0 = no cap for benign 

HIGH_CARD_THRESHOLD = 2000          # drop cats with more unique values than this

SAMPLE_GROUP_MODE = "file"         # "file" 
SAMPLE_THRESHOLD_COUNT = 10        # classify sample as malicious if predicted mal flows >= this
SAMPLE_THRESHOLD_RATIO = None      # OR ratio >= this (e.g 0.02); set to None to disable

# Columns to exclude (IDs / raw tokens / payloads)
DEFAULT_EXCLUDE_COLS = [
    "flow_id", "src", "dst", "domain", "dns_qname", "dns_tld",
    "http_uri", "http_user_agent", "tls_sni",
    "ftp_user", "ftp_pass", "reasons", "score", "malicious","timestamp", "__source_file",

    
]

# Map typical boolean/string labels
BOOL_LIKE = {"TRUE": 1, "FALSE": 0, "True": 1, "False": 0, True: 1, False: 0, 1: 1, 0: 0}

# ---------------------------
# Helpers
# ---------------------------

def find_csvs(root: Path, exclude_substr: str) -> List[Path]:
    csvs = []
    for p in root.rglob("*.csv"):
        if exclude_substr and exclude_substr in p.name:
            continue
        csvs.append(p)
    return sorted(csvs)

def load_frame(csv_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="latin-1", low_memory=False)
    df["__source_file"] = str(csv_path)
    return df

def coerce_bool_like(col: pd.Series) -> pd.Series:
    uniques = set(str(x) for x in col.dropna().unique())
    if uniques.issubset(set(map(str, BOOL_LIKE.keys())) | {"nan"}):
        return col.map(BOOL_LIKE).astype("float64")
    return col

def mostly_numeric(col: pd.Series, thresh: float = 0.95, sample: int = 10000) -> bool:
    s = col.dropna()
    if s.empty:
        return False
    if len(s) > sample:
        s = s.sample(sample, random_state=0)
    s = s.astype(str)
    ok = s.str.match(r"^-?\d+(\.\d+)?$").mean()
    return ok >= thresh

def clean_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Convert obvious boolean-like object columns
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = coerce_bool_like(df[c])
    # Convert mostly-numeric object columns
    for c in df.columns:
        if df[c].dtype == "object" and mostly_numeric(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Timestamps: if present and numeric-ish, coerce to float seconds
    if "timestamp" in df.columns and df["timestamp"].dtype == "object" and mostly_numeric(df["timestamp"]):
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    return df

def ensure_binary_target(df: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    """Coerce 'malicious' column to 0/1 and store as target_binary."""
    if "malicious" not in df.columns:
        raise ValueError(f"'malicious' column not found in {csv_path}")
    mal = df["malicious"]
    if mal.dtype == "object":
        mal = coerce_bool_like(mal)
    mal_num = pd.to_numeric(mal, errors="coerce").fillna(0).astype(int).clip(0, 1)
    df = df.copy()
    df["target_binary"] = mal_num
    return df

def split_features_targets(df: pd.DataFrame, target_col: str, exclude_cols: list) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[c for c in [target_col] if c in df.columns], errors="ignore")
    X = X.drop(columns=[c for c in exclude_cols if c in X.columns], errors="ignore")
    y = df[target_col].astype("category")
    return X, y

def make_ohe():
    """Prefer sparse_output=True (sklearn >=1.2); fallback to sparse=True."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)

def build_preprocessor(X: pd.DataFrame):
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    # Drop very high-cardinality categoricals
    high_card = [c for c in categorical_features if X[c].nunique(dropna=True) > HIGH_CARD_THRESHOLD]
    categorical_features = [c for c in categorical_features if c not in high_card]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),  
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_ohe()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor, numeric_features, categorical_features, high_card

def model_zoo(random_state: int = 42):
    return {
        #"svm_rbf": SVC(kernel="rbf", probability=False, class_weight="balanced", random_state=random_state),  # optional/slow
        "random_forest": RandomForestClassifier(
            n_estimators=150, max_depth=30, n_jobs=-1,
            class_weight="balanced_subsample", random_state=random_state
        ),
        "log_reg_saga": LogisticRegression(
            solver="saga", max_iter=2000, n_jobs=-1, class_weight="balanced"
        ),
        "grad_boost": GradientBoostingClassifier(random_state=random_state),
    }

def leakage_checks(data: pd.DataFrame):
    print("\n[Leakage] Top numeric correlations with target_binary:")
    try:
        numdf = data.select_dtypes(include=[np.number])
        if "target_binary" in numdf.columns:
            corr_col = numdf.corr()["target_binary"].abs().sort_values(ascending=False)
            print(corr_col.head(15).to_string())
        else:
            print("  (target_binary not in numeric cols)")
    except Exception as e:
        print(f"  (skipped numeric corr: {e})")

    # Flag categorical values that are pure 0 or 1 (suspicious)
    cat_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in ["__source_file", "__source_basename"]]
    suspicious = []
    for c in cat_cols:
        vc = data.groupby(c)["target_binary"].mean()
        if (vc.eq(0).any() or vc.eq(1).any()) and vc.notna().any():
            # Gather examples of pure categories
            pure_1 = vc[vc == 1.0].index.tolist()[:3]
            pure_0 = vc[vc == 0.0].index.tolist()[:3]
            if pure_1 or pure_0:
                suspicious.append((c, pure_1, pure_0))
    if suspicious:
        print("\n[Leakage] Categorical values that are 100% one class (examples):")
        for c, ones, zeros in suspicious:
            print(f"  - {c}: 100% malicious={ones} ; 100% benign={zeros}")
    else:
        print("\n[Leakage] No pure categorical values detected in top scan.")

def evaluate_at_sample_level(y_true, y_pred, groups, out_dir: Path,
                             threshold_count: int = SAMPLE_THRESHOLD_COUNT,
                             threshold_ratio: Optional[float] = SAMPLE_THRESHOLD_RATIO):
    """
    Aggregate flow predictions -> sample predictions.

    true_sample = 1 if a sample contains >=1 true malicious flow
    pred_sample = 1 if (#pred_mal >= threshold_count) OR (ratio_pred_mal >= threshold_ratio, if set)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Normalize inputs to 1-D numpy arrays
    y_true_arr  = np.asarray(y_true).astype(int).ravel()
    y_pred_arr  = np.asarray(y_pred).astype(int).ravel()
    groups_arr  = np.asarray(groups).ravel()

    # Sanity check lengths
    if not (len(y_true_arr) == len(y_pred_arr) == len(groups_arr)):
        raise ValueError(f"Length mismatch: y_true={len(y_true_arr)}, y_pred={len(y_pred_arr)}, groups={len(groups_arr)}")

    df = pd.DataFrame({
        "true":  y_true_arr,
        "pred":  y_pred_arr,
        "group": groups_arr.astype(str),   
    })

    agg = df.groupby("group", sort=False).agg(
        n_flows    = ("pred", "size"),
        n_true_mal = ("true", "sum"),
        n_pred_mal = ("pred", "sum"),
    )
    
    agg["true_sample"] = (agg["n_true_mal"] > 0).astype(int)

    # Predicted label per sample: threshold rule
    rule_count = (agg["n_pred_mal"] >= int(threshold_count))
    if threshold_ratio is not None:
        ratio = agg["n_pred_mal"] / agg["n_flows"].clip(lower=1)
        rule_ratio = (ratio >= float(threshold_ratio))
        agg["pred_sample"] = ((rule_count) | (rule_ratio)).astype(int)
    else:
        agg["pred_sample"] = rule_count.astype(int)

    # Metrics
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    print("\n=== Sample-level Evaluation ===")
    rule_msg = f"Rule: pred_sample=1 if (n_pred_mal >= {threshold_count})"
    if threshold_ratio is not None:
        rule_msg += f" OR (ratio >= {float(threshold_ratio):.4f})"
    print(rule_msg)

    acc = accuracy_score(agg["true_sample"], agg["pred_sample"])
    print(f"Sample-level accuracy: {acc:.4f}\n")

    rep = classification_report(agg["true_sample"], agg["pred_sample"], output_dict=True, zero_division=0)
    print(pd.DataFrame(rep).T.to_string())

    cm = confusion_matrix(agg["true_sample"], agg["pred_sample"])
    print("\nSample-level Confusion Matrix:\n", cm)

    # Save artifacts
    agg.to_csv(out_dir / "sample_aggregate.csv", index=True)
    pd.DataFrame(rep).T.to_csv(out_dir / "sample_classification_report.csv", index=True)
    pd.DataFrame(cm,
                 index=["true_0","true_1"],
                 columns=["pred_0","pred_1"]).to_csv(out_dir / "sample_confusion_matrix.csv", index=True)

    return agg



# ---------------------------
# Evaluation (binary only)
# ---------------------------

def evaluate_models_binary(X: pd.DataFrame, y: pd.Series, groups_all: pd.Series, outdir: Path, random_state: int = 42):
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    preprocessor, num_cols, cat_cols, dropped_high_card = build_preprocessor(X)
    print(f"[Preprocess] numeric: {len(num_cols)}, categorical: {len(cat_cols)}, dropped_high_card: {len(dropped_high_card)}")

    results = []

    # Split once for the holdout
    X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
        X, y, groups_all, test_size=0.2, stratify=y, random_state=random_state
    )
    print(f"[Split] Train: {X_train.shape}, Test: {X_test.shape}")

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    for name, estimator in model_zoo(random_state).items():
        print(f"\n[Model] {name} — starting CV...")
        t_model = time.perf_counter()

        pipe = Pipeline(steps=[("prep", preprocessor), ("clf", estimator)])

        scoring = ["accuracy", "f1_macro", "roc_auc"] if hasattr(estimator, "predict_proba") else ["accuracy", "f1_macro"]
        cv_scores = cross_validate(pipe, X, y, cv=skf, scoring=scoring, n_jobs=-1, return_train_score=False)

        # Summarize CV
        msg = " | ".join([f"{m}={cv_scores['test_'+m].mean():.4f}" for m in scoring])
        print(f"[Model] {name} — CV done: {msg}")

        print(f"[Model] {name} — fitting on train and evaluating on holdout...")
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).T
        report_df.insert(0, "model", name)

        labels_sorted = sorted(y.unique())
        cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
        cm_df = pd.DataFrame(cm,
            index=[f"true_{l}" for l in labels_sorted],
            columns=[f"pred_{l}" for l in labels_sorted])
        cm_df.insert(0, "model", name)

        roc_auc = None
        if hasattr(pipe.named_steps["clf"], "predict_proba"):
            try:
                y_proba = pipe.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(pd.Categorical(y_test).codes, y_proba)
                print(f"[Model] {name} — Holdout ROC AUC: {roc_auc:.4f}")
            except Exception:
                roc_auc = np.nan
                print(f"[Model] {name} — Holdout ROC AUC: <failed>")

        model_dir = outdir / f"binary__{name}"
        model_dir.mkdir(exist_ok=True, parents=True)
        report_df.to_csv(model_dir / "classification_report.csv", index=True)
        cm_df.to_csv(model_dir / "confusion_matrix.csv", index=True)

        row = {
            "model": name,
            "cv_accuracy_mean": float(np.mean(cv_scores["test_accuracy"])),
            "cv_f1_macro_mean": float(np.mean(cv_scores["test_f1_macro"])),
            "test_accuracy": float(report.get("accuracy", np.nan)),
            "test_f1_macro": float(report_df.loc["macro avg", "f1-score"]) if "macro avg" in report_df.index else np.nan,
            "test_weighted_f1": float(report_df.loc["weighted avg", "f1-score"]) if "weighted avg" in report_df.index else np.nan,
            "test_roc_auc": None if roc_auc is None else float(roc_auc),
            "n_features_numeric": len(num_cols),
            "n_features_categorical": len(cat_cols),
            "n_features_dropped_high_cardinality": len(dropped_high_card),
        }
        results.append(row)

                # ---------- NEW: SAMPLE-LEVEL reporting ----------
        evaluate_at_sample_level(
            y_true=y_test,
            y_pred=y_pred,
            groups=g_test,                     # <— file IDs aligned to test rows
            out_dir=model_dir / "samples",
            threshold_count=SAMPLE_THRESHOLD_COUNT,
            threshold_ratio=SAMPLE_THRESHOLD_RATIO,  # None if you only want the count rule
        )


        print(f"[Model] {name} — done in {(time.perf_counter() - t_model):.2f}s")

    summary_df = pd.DataFrame(results)
    summary_df.to_csv(outdir / "binary__summary.csv", index=False)
    print(f"\n[Summary] Wrote: {outdir / 'binary__summary.csv'}")
    print(f"[Timing] Full evaluation took {(time.perf_counter() - t0):.2f}s")
    return summary_df

# ---------------------------
# Main
# ---------------------------

def main():
    benign_dir = Path(BENIGN_DIR).resolve()
    ransomware_root = Path(RANSOMWARE_DIR).resolve()
    output_dir = Path(OUTPUT_ROOT).resolve() / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Setup] BENIGN_DIR={benign_dir}")
    print(f"[Setup] RANSOMWARE_DIR={ransomware_root}")
    print(f"[Setup] RANSOMWARE_LIMIT_PER_FOLDER={RANSOMWARE_LIMIT_PER_FOLDER} | BENIGN_LIMIT={BENIGN_LIMIT} | HIGH_CARD_THRESHOLD={HIGH_CARD_THRESHOLD}")
    print(f"[Setup] SAMPLE_GROUP_MODE={SAMPLE_GROUP_MODE} | "
          f"THRESHOLD_COUNT={SAMPLE_THRESHOLD_COUNT} | THRESHOLD_RATIO={SAMPLE_THRESHOLD_RATIO}")

    # Discover CSVs
    benign_csvs = find_csvs(benign_dir, EXCLUDE_SUBSTR)
    # Do NOT cap benign
    if BENIGN_LIMIT and len(benign_csvs) > BENIGN_LIMIT:
        benign_csvs = benign_csvs[:BENIGN_LIMIT]   # (keeps behavior if you ever set BENIGN_LIMIT > 0)
    print(f"[Files] Benign CSVs: {len(benign_csvs)}{' (capped)' if BENIGN_LIMIT else ' (no cap)'}")

    family_dirs = [d for d in ransomware_root.iterdir() if d.is_dir()]
    ransomware_csvs = []
    for fam_dir in family_dirs:
        csvs = find_csvs(fam_dir, EXCLUDE_SUBSTR)
        if RANSOMWARE_LIMIT_PER_FOLDER and len(csvs) > RANSOMWARE_LIMIT_PER_FOLDER:
            csvs = csvs[:RANSOMWARE_LIMIT_PER_FOLDER]
        ransomware_csvs.extend(csvs)
    print(f"[Files] Ransomware CSVs (total across families): {len(ransomware_csvs)}"
        f"{' (capped per family)' if RANSOMWARE_LIMIT_PER_FOLDER else ''}")

    frames = []
    total_rows = 0
    for p in benign_csvs + ransomware_csvs:
        print(f"[Load] {p} ...", end="", flush=True)
        df = load_frame(p)
        before = len(df)
        df = clean_types(df)
        df = ensure_binary_target(df, p)
        after = len(df)
        frames.append(df)
        total_rows += after
        print(f" rows={after} (cleaned)")

    if not frames:
        raise SystemExit("No CSV files found. Check your paths and exclude pattern.")

    data = pd.concat(frames, ignore_index=True)

    # Manifest + label counts
    pd.DataFrame({"source_file": sorted(set(data['__source_file'].astype(str)))}) \
        .to_csv(output_dir / "loaded_files_manifest.csv", index=False)

    label_counts = data["target_binary"].value_counts(dropna=False).rename_axis("label").reset_index(name="count")
    label_counts.to_csv(output_dir / "label_counts.csv", index=False)

    # Progress prints 
    features_df = data.drop(columns=DEFAULT_EXCLUDE_COLS, errors="ignore")
    print("\n========== DATA OVERVIEW ==========")
    print(f"Shape of final dataset (incl. target + features before split): {features_df.shape}")
    print(f"Number of malicious samples: {int((data['target_binary'] == 1).sum())}")
    print(f"Number of benign samples: {int((data['target_binary'] == 0).sum())}")
    print("===================================\n")

    leakage_checks(data)

    # Build X,y for binary task
    X_bin, y_bin = split_features_targets(data, "target_binary", DEFAULT_EXCLUDE_COLS)
    groups_all = data["__source_file"].astype(str)

    # Evaluate (binary only)
    binary_dir = output_dir / "binary"
    summary_bin = evaluate_models_binary(X_bin, y_bin, groups_all=groups_all, outdir=binary_dir, random_state=RANDOM_STATE)

    # Save summary copy at root
    summary_bin.to_csv(output_dir / "binary__summary.csv", index=False)

    # README
    (output_dir / "README.txt").write_text(
f"""# Model Reports (Binary only and Sample)
Created: {datetime.now().isoformat()}

- Loaded {data['__source_file'].nunique()} unique CSV files.
- Total rows: {len(data)}
- Sample grouping: {SAMPLE_GROUP_MODE} (file=session? {SAMPLE_GROUP_MODE=='session'})
- Sample rule: malicious if n_pred_mal >= {SAMPLE_THRESHOLD_COUNT}{' or ratio >= '+str(SAMPLE_THRESHOLD_RATIO) if SAMPLE_THRESHOLD_RATIO is not None else ''}
- Excluded columns: {', '.join(DEFAULT_EXCLUDE_COLS)}
- Labeling:
  - 'target_binary' from CSV 'malicious' (coerced 0/1)
- Input roots:
  - BENIGN_DIR: {benign_dir}
  - RANSOMWARE_DIR: {ransomware_root}
- - Limits:
  - Ransomware per-family cap: {RANSOMWARE_LIMIT_PER_FOLDER if RANSOMWARE_LIMIT_PER_FOLDER else 'no cap'}
  - Benign cap: {BENIGN_LIMIT if BENIGN_LIMIT else 'no cap'}
- Excluded filenames containing: {EXCLUDE_SUBSTR}
- Preprocessing: sparse one-hot, sparse-friendly scaling, high-cardinality drop (>{HIGH_CARD_THRESHOLD})
- Models: RandomForest, LogisticRegression(saga), GradientBoosting, SVM
- Evaluation: Stratified 3-fold CV + 80/20 holdout

Generated files:
- loaded_files_manifest.csv
- label_counts.csv
- binary__summary.csv
- binary/binary__<model>/classification_report.csv
- binary/binary__<model>/confusion_matrix.csv
"""
    )

    print("\n[Done] Reports saved to:", output_dir)

if __name__ == "__main__":
    main()
