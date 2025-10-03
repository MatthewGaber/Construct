"""
Network ML Pipeline (Binary only, progress prints, optimized)
Adds sample-level classification by grouping flows per sample (FILE-ONLY) and
applying a threshold rule:
    - malicious if (#predicted_malicious_flows >= SAMPLE_THRESHOLD_COUNT)
      OR (ratio_predicted_malicious >= SAMPLE_THRESHOLD_RATIO, if set)
Outputs per-model CSVs for both flow-level and sample-level evaluation.

Key changes in this version:
- **Group-aware holdout split by file** using GroupShuffleSplit (no file appears in both train/test).
- **Group-aware CV** when available (StratifiedGroupKFold preferred; falls back to GroupKFold; finally StratifiedKFold).
- SAMPLE_GROUP_MODE is locked to "file".

- Loads CSVs from:
    BENIGN_DIR = "analysis_output_benign_baseline_labelled"
    RANSOMWARE_DIR = "Ransomware" (recursively per family)
- Uses the CSV column 'malicious' (coerced to 0/1) as the ONLY target.
- Drops high-cardinality text features before one-hot to keep things fast.
- Preprocessing: impute -> scale (sparse-friendly) -> one-hot (sparse).
- Models: RandomForest, LogisticRegression(saga), GradientBoosting (SVM optional but disabled).
- Prints detailed progress (file discovery, row counts, label distribution, feature types, per-model timing).

Outputs:
    model_reports/<timestamp>/ with:
      - loaded_files_manifest.csv
      - label_counts.csv
      - binary__summary.csv
      - per-model reports + confusion matrices under binary/
      - per-model sample-level reports under binary/<model>/samples/
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
from sklearn.model_selection import (
    cross_validate,
    StratifiedKFold,
    GroupShuffleSplit,
)
# Try to use group-aware stratified CV when available
try:
    from sklearn.model_selection import StratifiedGroupKFold  # sklearn >= 1.1
    HAS_STRAT_GROUP_KFOLD = True
except Exception:
    HAS_STRAT_GROUP_KFOLD = False
    from sklearn.model_selection import GroupKFold

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC  
from sklearn.calibration import CalibratedClassifierCV


try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# ---------------------------
# Config
# ---------------------------
BENIGN_DIR = "Benign/analysis_output_benign_baseline_labelled"
RANSOMWARE_DIR = "Spyware"
EXCLUDE_SUBSTR = "-with-ipinfo-"
OUTPUT_ROOT = "model_reports"
RANDOM_STATE = 42

RANSOMWARE_LIMIT_PER_FOLDER = 0   # cap per ransomware family (0 = no cap)
BENIGN_LIMIT = 0                  # 0 = no cap for benign

HIGH_CARD_THRESHOLD = 2000        # drop cats with more unique values than this

# File-only aggregation + decision threshold
SAMPLE_THRESHOLD_COUNT = 20        # classify sample as malicious if predicted mal flows >= this
SAMPLE_THRESHOLD_RATIO = 0.045      # OR ratio >= this (e.g., 0.02); set to None to disable

# Columns to exclude 
DEFAULT_EXCLUDE_COLS = [
    "flow_id", "src", "dst", "domain", "dns_qname", "dns_tld",
    "http_uri", "http_user_agent", "tls_sni",
    "ftp_user", "ftp_pass", "reasons", "score", "malicious", "timestamp", "__source_file",
    
]

# Map typical boolean/string labels
BOOL_LIKE = {"TRUE": 1, "FALSE": 0, "True": 1, "False": 0, True: 1, False: 0, 1: 1, 0: 0}

# ---------------------------
# Helpers
# ---------------------------
def debug_preprocessor_per_file(preprocessor, X, y, groups):
    """Fit *once* on the whole dataset (just for debugging),
       then transform per file to see who explodes."""
    ok, bad = [], []
    pre_fit = preprocessor.fit(X, y)  # debug-only fit
    for fname, idx in X.groupby(groups).groups.items():
        try:
            _ = pre_fit.transform(X.loc[idx])
            ok.append(fname)
        except Exception as e:
            bad.append((fname, repr(e)))
    print(f"[Debug] Per-file transform OK={len(ok)}, BAD={len(bad)}")
    for f, err in bad[:10]:
        print("  BAD:", f, "=>", err)
    return ok, bad

def debug_preprocessor_per_fold(preprocessor, X, y, groups, n_splits=3, random_state=42):
    """Use the SAME splitter as CV to find which fold breaks."""
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold, (tr, te) in enumerate(skf.split(X, y, groups)):
        Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
        try:
            pre = preprocessor.fit(Xtr, ytr)
            _ = pre.transform(Xte)  
            print(f"[Debug] Fold {fold}: OK  (train {Xtr.shape}, test {Xte.shape})")
        except Exception as e:
            print(f"[Debug] Fold {fold}: **FAIL**  (train {Xtr.shape}, test {Xte.shape}) -> {repr(e)}")
            bad_files = Xte.index.to_series().map(groups).unique().tolist()
            print("        Test files in failing fold (first 10):")
            for bf in bad_files[:10]: print("         -", bf)
            return fold, bad_files
    return None, []

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


def compute_pos_weight(y: pd.Series) -> float:
    """Return neg/pos ratio for scale_pos_weight-style imbalance handling."""
    y = pd.Series(y).astype(int)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0:
        return 1.0
    return max(n_neg / n_pos, 1.0)


def model_zoo(random_state: int = 42, pos_weight: float = 1.0):
    zoo = {
        "svm_linear": CalibratedClassifierCV(estimator=LinearSVC(C=0.5, class_weight='balanced', dual='auto', max_iter=20000),method='sigmoid', cv=3),
        "random_forest": RandomForestClassifier(
            n_estimators=150, max_depth=30, n_jobs=-1,
            class_weight="balanced_subsample", random_state=random_state
        ),
        "log_reg_saga": LogisticRegression(
            solver="saga", max_iter=2000, n_jobs=-1, class_weight="balanced"
        ),
        "grad_boost": GradientBoostingClassifier(random_state=random_state),
    }
    if HAS_LGBM:
        zoo["lightgbm"] = LGBMClassifier(
            n_estimators=800,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1,
            # imbalance:
            scale_pos_weight=pos_weight,
            
        )

    if HAS_XGB:
        zoo["xgboost"] = XGBClassifier(
            n_estimators=800,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="auc",
            # imbalance:
            scale_pos_weight=pos_weight,
        )

    return zoo


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


# ---------------------------
# Sample-level aggregation
# ---------------------------

def evaluate_at_sample_level(
    y_true,
    y_pred,
    groups,
    out_dir: Path,
    threshold_count: int = SAMPLE_THRESHOLD_COUNT,
    threshold_ratio: Optional[float] = SAMPLE_THRESHOLD_RATIO,
    true_sample_map: Optional[dict] = None,
):
    """
    Aggregate flow predictions -> sample predictions (file-level).

    true_sample (folder-based if true_sample_map provided, else flow-derived):
      - folder-based: label comes from directory (0=benign, 1=ransomware)
      - flow-derived: 1 if the sample has >=1 true-malicious flow in the TEST subset

    pred_sample (THE AND RULE):
      1 if (n_pred_mal >= threshold_count) AND (n_pred_mal / n_flows >= threshold_ratio)
      If threshold_ratio is None, it's treated as TRUE (i.e., count-only).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Rebuild df  ----
    y_true_arr = np.asarray(y_true).astype(int).ravel()
    y_pred_arr = np.asarray(y_pred).astype(int).ravel()
    groups_arr = np.asarray(groups).astype(str).ravel()

    # Sanity check
    if not (len(y_true_arr) == len(y_pred_arr) == len(groups_arr)):
        raise ValueError(
            f"Length mismatch: y_true={len(y_true_arr)}, y_pred={len(y_pred_arr)}, groups={len(groups_arr)}"
        )

    df = pd.DataFrame(
        {"true": y_true_arr, "pred": y_pred_arr, "group": groups_arr}
    )

    # ---- Aggregate to file level ----
    agg = df.groupby("group", sort=False).agg(
        n_flows=("pred", "size"),
        n_true_mal=("true", "sum"),
        n_pred_mal=("pred", "sum"),
    )

    # ---- Choose ground truth (folder-based or flow-derived) ----
    if true_sample_map is not None:
        truth_used = "folder"
        agg["true_sample"] = (
            agg.index.to_series().map(true_sample_map).fillna(0).astype(int)
        )
    else:
        truth_used = "flow-derived"
        agg["true_sample"] = (agg["n_true_mal"] > 0).astype(int)

    # ---- AND rule (count AND ratio) ----
    ratio = agg["n_pred_mal"] / agg["n_flows"].clip(lower=1)
    rule_count = agg["n_pred_mal"] >= int(threshold_count)
    if threshold_ratio is None:
        # If None, treat ratio condition as always-true 
        rule_ratio = pd.Series(True, index=agg.index)
    else:
        rule_ratio = ratio >= float(threshold_ratio)

    agg["pred_sample"] = (rule_count & rule_ratio).astype(int)

    # ---- Metrics ----
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    print("\n=== Sample-level Evaluation ===")
    print(f"Ground-truth used: {truth_used}")
    rule_msg = f"AND rule: pred_sample=1 if (n_pred_mal >= {int(threshold_count)}) AND (ratio >= {threshold_ratio if threshold_ratio is not None else 'N/A'})"
    print(rule_msg)

    acc = accuracy_score(agg["true_sample"], agg["pred_sample"])
    print(f"Sample-level accuracy: {acc:.4f}\n")

    rep = classification_report(
        agg["true_sample"], agg["pred_sample"], output_dict=True, zero_division=0
    )
    print(pd.DataFrame(rep).T.to_string())

    cm = confusion_matrix(agg["true_sample"], agg["pred_sample"])
    print("\nSample-level Confusion Matrix:\n", cm)

    
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        per_file = agg.copy()
        per_file["pred_pct"] = (per_file["n_pred_mal"] / per_file["n_flows"].clip(lower=1) * 100).round(2)
        cols = ["n_flows","n_pred_mal","pred_pct","true_sample","pred_sample"]
        print("\nPer-file predicted malicious % (test set):")
        print(per_file[cols].sort_values("pred_pct", ascending=False).to_string())

    # Save artifacts (unchanged)
    agg.to_csv(out_dir / "sample_aggregate.csv", index=True)
    pd.DataFrame(rep).T.to_csv(out_dir / "sample_classification_report.csv", index=True)
    pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(
        out_dir / "sample_confusion_matrix.csv", index=True
    )

    return agg




# ---------------------------
# Evaluation (binary only) — FILE-GROUPED
# ---------------------------

def evaluate_models_binary(X: pd.DataFrame, y: pd.Series, groups_all: pd.Series, outdir: Path, random_state: int = 42, true_sample_map=None):
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    preprocessor, num_cols, cat_cols, dropped_high_card = build_preprocessor(X)
    print(f"[Preprocess] numeric: {len(num_cols)}, categorical: {len(cat_cols)}, dropped_high_card: {len(dropped_high_card)}")

    results = []

    # ---- Holdout split by FILE (no contamination) ----
    # ---- Holdout split by FILE (group-aware + stratified) ----
    if HAS_STRAT_GROUP_KFOLD:
        # use 5 folds and take the first as a stratified, group-aware holdout
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
        train_idx, test_idx = next(sgkf.split(X, y, groups=groups_all))
        split_desc = "StratifiedGroupKFold-based holdout (fold 0)"
    else:
        # fallback: not stratified by label
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
        train_idx, test_idx = next(gss.split(X, y, groups=groups_all))
        split_desc = "GroupShuffleSplit (not stratified)"

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    g_train, g_test = groups_all.iloc[train_idx], groups_all.iloc[test_idx]

    print(f"[Split] {split_desc}")
    print(f"[Split] Train: {X_train.shape}, Test: {X_test.shape} (grouped by file)")

    # optional: show class balance to verify stratification worked
    def _rate(y_ser):
        yy = pd.Series(y_ser).astype(int)
        return float((yy == 1).mean())

    print(f"[Holdout balance] overall pos={_rate(y):.3f} | "
        f"train pos={_rate(y_train):.3f} | test pos={_rate(y_test):.3f}")
    print(f"[Groups] total files={groups_all.nunique()} | "
        f"train files={g_train.nunique()} | test files={g_test.nunique()}")


    pos_weight = compute_pos_weight(y)
    print(f"[Imbalance] scale_pos_weight ≈ {pos_weight:.2f}")

    # ---- Group-aware CV when possible ----
    if HAS_STRAT_GROUP_KFOLD:
        cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=random_state)
        cv_groups = groups_all
        cv_desc = "StratifiedGroupKFold (group-aware)"
    else:
        try:
            cv = GroupKFold(n_splits=3)
            cv_groups = groups_all
            cv_desc = "GroupKFold (group-aware, not stratified)"
        except Exception:
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
            cv_groups = None
            cv_desc = "StratifiedKFold (row-wise; upgrade sklearn for StratifiedGroupKFold)"
    print(f"[CV] Using {cv_desc}")

    for name, estimator in model_zoo(random_state, pos_weight).items():
        print(f"\n[Model] {name} — starting CV...")
        t_model = time.perf_counter()

        pipe = Pipeline(steps=[("prep", preprocessor), ("clf", estimator)])

        scoring = ["accuracy", "f1_macro"]
        if hasattr(estimator, "predict_proba"):
            scoring.append("roc_auc")

        if cv_groups is not None:
            cv_scores = cross_validate(pipe, X, y, cv=cv, groups=groups_all, scoring=scoring, n_jobs=-1, return_train_score=False)
        else:
            cv_scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)

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

        # ---------- SAMPLE-LEVEL reporting ----------
        #evaluate_at_sample_level(
            #y_true=y_test,
            #y_pred=y_pred,
            #groups=g_test,
            #out_dir=model_dir / "samples",
            #threshold_count=SAMPLE_THRESHOLD_COUNT,
            #threshold_ratio=SAMPLE_THRESHOLD_RATIO,
            #true_sample_map=true_sample_map,          
        #)


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
    print(f"[Setup] Sample grouping: FILE ONLY | THRESHOLD_COUNT={SAMPLE_THRESHOLD_COUNT} | THRESHOLD_RATIO={SAMPLE_THRESHOLD_RATIO}")

    # Discover CSVs
    benign_csvs = find_csvs(benign_dir, EXCLUDE_SUBSTR)
    # Do NOT cap benign by default
    if BENIGN_LIMIT and len(benign_csvs) > BENIGN_LIMIT:
        benign_csvs = benign_csvs[:BENIGN_LIMIT]
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
    # Folder-based truth mapping for files (0=benign, 1=ransomware)
    def _file_label_from_path(path_str: str, benign_root: Path, ransom_root: Path) -> int:
        p = Path(path_str)
        try:
            p.relative_to(benign_root); return 0
        except ValueError:
            pass
        try:
            p.relative_to(ransom_root); return 1
        except ValueError:
            return 0  # default benign if outside both roots

    file_truth_map = {s: _file_label_from_path(s, benign_dir, ransomware_root)
                    for s in data["__source_file"].astype(str).unique()}

    # Manifest + label counts
    pd.DataFrame({"source_file": sorted(set(data['__source_file'].astype(str)))}) \
        .to_csv(output_dir / "loaded_files_manifest.csv", index=False)

    label_counts = data["target_binary"].value_counts(dropna=False).rename_axis("label").reset_index(name="count")
    label_counts.to_csv(output_dir / "label_counts.csv", index=False)

    # Progress prints 
    features_df = data.drop(columns=DEFAULT_EXCLUDE_COLS, errors="ignore")
    print("\n========== DATA OVERVIEW ==========")
    print(f"Shape of final dataset (incl. target + features before split): {features_df.shape}")
    print(f"Number of malicious flows: {int((data['target_binary'] == 1).sum())}")
    print(f"Number of benign flows: {int((data['target_binary'] == 0).sum())}")
    print("===================================\n")

    leakage_checks(data)

    # Build X,y and file groups
    X_bin, y_bin = split_features_targets(data, "target_binary", DEFAULT_EXCLUDE_COLS)
    groups_all = data["__source_file"].astype(str)  # FILE-ONLY grouping

    print(X_bin.dtypes)
    # --- fix: make categoricals uniform strings before the pipeline ---
    cat_cols = X_bin.select_dtypes(exclude=[np.number]).columns
    X_bin[cat_cols] = X_bin[cat_cols].astype("string").fillna("__MISSING__").apply(lambda s: s.str.strip())


    pre, num_cols, cat_cols, dropped_high_card = build_preprocessor(X_bin)

    # 1) Per-file smoke test
    debug_preprocessor_per_file(pre, X_bin, y_bin, groups_all)

    # 2) Fold-level test aligned with CV splitter
    debug_preprocessor_per_fold(pre, X_bin, y_bin, groups_all, n_splits=3, random_state=RANDOM_STATE)

    # Evaluate (binary only)
    binary_dir = output_dir / "binary"
    summary_bin = evaluate_models_binary(X_bin, y_bin, groups_all=groups_all, outdir=binary_dir, random_state=RANDOM_STATE, true_sample_map=file_truth_map,)

    # Save summary copy at root
    summary_bin.to_csv(output_dir / "binary__summary.csv", index=False)

    # README
    (output_dir / "README.txt").write_text(
f"""# Model Reports (Binary only + File-grouped)
Created: {datetime.now().isoformat()}

- Label: 'target_binary' from CSV 'malicious' (coerced 0/1)
- Total rows: {len(data)}
- Grouping: FILE ONLY (GroupShuffleSplit holdout; group-aware CV when available)
- Sample rule: malicious if n_pred_mal >= {SAMPLE_THRESHOLD_COUNT}{' or ratio >= '+str(SAMPLE_THRESHOLD_RATIO) if SAMPLE_THRESHOLD_RATIO is not None else ''}
- Excluded columns: {', '.join(DEFAULT_EXCLUDE_COLS)}
- Preprocessing: sparse one-hot, sparse-friendly scaling, high-card drop (>{HIGH_CARD_THRESHOLD})
- Models: RandomForest, LogisticRegression(saga), GradientBoosting
- Holdout: 80/20 by file (no cross-contamination)
- CV: {('StratifiedGroupKFold' if HAS_STRAT_GROUP_KFOLD else 'GroupKFold (or StratifiedKFold fallback)')}

Generated files:
- loaded_files_manifest.csv
- label_counts.csv
- binary__summary.csv
- binary/binary__<model>/classification_report.csv
- binary/binary__<model>/confusion_matrix.csv
- binary/binary__<model>/samples/sample_aggregate.csv
- binary/binary__<model>/samples/sample_classification_report.csv
- binary/binary__<model>/samples/sample_confusion_matrix.csv
"""
    )

    print("\n[Done] Reports saved to:", output_dir)


if __name__ == "__main__":
    main()
