"""
LSTM flow-level classifier (file-respecting split, OHE categorical, variable-length batches)
 
- Temporal features: delta_t (log1p) + rolling stats (mean/std/max over last 10)
- Heavy-tailed numerics: log1p on counts/durations
- Port engineering: privileged flag + bucket categories; drop raw sport/dport
- Lower OHE cardinality cap (30) and keep leaky text-ish cols excluded
- Model head: LayerNorm + small MLP; default BIDIR=False
- Training: early stopping on PR-AUC (average precision) + ReduceLROnPlateau; up to 20 epochs
- Split: by FILE, stratified by file-level positive-rate bins for stabler eval
- Threshold tuned on validation (max F1), used for test
- Saves flow metrics CSVs and best model
"""
 
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Dict
 
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
 
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_recall_fscore_support, accuracy_score,
    classification_report, confusion_matrix, precision_recall_curve,
    average_precision_score
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
 
# ================== Config ==================
BENIGN_DIR = "Benign/analysis_output_benign_baseline_labelled"
RANSOMWARE_DIR = "Tool"  # or "Ransomware" in your tree
EXCLUDE_SUBSTR = "-with-ipinfo-"
OUTPUT_ROOT = "model_reports"
 
# Cap ransomware files per family (benign has no cap)
RANSOMWARE_LIMIT_PER_FOLDER = 0     # 0 = no cap per ransomware family
 
GROUP_COL = "__source_file"         # file path is our group ID
 
# Categorical handling
CAT_MAX_UNIQUES = 30                # lowered from 50
 
# Training knobs
BATCH_SIZE = 32
EPOCHS_MAX = 20                     # with early stopping
LR = 1e-3
HIDDEN = 128
LAYERS = 1
BIDIR = False                       # default to causal
DROPOUT = 0.2
GRAD_CLIP_NORM = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
 
# Class imbalance
POS_WEIGHT_CAP = 50.0
 
# Early stop/lr sched
EARLY_STOP_PATIENCE = 5
 
# Columns to exclude entirely (IDs, payloads, high-cardinality tokens, leaks)
EXCLUDE_ALWAYS = {
    # targets/leaks
    "target_binary", "malicious", "score", "reasons", "timestamp",
    # explicit IDs or very high-cardinality text-y columns
    "flow_id", "src", "dst", "domain", "dns_qname", "dns_tld",
    "http_uri", "http_user_agent", "tls_sni", "uri_base64", "base64_payload",
    "ftp_user", "ftp_pass",
}
 
# Heavy-tailed numeric features for log1p
LOG1P_NUMS = ["packet_count", "byte_count", "session_duration",
              "burst_count", "inter_packet_timing_stddev"]
 
# Port buckets
WELL_KNOWN_PORTS = {53: "dns", 80: "http", 443: "https", 123: "ntp", 1900: "ssdp", 5355: "llmnr"}
 
# ================== Helpers ==================
BOOL_LIKE = {"TRUE": 1, "FALSE": 0, "True": 1, "False": 0, True: 1, False: 0, 1: 1, 0: 0}

def stringify_cats(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    if not cols:
        return df
    out = df.copy()
    # ensure all categorical inputs are *strings* (not mixed float/str)
    out[cols] = (
        out[cols]
        .astype("string")
        .fillna("__MISSING__")
        .apply(lambda s: s.str.strip())
    )
    return out


def set_seed(seed: int = SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
 
def find_csvs(root: Path, exclude_substr: str) -> List[Path]:
    csvs = []
    for p in root.rglob("*.csv"):
        if p.name.startswith("._"):
            continue
        if exclude_substr and exclude_substr in p.name:
            continue
        csvs.append(p)
    return sorted(csvs)
 
def load_frame(csv_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="latin-1", low_memory=False)
    df[GROUP_COL] = str(csv_path)
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
    # Convert obvious boolean-like strings
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = coerce_bool_like(df[c])
    # Convert mostly-numeric objects to numeric
    for c in df.columns:
        if df[c].dtype == "object" and mostly_numeric(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Coerce timestamp if numeric-ish (we only use it for sorting)
    if "timestamp" in df.columns and df["timestamp"].dtype == "object" and mostly_numeric(df["timestamp"]):
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    return df
 
def ensure_binary_target(df: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    if "malicious" not in df.columns:
        raise ValueError(f"'malicious' column not found in {csv_path}")
    mal = df["malicious"]
    if mal.dtype == "object":
        mal = coerce_bool_like(mal)
    mal_num = pd.to_numeric(mal, errors="coerce").fillna(0).astype(int).clip(0, 1)
    df = df.copy()
    df["target_binary"] = mal_num
    return df
 
def add_port_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["sport", "dport"]:
        if col in df.columns:
            # privileged
            df[f"{col}_priv"] = (pd.to_numeric(df[col], errors="coerce").fillna(-1) < 1024).astype(float)
            # bucket category
            df[f"{col}_bucket"] = pd.to_numeric(df[col], errors="coerce").map(WELL_KNOWN_PORTS).fillna("other")
    # drop raw ports to avoid double counting
    for raw in ["sport", "dport"]:
        if raw in df.columns:
            EXCLUDE_ALWAYS.add(raw)
    return df
 
def add_temporal_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Add delta_t (log1p seconds) and simple rolling stats (last 10) per file."""
    df = df_raw.copy()
    if "timestamp" in df.columns and pd.api.types.is_numeric_dtype(df["timestamp"]):
        df.sort_values([GROUP_COL, "timestamp"], inplace=True)
        dt = df.groupby(GROUP_COL)["timestamp"].diff().fillna(0)
        df["delta_t"] = np.log1p(dt.clip(lower=0))
    else:
        df["delta_t"] = 0.0
 
    base_roll = ["packet_count", "byte_count", "session_duration", "burst_count"]
    for c in base_roll:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            g = df.groupby(GROUP_COL)[c]
            df[c+"_roll10_mean"] = g.rolling(10, min_periods=1).mean().reset_index(level=0, drop=True)
            df[c+"_roll10_std"]  = g.rolling(10, min_periods=1).std().fillna(0).reset_index(level=0, drop=True)
            df[c+"_roll10_max"]  = g.rolling(10, min_periods=1).max().reset_index(level=0, drop=True)
    return df
 
def folder_label(file_path: str, benign_root: Path, ransom_root: Path) -> int:
    p = Path(file_path)
    try:
        p.relative_to(benign_root); return 0
    except ValueError:
        pass
    try:
        p.relative_to(ransom_root); return 1
    except ValueError:
        return 0
 
# ================== Dataset / Model ==================
class SeqDataset(Dataset):
    """Per-file VARIABLE-LENGTH sequences of (features, per-flow labels)."""
    def __init__(self, seq_table: pd.DataFrame):
        self.files = seq_table["file"].tolist()
        self.X = [np.asarray(x, dtype=np.float32) for x in seq_table["features"].tolist()]  # [Li, D]
        self.Y = [np.asarray(y, dtype=np.float32) for y in seq_table["labels"].tolist()]    # [Li]
 
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        x, y = self.X[i], self.Y[i]
        return x, y, x.shape[0], self.files[i]
 
def collate_batch(batch):
    xs, ys, Ls, files = zip(*batch)
    B = len(xs)
    D = xs[0].shape[1]
    T = max(Ls)
    X = torch.zeros(B, T, D, dtype=torch.float32)
    Y = torch.zeros(B, T,    dtype=torch.float32)
    L = torch.tensor(Ls, dtype=torch.long)
    for i, (x, y) in enumerate(zip(xs, ys)):
        li = x.shape[0]
        X[i, :li] = torch.from_numpy(x)
        Y[i, :li] = torch.from_numpy(y)
    return X, Y, L, list(files)
 
class LSTMTagger(nn.Module):
    """Sequence labeller: per-flow logits."""
    def __init__(self, input_dim, hidden=HIDDEN, layers=LAYERS, bidir=BIDIR, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=layers,
                            batch_first=True, bidirectional=bidir,
                            dropout=(dropout if layers > 1 else 0.0))
        out_dim = hidden * (2 if bidir else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim // 2, 1)
        )
 
    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        pad_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)  # [B, Tvar, H*]
        logits = self.head(pad_out).squeeze(-1)  # [B, Tvar]
        return logits
 
# ================== Main ==================
def main():
    set_seed(SEED)
 
    benign_dir = Path(BENIGN_DIR).resolve()
    ransom_root = Path(RANSOMWARE_DIR).resolve()
    out_dir = Path(OUTPUT_ROOT).resolve() / datetime.now().strftime("%Y%m%d_%H%M%S") / "lstm_flows"
    out_dir.mkdir(parents=True, exist_ok=True)
 
    print(f"[Setup] BENIGN_DIR={benign_dir}")
    print(f"[Setup] RANSOMWARE_DIR={ransom_root}")
    print(f"[Setup] OUTPUT={out_dir}")
    print(f"[Setup] DEVICE={DEVICE} | EPOCHS_MAX={EPOCHS_MAX} | BATCH_SIZE={BATCH_SIZE}")
 
    # ---- Discover CSVs
    benign_csvs = find_csvs(benign_dir, EXCLUDE_SUBSTR)
    fam_dirs = [d for d in ransom_root.iterdir() if d.is_dir()]
    ransom_csvs = []
    for fam in fam_dirs:
        csvs = find_csvs(fam, EXCLUDE_SUBSTR)
        if RANSOMWARE_LIMIT_PER_FOLDER and len(csvs) > RANSOMWARE_LIMIT_PER_FOLDER:
            csvs = csvs[:RANSOMWARE_LIMIT_PER_FOLDER]
        ransom_csvs.extend(csvs)
 
    print(f"[Files] Benign CSVs: {len(benign_csvs)} (no cap)")
    print(f"[Files] Ransomware CSVs: {len(ransom_csvs)}"
          f"{' (capped per family)' if RANSOMWARE_LIMIT_PER_FOLDER else ''}")
 
    # ---- Load & clean
    frames = []
    for p in benign_csvs + ransom_csvs:
        print(f"[Load] {p} ...", end="", flush=True)
        df = load_frame(p)
        df = clean_types(df)
        df = ensure_binary_target(df, p)
        df = add_port_features(df)
        print(f" rows={len(df)}")
        frames.append(df)
 
    if not frames:
        raise SystemExit("No CSVs loaded. Check paths and EXCLUDE_SUBSTR.")
 
    data = pd.concat(frames, ignore_index=True)
 
    # ---- Temporal features (needs timestamp + grouping)
    data = add_temporal_features(data)
 
    # ---- Overview
    print("\n========== DATA OVERVIEW ==========")
    print(f"Rows: {len(data)} | Files: {data[GROUP_COL].nunique()}")
    print(f"Malicious flows: {int((data['target_binary']==1).sum())}")
    print(f"Benign flows   : {int((data['target_binary']==0).sum())}")
    print("===================================\n")
 
    # ---------- Choose feature columns (numeric + low-cardinality categoricals) ----------
    candidate_cols = [c for c in data.columns if c not in EXCLUDE_ALWAYS and c != GROUP_COL]
 
    # Log1p heavy-tailed numerics (in-place)
    for c in LOG1P_NUMS:
        if c in data.columns and pd.api.types.is_numeric_dtype(data[c]):
            data[c] = np.log1p(data[c].clip(lower=0))
 
    numeric_cols = list(data[candidate_cols].select_dtypes(include=[np.number]).columns)
    cat_candidates = [c for c in candidate_cols if c not in numeric_cols]
 
    # Low-cardinality categorical filter
    cat_cols = []
    for c in cat_candidates:
        nunique = data[c].nunique(dropna=True)
        if 1 <= nunique <= CAT_MAX_UNIQUES:
            cat_cols.append(c)
 
    feature_cols = numeric_cols + cat_cols
    if not feature_cols:
        raise SystemExit("No usable features after filtering. Adjust EXCLUDE_ALWAYS / CAT_MAX_UNIQUES.")
 
    print(f"[Features] Numeric: {len(numeric_cols)} | Categorical (OHE): {len(cat_cols)} | Total input cols: {len(feature_cols)}")
    if cat_cols:
        print(f"          OHE cols: {cat_cols}")
 
    # ---------- Folder truth for stratified file split ----------
    # Get unique file keys as plain strings (no integer index shenanigans)
    files_unique = data[GROUP_COL].astype(str).drop_duplicates()
 
    # Compute file-level positive rate indexed by file path (strings)
    file_pos_rate = (
        data.groupby(GROUP_COL)["target_binary"]
            .mean()
            .reindex(files_unique.tolist())  # reindex using the values (order from files_unique)
            .fillna(0.0)
    )
 
    # Bin by prevalence for stratified file split (quantiles), fallback to equal-width bins
    try:
        bins = pd.qcut(file_pos_rate, q=5, duplicates="drop")
    except Exception:
        bins = pd.cut(file_pos_rate, bins=5, include_lowest=True)
 
    # IMPORTANT: build the DataFrame using the SAME index (file paths)
    sample_df = pd.DataFrame({
        GROUP_COL: file_pos_rate.index.astype(str),  # column from index
        "strat": bins.values,                        # aligned values
    })
 
    # Split strictly by FILE, stratified by file-level prevalence bins
    train_groups, test_groups = train_test_split(
        sample_df[GROUP_COL],
        test_size=0.2,
        stratify=sample_df["strat"],
        random_state=SEED,
    )
    val_groups, test_groups = train_test_split(test_groups, test_size=0.5, random_state=SEED)
 
    def subset_by_groups(df, groups):
        return df[df[GROUP_COL].isin(set(groups))].copy()
 
    train_df = subset_by_groups(data, train_groups)
    val_df   = subset_by_groups(data, val_groups)
    test_df  = subset_by_groups(data, test_groups)
 
    print(f"[Split] Files — Train: {train_df[GROUP_COL].nunique()}, "
          f"Val: {val_df[GROUP_COL].nunique()}, Test: {test_df[GROUP_COL].nunique()}")
    print(f"[Split] Flows — Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
 
    # ---------- Build preprocessing (fit on TRAIN only) ----------
    for df_ in (train_df, val_df, test_df):
        df_[feature_cols] = df_[feature_cols].replace([np.inf, -np.inf], np.nan)
 
    # Remove near-constant numeric features based on TRAIN only
    def low_variance(cols, df_train, thresh=1e-6):
        keep = []
        for c in cols:
            if c in df_train.columns and pd.api.types.is_numeric_dtype(df_train[c]):
                if np.nanstd(df_train[c].values) > thresh:
                    keep.append(c)
        return keep
    numeric_cols = low_variance(numeric_cols, train_df)
    feature_cols = numeric_cols + cat_cols
 
    # 1) Impute numerics + cats
    pre_imp = ColumnTransformer(
        transformers=[
            ("num_imp", SimpleImputer(strategy="median"), numeric_cols),
            ("cat_imp", SimpleImputer(strategy="constant", fill_value="__MISSING__"), cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    pre_imp.fit(train_df[feature_cols])
 
    def transform_after_impute(df_subset: pd.DataFrame) -> pd.DataFrame:
        Z = pre_imp.transform(df_subset[feature_cols])
        cols = numeric_cols + cat_cols
        return pd.DataFrame(Z, columns=cols, index=df_subset.index)
 
    train_imp = stringify_cats(transform_after_impute(train_df), cat_cols)
    val_imp   = stringify_cats(transform_after_impute(val_df),   cat_cols)
    test_imp  = stringify_cats(transform_after_impute(test_df),  cat_cols)
 
    # 2) Scale numerics + OHE cats
    scaler = StandardScaler()
    if numeric_cols:
        scaler.fit(train_imp[numeric_cols])
 
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float32)
    if cat_cols:
        ohe.fit(train_imp[cat_cols])
 
    def final_transform(df_imp: pd.DataFrame) -> np.ndarray:
        parts = []
        if numeric_cols:
            parts.append(scaler.transform(df_imp[numeric_cols]).astype(np.float32))
        if cat_cols:
            parts.append(ohe.transform(df_imp[cat_cols]).astype(np.float32))
        if parts:
            return np.concatenate(parts, axis=1)
        return np.empty((len(df_imp), 0), dtype=np.float32)
 
    # ---------- Build per-file sequences (variable length) ----------
    def build_sequences(df_raw: pd.DataFrame, df_imp: pd.DataFrame) -> pd.DataFrame:
        sort_cols = [GROUP_COL] + (["timestamp"] if "timestamp" in df_raw.columns else [])
        order = df_raw.sort_values(sort_cols).index
        df_sorted = df_raw.loc[order]
        X_sorted = final_transform(df_imp.loc[order])
        y_sorted = df_sorted["target_binary"].astype(np.float32).values
        files    = df_sorted[GROUP_COL].astype(str).values
 
        seq_feats, seq_labels, file_keys = [], [], []
        _, idx = np.unique(files, return_index=True)
        idx = np.sort(idx)
        bounds = list(idx) + [len(files)]
        for i in range(len(bounds)-1):
            s, e = bounds[i], bounds[i+1]
            seq_feats.append(X_sorted[s:e])
            seq_labels.append(y_sorted[s:e])
            file_keys.append(files[s])
        return pd.DataFrame({"file": file_keys, "features": seq_feats, "labels": seq_labels})
 
    train_seqs = build_sequences(train_df, train_imp)
    val_seqs   = build_sequences(val_df,   val_imp)
    test_seqs  = build_sequences(test_df,  test_imp)
 
    # ---- Datasets / Loaders
    train_ds = SeqDataset(train_seqs)
    val_ds   = SeqDataset(val_seqs)
    test_ds  = SeqDataset(test_seqs)
 
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_batch)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
 
    # ---- Model
    input_dim = train_ds.X[0].shape[1] if len(train_ds) else 0
    if input_dim == 0:
        raise SystemExit("No features after preprocessing. Check filters / CAT_MAX_UNIQUES.")
    model = LSTMTagger(input_dim).to(DEVICE)
 
    # ---- Class imbalance from TRAIN FLOWS
    pos_tokens = int((train_df["target_binary"] == 1).sum())
    tot_tokens = int(len(train_df))
    neg_tokens = max(tot_tokens - pos_tokens, 0)
    raw_pw = neg_tokens / max(pos_tokens, 1)
    pos_weight = torch.tensor(min(raw_pw, POS_WEIGHT_CAP), dtype=torch.float32, device=DEVICE)
    print(f"[Imbalance] train pos={pos_tokens} neg={neg_tokens} -> pos_weight={pos_weight.item():.2f} (cap={POS_WEIGHT_CAP})")
 
    # Bias-init final layer
    prior = pos_tokens / max(tot_tokens, 1)
    with torch.no_grad():
        # last Linear is head[-1]
        model.head[-1].bias.fill_(float(np.log((prior + 1e-12) / max(1e-12, 1 - prior))))
 
    criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1, verbose=True)
    # ---- LR scheduler (Plateau), version-safe (some builds don't support `verbose`)
    try:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=1, threshold=1e-4
        )
    except TypeError:
        # Fallback if even the signature differs further
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=1
        )
 
    def _current_lr(opt):
        return float(opt.param_groups[0]["lr"])
 
    # ---- Train / Eval loops
    def step(loader, train=True):
        model.train(mode=train)
        total_loss, total_steps = 0.0, 0.0
        all_true, all_proba = [], []
        for X, Y, L, _files in loader:
            X = X.to(DEVICE); Y = Y.to(DEVICE); L = L.to(DEVICE)
            if train: optimizer.zero_grad()
            logits = model(X, L)     # [B, Tvar]
            T = logits.size(1)
            mask = (torch.arange(T, device=DEVICE)[None, :] < L[:, None])
            loss_raw = criterion(logits, Y)
            loss = (loss_raw * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()
            total_loss += loss.item() * mask.float().sum().item()
            total_steps += mask.float().sum().item()
            proba = torch.sigmoid(logits).detach().cpu().numpy()
            Y_np  = Y.detach().cpu().numpy()
            mask_np = mask.detach().cpu().numpy()
            all_true.append(Y_np[mask_np].ravel())
            all_proba.append(proba[mask_np].ravel())
        all_true = np.concatenate(all_true) if len(all_true) else np.array([])
        all_proba = np.concatenate(all_proba) if len(all_proba) else np.array([])
        return (total_loss / max(total_steps, 1.0)), all_true, all_proba
 
    def summarize(y_true, y_proba, name="set", thr=0.5):
        if y_true.size == 0:
            print(f"[{name}] no samples.")
            return {}
        yhat = (y_proba >= thr).astype(int)
        try:
            auc = roc_auc_score(y_true, y_proba)
        except Exception:
            auc = float("nan")
        try:
            ap = average_precision_score(y_true, y_proba)
        except Exception:
            ap = float("nan")
        acc = accuracy_score(y_true, yhat)
        p, r, f1, _ = precision_recall_fscore_support(y_true, yhat, average="binary", zero_division=0)
        print(f"[{name}] AUC {auc:.3f} | AP {ap:.3f} | F1 {f1:.3f} | Acc {acc:.3f} | P {p:.3f} | R {r:.3f} | thr {thr:.3f}")
        return {"auc": auc, "ap": ap, "f1": f1, "acc": acc, "prec": p, "recall": r}
 
    best_val_score = -np.inf
    bad = 0
 
    for epoch in range(1, EPOCHS_MAX + 1):
        tr_loss, tr_y, tr_p = step(train_loader, train=True)
        va_loss, va_y, va_p = step(val_loader,   train=False)
        try:
            va_ap = average_precision_score(va_y, va_p)
        except Exception:
            va_ap = float("nan")
        print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} | val loss {va_loss:.4f} | val-AP {va_ap:.4f}")
 
        prev_lr = _current_lr(optimizer)
        if np.isfinite(va_ap):
            scheduler.step(va_ap)
        new_lr = _current_lr(optimizer)
        if new_lr < prev_lr:
            print(f"[LR] ReduceLROnPlateau: {prev_lr:.2e} -> {new_lr:.2e}")
 
        if np.isfinite(va_ap):
            scheduler.step(va_ap)
        score = va_ap if np.isfinite(va_ap) else -np.inf
        if score > best_val_score:
            best_val_score = score
            bad = 0
            torch.save(model.state_dict(), out_dir / "best_lstm_flows.pt")
        else:
            bad += 1
            if bad >= EARLY_STOP_PATIENCE:
                print("[EarlyStop] patience reached.")
                break
 
    # ---- Load best & tune threshold on validation flows (max F1)
    model.load_state_dict(torch.load(out_dir / "best_lstm_flows.pt", map_location=DEVICE))
    _, va_y2, va_p2 = step(val_loader, train=False)
    if len(va_y2) > 0:
        prec, rec, thr = precision_recall_curve(va_y2, va_p2)
        f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
        best_idx = int(np.argmax(f1)) if len(f1) else 0
        BEST_THR = float(thr[best_idx]) if len(thr) else 0.5
    else:
        BEST_THR = 0.5
    print(f"[Threshold] Selected BEST_THR={BEST_THR:.4f} (max F1 on validation)")
 
    # ---- Test (flow-level only) + SAVE CSVs
    te_loss, te_y, te_p = step(test_loader, train=False)
    print(f"[test] loss {te_loss:.4f}", end="  ")
    test_stats = summarize(te_y, te_p, "test", thr=BEST_THR)
 
    yhat = (te_p >= BEST_THR).astype(int)
    rep_dict = classification_report(te_y, yhat, output_dict=True, zero_division=0)
    rep_df = pd.DataFrame(rep_dict).T
    cm = confusion_matrix(te_y, yhat)
    cm_df = pd.DataFrame(cm, index=["true_0","true_1"], columns=["pred_0","pred_1"])
 
    # Save CSVs
    rep_df.to_csv(out_dir / "flow_classification_report.csv", index=True)
    cm_df.to_csv(out_dir / "flow_confusion_matrix.csv", index=True)
    pd.DataFrame([{
        "set": "test",
        "threshold": BEST_THR,
        "n_flows": int(len(te_y)),
        "auc": float(test_stats.get("auc", np.nan)),
        "ap": float(test_stats.get("ap", np.nan)),
        "f1": float(test_stats.get("f1", np.nan)),
        "accuracy": float(test_stats.get("acc", np.nan)),
        "precision": float(test_stats.get("prec", np.nan)),
        "recall": float(test_stats.get("recall", np.nan)),
    }]).to_csv(out_dir / "flow_metrics_summary.csv", index=False)
 
    # Print for convenience
    print("\n[Flow-level classification report]")
    print(rep_df.to_string())
    print("\nConfusion matrix:\n", cm_df.to_string())
    print(f"\n[Saved] best model -> {out_dir/'best_lstm_flows.pt'}")
    print(f"[Saved] flow_classification_report.csv, flow_confusion_matrix.csv, flow_metrics_summary.csv in {out_dir}")
 
if __name__ == "__main__":
    main()