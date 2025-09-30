"""
LSTM flow-level classifier (file-respecting split)

- Loads CSV flows from BENIGN_DIR and RANSOMWARE_DIR/** (per family)
- Uses CSV 'malicious' -> target_binary (0/1)
- Numeric features only; timestamp only for ordering
- Splits by FILE (no flow mixing across splits)
- LSTM sequence labeling: predicts per-flow malicious (0/1)
- Prints flow-level metrics + per-file predicted malicious %
"""

from pathlib import Path
from datetime import datetime
from typing import List, Optional

import os
import numpy as np
import pandas as pd

# Torch / Sklearn
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_recall_fscore_support, accuracy_score,
    classification_report, confusion_matrix
)

# =============== Config ===============
BENIGN_DIR = "analysis_output_benign_baseline_labelled"
RANSOMWARE_DIR = "Ransomware"
EXCLUDE_SUBSTR = "-with-ipinfo-"
OUTPUT_ROOT = "model_reports"

# Capping (no cap for benign; optional per-ransomware-family cap)
RANSOMWARE_LIMIT_PER_FOLDER = 0   # 0 = no cap on ransomware families

# LSTM / training knobs
GROUP_COL = "__source_file"   # file path is our group ID
MAX_LEN = 256
BATCH_SIZE = 64
EPOCHS = 8
LR = 1e-3
HIDDEN = 128
LAYERS = 1
BIDIR = True
DROPOUT = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# Columns we never use as numeric features (IDs/leaky)
DROP_FOR_FEATURES = {
    "target_binary", "malicious", "score", "reasons", "timestamp",  # target & leak
    "flow_id", "src", "dst", "domain", "dns_qname", "dns_tld",      # IDs / free text
    "http_uri", "http_user_agent", "tls_sni", "uri_base64", "base64_payload",
    "ftp_user", "ftp_pass"
}

# =============== Helpers ===============
BOOL_LIKE = {"TRUE": 1, "FALSE": 0, "True": 1, "False": 0, True: 1, False: 0, 1: 1, 0: 0}

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
    # Convert boolean-like strings
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = coerce_bool_like(df[c])
    # Convert mostly-numeric objects
    for c in df.columns:
        if df[c].dtype == "object" and mostly_numeric(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Coerce timestamp if numeric-ish
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

# =============== LSTM dataset / model ===============
class SeqDataset(Dataset):
    """Holds per-file sequences of (features, per-flow labels), padded/truncated."""
    def __init__(self, seq_table: pd.DataFrame, max_len: int, input_dim: int):
        self.files = seq_table["file"].tolist()
        Xs, Ys, Ls = [], [], []
        for x, y in zip(seq_table["features"], seq_table["labels"]):
            x = np.asarray(x, dtype=np.float32)
            y = np.asarray(y, dtype=np.float32)
            L = min(len(x), max_len)
            Xp = np.zeros((max_len, input_dim), dtype=np.float32)
            Yp = np.zeros((max_len,), dtype=np.float32)
            Xp[:L] = x[:L]
            Yp[:L] = y[:L]
            Xs.append(Xp); Ys.append(Yp); Ls.append(L)
        self.X = np.stack(Xs, axis=0).astype(np.float32)     # [N, T, D]
        self.Y = np.stack(Ys, axis=0).astype(np.float32)     # [N, T]
        self.L = np.asarray(Ls, dtype=np.int64)              # [N]

    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        return (torch.from_numpy(self.X[i]),
                torch.from_numpy(self.Y[i]),
                torch.tensor(self.L[i], dtype=torch.long),
                self.files[i])

class LSTMTagger(nn.Module):
    """Sequence labeller: per-flow logits."""
    def __init__(self, input_dim, hidden=HIDDEN, layers=LAYERS, bidir=BIDIR, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=layers,
                            batch_first=True, bidirectional=bidir,
                            dropout=(dropout if layers > 1 else 0.0))
        out_dim = hidden * (2 if bidir else 1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_dim, 1)
        )

    def forward(self, x, lengths):
        # x: [B, T, D], lengths: [B]
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        pad_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=MAX_LEN)  # [B, T, H*]
        logits = self.head(pad_out).squeeze(-1)  # [B, T]
        return logits

# =============== Main ===============
def main():
    # Seeds
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    benign_dir = Path(BENIGN_DIR).resolve()
    ransom_root = Path(RANSOMWARE_DIR).resolve()
    out_dir = Path(OUTPUT_ROOT).resolve() / datetime.now().strftime("%Y%m%d_%H%M%S") / "lstm_flows"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Setup] BENIGN_DIR={benign_dir}")
    print(f"[Setup] RANSOMWARE_DIR={ransom_root}")
    print(f"[Setup] OUTPUT={out_dir}")
    print(f"[Setup] DEVICE={DEVICE} | MAX_LEN={MAX_LEN} | EPOCHS={EPOCHS}")

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
        print(f" rows={len(df)}")
        frames.append(df)

    if not frames:
        raise SystemExit("No CSVs loaded. Check paths and EXCLUDE_SUBSTR.")

    data = pd.concat(frames, ignore_index=True)

    # ---- Overview
    print("\n========== DATA OVERVIEW ==========")
    print(f"Rows: {len(data)} | Files: {data['__source_file'].nunique()}")
    print(f"Malicious flows: {int((data['target_binary']==1).sum())}")
    print(f"Benign flows   : {int((data['target_binary']==0).sum())}")
    print("===================================\n")

    # ---- Numeric features only (drop IDs/leaks)
    num_cols = [c for c in data.select_dtypes(include=[np.number]).columns if c not in DROP_FOR_FEATURES]
    if not num_cols:
        raise SystemExit("No numeric features after dropping leaky/ID columns. Check DROP_FOR_FEATURES.")
    print(f"[Features] Using {len(num_cols)} numeric columns.")

    def folder_label(file_path: str, benign_root: Path, ransom_root: Path) -> int:
        p = Path(file_path)
        try:
            p.relative_to(benign_root)
            return 0  # benign
        except ValueError:
            pass
        try:
            p.relative_to(ransom_root)
            return 1  # ransomware
        except ValueError:
            return 0  # default to benign if outside both roots

    # Build sample_df using FOLDER truth (not flow-derived)
    files_unique = data["__source_file"].astype(str).drop_duplicates()
    sample_df = pd.DataFrame({GROUP_COL: files_unique})
    sample_df["sample_label"] = sample_df[GROUP_COL].map(
        lambda s: folder_label(s, benign_dir, ransom_root)
    ).astype(int)

    # Split strictly by FILE, stratified by folder truth
    train_groups, test_groups = train_test_split(
        sample_df[GROUP_COL],
        test_size=0.2,
        stratify=sample_df["sample_label"],
        random_state=SEED,
    )
    val_groups, test_groups = train_test_split(
        test_groups, test_size=0.5, random_state=SEED
    )

    def subset_by_groups(df, groups):
        return df[df[GROUP_COL].isin(set(groups))].copy()

    train_df = subset_by_groups(data, train_groups)
    val_df   = subset_by_groups(data, val_groups)
    test_df  = subset_by_groups(data, test_groups)

    print(f"[Split] Files — Train: {train_df[GROUP_COL].nunique()}, "
        f"Val: {val_df[GROUP_COL].nunique()}, Test: {test_df[GROUP_COL].nunique()}")
    print(f"[Split] Flows — Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # --- sanitize numeric features ---
    for df in (train_df, val_df, test_df):
        # normalize weird values to NaN
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

    # compute train medians once
    med = train_df[num_cols].median()

    # fill NaNs with train medians
    train_df[num_cols] = train_df[num_cols].fillna(med)
    val_df[num_cols]   = val_df[num_cols].fillna(med)
    test_df[num_cols]  = test_df[num_cols].fillna(med)

    # drop columns that are still all-NaN (shouldn't happen after fill) or truly constant
    const_cols = [c for c in num_cols if train_df[c].nunique(dropna=False) <= 1]
    if const_cols:
        print(f"[Sanity] Dropping constant/empty columns: {const_cols}")
        for df in (train_df, val_df, test_df):
            df.drop(columns=const_cols, inplace=True, errors="ignore")
        num_cols = [c for c in num_cols if c not in const_cols]

    # final guardrails (fail fast if any remain)
    assert train_df[num_cols].isna().sum().sum() == 0, "NaNs left in train after fill!"
    assert val_df[num_cols].isna().sum().sum() == 0,   "NaNs left in val after fill!"
    assert test_df[num_cols].isna().sum().sum() == 0,  "NaNs left in test after fill!"


    
    # ---- Scale numeric (fit on TRAIN only)
    scaler = StandardScaler()
    scaler.fit(train_df[num_cols])

    # ---- Build per-file sequences (features + per-flow labels)
    def build_sequences(df: pd.DataFrame) -> pd.DataFrame:
        sort_cols = [GROUP_COL] + (["timestamp"] if "timestamp" in df.columns else [])
        df = df.sort_values(sort_cols)
        feats = scaler.transform(df[num_cols]).astype(np.float32)
        labels = df["target_binary"].astype(np.float32).values
        files = df[GROUP_COL].astype(str).values

        seq_feats, seq_labels, file_keys = [], [], []
        _, idx = np.unique(files, return_index=True)
        idx = np.sort(idx)
        bounds = list(idx) + [len(files)]
        for i in range(len(bounds)-1):
            s, e = bounds[i], bounds[i+1]
            seq_feats.append(feats[s:e])
            seq_labels.append(labels[s:e])
            file_keys.append(files[s])
        return pd.DataFrame({"file": file_keys, "features": seq_feats, "labels": seq_labels})

    train_seqs = build_sequences(train_df)
    val_seqs   = build_sequences(val_df)
    test_seqs  = build_sequences(test_df)

    input_dim = len(num_cols)
    train_ds = SeqDataset(train_seqs, MAX_LEN, input_dim)
    val_ds   = SeqDataset(val_seqs,   MAX_LEN, input_dim)
    test_ds  = SeqDataset(test_seqs,  MAX_LEN, input_dim)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    # ---- Model
    model = LSTMTagger(input_dim).to(DEVICE)

    # Imbalance weight over TRAIN valid steps
    with torch.no_grad():
        mask_train = torch.zeros((len(train_ds), MAX_LEN), dtype=torch.float32)
        for i, L in enumerate(train_ds.L):
            mask_train[i, :L] = 1.0
        y_train_steps = torch.from_numpy(train_ds.Y) * mask_train
        pos = y_train_steps.sum().item()
        tot = mask_train.sum().item()
        neg = tot - pos
        pos_weight = torch.tensor((neg / max(pos, 1.0)), dtype=torch.float32, device=DEVICE)

    criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ---- Train / Eval loops
    def step(loader, train=True):
        model.train(mode=train)
        total_loss, total_steps = 0.0, 0.0
        all_true, all_proba = [], []
        per_file_rows = []  # (file, n_flows, n_pred)
        for X, Y, L, files in loader:
            X = X.to(DEVICE)         # [B, T, D]
            Y = Y.to(DEVICE)         # [B, T]
            L = L.to(DEVICE)         # [B]
            if train: optimizer.zero_grad()
            logits = model(X, L)     # [B, T]
            # mask padded steps
            mask = torch.arange(MAX_LEN, device=DEVICE)[None, :].expand(X.size(0), -1) < L[:, None]
            loss_raw = criterion(logits, Y)            # [B, T]
            loss = (loss_raw * mask.float()).sum()
            denom = mask.float().sum().clamp_min(1.0)
            loss = loss / denom
            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * denom.item()
            total_steps += denom.item()

            proba = torch.sigmoid(logits).detach().cpu().numpy()
            Y_np  = Y.detach().cpu().numpy()
            mask_np = mask.detach().cpu().numpy()

            # per-flow accumulation
            all_true.append(Y_np[mask_np].ravel())
            all_proba.append(proba[mask_np].ravel())

            # per-file rows
            n_flows = mask_np.sum(axis=1)
            n_pred  = (proba >= 0.5).astype(int)
            n_pred  = (n_pred * mask_np).sum(axis=1)
            per_file_rows.extend([(f, int(nf), int(np_)) for f, nf, np_ in zip(files, n_flows, n_pred)])

        all_true = np.concatenate(all_true) if len(all_true) else np.array([])
        all_proba = np.concatenate(all_proba) if len(all_proba) else np.array([])
        return (total_loss / max(total_steps, 1.0)), all_true, all_proba, per_file_rows

    def summarize(y_true, y_proba, name="set"):
        if y_true.size == 0:
            print(f"[{name}] no samples.")
            return {}
        yhat = (y_proba >= 0.5).astype(int)
        try:
            auc = roc_auc_score(y_true, y_proba)
        except Exception:
            auc = float("nan")
        acc = accuracy_score(y_true, yhat)
        p, r, f1, _ = precision_recall_fscore_support(y_true, yhat, average="binary", zero_division=0)
        print(f"[{name}] AUC {auc:.3f} | F1 {f1:.3f} | Acc {acc:.3f} | P {p:.3f} | R {r:.3f}")
        return {"auc": auc, "f1": f1, "acc": acc, "prec": p, "recall": r}

    best_val = -np.inf
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_y, tr_p, _ = step(train_loader, train=True)
        va_loss, va_y, va_p, _ = step(val_loader,   train=False)
        print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} | val loss {va_loss:.4f}", end="  ")
        va = summarize(va_y, va_p, "val")
        score = va.get("auc", -np.inf)
        if np.isfinite(score) and score > best_val:
            best_val = score
            torch.save(model.state_dict(), out_dir / "best_lstm_flows.pt")

    # ---- Test (per-flow + per-file %)
    model.load_state_dict(torch.load(out_dir / "best_lstm_flows.pt", map_location=DEVICE))

    te_loss, te_y, te_p, test_file_rows = step(test_loader, train=False)
    print(f"[test] loss {te_loss:.4f}", end="  ")
    summarize(te_y, te_p, "test")

    # Flow-level detailed report
    yhat = (te_p >= 0.5).astype(int)
    print("\n[Flow-level classification report]")
    print(classification_report(te_y, yhat, digits=4, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(te_y, yhat))

    # Per-file predicted malicious %
    test_file_df = pd.DataFrame(test_file_rows, columns=["file","n_flows","n_pred_mal"])
    test_file_df = (test_file_df.groupby("file", as_index=False).sum())
    test_file_df["pred_pct"] = (100.0 * test_file_df["n_pred_mal"] / test_file_df["n_flows"].clip(lower=1)).round(2)

    print("\nPer-file predicted malicious % (test set):")
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(test_file_df.sort_values("pred_pct", ascending=False).to_string(index=False))

    # Optionally save artifacts
    test_file_df.to_csv(out_dir / "per_file_test_pred_pct.csv", index=False)
    pd.DataFrame({"feature": num_cols}).to_csv(out_dir / "numeric_features_used.csv", index=False)
    print(f"\n[Saved] best model -> {out_dir/'best_lstm_flows.pt'}")
    print(f"[Saved] per-file table -> {out_dir/'per_file_test_pred_pct.csv'}")

if __name__ == "__main__":
    main()
