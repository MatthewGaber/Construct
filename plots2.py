import os
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

sns.set(style="whitegrid")

def extract_family_from_folder(folder_name):
    name = os.path.basename(folder_name).lower()
    if "apt29" in name:
        return "APT29"
    elif "apt28" in name:
        return "APT28"
    elif "analysis_output_" in name:
        stripped = name.replace("analysis_output_", "").split("_")[0]
        return stripped.capitalize()
    else:
        return os.path.basename(folder_name).capitalize()

# ----------------------------
# Loaders
# ----------------------------
def load_csvs_flat(folder_path, label='Benign'):
    all_files = glob(os.path.join(folder_path, '*.csv'))
    dfs = []
    label_name = extract_family_from_folder(folder_path) if label == 'Benign' else label
    for file in all_files:
        try:
            df = pd.read_csv(file)
            df['family'] = label_name
            df['sample_id'] = os.path.basename(file)
            dfs.append(df)
        except Exception as e:
            print(f"[ERROR] loading benign {file}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def load_csvs_nested_all(parent_folder):
    family_frames = []
    family_folders = [f.path for f in os.scandir(parent_folder) if f.is_dir()]
    for folder in family_folders:
        family_name = extract_family_from_folder(folder)
        all_files = glob(os.path.join(folder, '*.csv'))
        family_dfs = []
        for file in all_files:
            try:
                df = pd.read_csv(file)
                df['family'] = family_name
                df['sample_id'] = os.path.basename(file)
                family_dfs.append(df)
                print(f"{file} → loaded {len(df)} rows")
            except Exception as e:
                print(f"[ERROR] loading {file}: {e}")
        if family_dfs:
            family_frames.append(pd.concat(family_dfs, ignore_index=True))
    return pd.concat(family_frames, ignore_index=True) if family_frames else pd.DataFrame()


# ----------------------------
# Load Data
# ----------------------------
benign_path = 'analysis_output_benign_baseline_labelled'
spyware_path = 'Spyware'

df_benign = load_csvs_flat(benign_path)
df_spyware = load_csvs_nested_all(spyware_path)

df_combined = pd.concat([df_benign, df_spyware], ignore_index=True)
assert not df_combined.empty, "Combined data is empty"

# ----------------------------
# Normalize & Clean
# ----------------------------
columns_to_convert = [
    'byte_count', 'packet_count', 'inter_packet_timing_stddev',
    'uri_entropy', 'session_duration', 'burst_count',
    'score', 'dns_entropy', 'dns_sub_entropy', 'domain_entropy'
]
for c in columns_to_convert:
    if c not in df_combined.columns:
        df_combined[c] = np.nan

for col in ['protocol', 'app_proto', 'transport']:
    if col not in df_combined.columns:
        df_combined[col] = ''
    df_combined[col] = df_combined[col].astype(str).fillna('').str.strip().str.upper()

df_combined[columns_to_convert] = df_combined[columns_to_convert].apply(pd.to_numeric, errors='coerce')
df_combined = df_combined.dropna(subset=columns_to_convert, how='all')
df_combined['malicious'] = pd.to_numeric(df_combined.get('malicious', 0), errors='coerce').fillna(0).astype(int)

# ----------------------------
# Label protocol_pair and presence_smart
# ----------------------------
def _norm(v):
    s = str(v).strip().upper()
    return None if s in {"", "UNKNOWN", "NAN", "NONE"} else s

def make_protocol_pair(row):
    p_app = _norm(row.get("app_proto"))
    p_low = _norm(row.get("protocol"))
    if p_low == "VSSMONITORING": p_low = None
    parts = [x for x in (p_app, p_low) if x]
    return " | ".join(parts) if parts else "Unknown"

def _presence_app(row):
    a = row.get("app_proto", "")
    p = row.get("protocol", "")
    if a in TRANSPORT_TOKENS: a = ""
    if p in TRANSPORT_TOKENS: p = ""
    cand = a if a else p
    return SYNONYM.get(cand, cand) or "UNKNOWN"

def app_with_transport_if_unusual(row):
    app = row["presence_label_app"]
    tr  = row.get("transport", "")
    exp = EXPECTED_TRANSPORT.get(app)
    if not app or app == "UNKNOWN": return tr or "UNKNOWN"
    if exp and tr == exp: return app
    if tr in {"TCP", "UDP"}: return f"{app} over {tr}"
    return app

TRANSPORT_TOKENS = {"TCP","UDP","ICMP","SCTP","ARP","OTHER","UNKNOWN","DATA","VSSMONITORING"}
SYNONYM = {"HTTP2":"HTTP", "HTTP/2":"HTTP", "HTTP3":"HTTP/3", "QUIC":"HTTP/3"}
EXPECTED_TRANSPORT = {
    "HTTP":"TCP", "HTTPS":"TCP", "HTTP/3":"UDP", "DNS":"UDP", "MDNS":"UDP",
    "LLMNR":"UDP", "NBNS":"UDP", "SSH":"TCP", "FTP":"TCP", "FTP-DATA":"TCP",
    "FTPS":"TCP", "SMTP":"TCP", "SMTPS":"TCP", "SMTP-SUBMISSION":"TCP", "POP3":"TCP",
    "POP3S":"TCP", "IMAP":"TCP", "IMAPS":"TCP", "RDP":"TCP", "VNC":"TCP", "SMB":"TCP",
    "NTP":"UDP", "BGP":"TCP", "LDAP":"TCP", "LDAPS":"TCP", "SNMP":"UDP",
    "SNMP-TRAP":"UDP", "RTSP":"TCP", "SSDP":"UDP", "ISAKMP":"UDP", "IPSEC-NAT-T":"UDP"
}

df_combined["protocol_pair"] = df_combined.apply(make_protocol_pair, axis=1)
df_combined["presence_label_app"] = df_combined.apply(_presence_app, axis=1)
df_combined["presence_smart"] = df_combined.apply(app_with_transport_if_unusual, axis=1)

# ----------------------------
# Family Order
# ----------------------------
_counts = df_combined.groupby("family")["sample_id"].nunique().sort_values(ascending=False)
family_order = ['Benign'] + [f for f in _counts.index if f != 'Benign']

# ----------------------------
# Plot 1: Normalized flow count per family
# ----------------------------
plt.figure(figsize=(max(12, 1.2*len(family_order)), 6))
flows_per_sample = df_combined.groupby(["family", "sample_id"]).size().groupby("family").mean()
sns.barplot(x=flows_per_sample.index, y=flows_per_sample.values)
plt.title("Avg Flow Count per Sample (Normalized)")
plt.ylabel("Flows per Sample")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# ----------------------------
# Plot 2: % Malicious vs Benign per Family (normalized)
# ----------------------------
mal_counts = df_combined.groupby(["family", "sample_id", "malicious"]).size()
mal_df = mal_counts.groupby(level=[0,2]).sum().unstack(fill_value=0)
mal_pct = mal_df.div(mal_df.sum(axis=1), axis=0) * 100
mal_pct = mal_pct.reindex(index=family_order)

mal_pct.plot(kind="bar", stacked=True, figsize=(12,6), colormap="coolwarm")
plt.title("% of Benign vs Malicious Flows per Family")
plt.ylabel("% of Flows")
plt.xlabel("Family")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Label", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()

pp_ct = pd.crosstab(df_combined["family"], df_combined["presence_smart"])
pp_ct = pp_ct.reindex(index=family_order)
pp_ct = pp_ct[pp_ct.sum().sort_values(ascending=False).index]

plt.figure(figsize=(max(10, 0.28*pp_ct.shape[1]), max(6, 0.6*pp_ct.shape[0])))
sns.heatmap(pp_ct.astype(int), annot=True, fmt="d", cmap="YlGnBu", linewidths=.5,
            cbar_kws={"label": "Flow count"})
plt.title("Protocol Usage by Family — Counts (Smart Label)")
plt.xlabel("Smart Protocol Label")
plt.ylabel("Family")
plt.tight_layout()
plt.show()

# ----------------------------
# Plot 4: Protocol usage by family (Smart Label) — % within family heatmap
# ----------------------------
pp_pct = pp_ct.div(pp_ct.sum(axis=1).replace(0, np.nan), axis=0) * 100
plt.figure(figsize=(max(10, 0.28*pp_pct.shape[1]), max(6, 0.6*pp_pct.shape[0])))
sns.heatmap(pp_pct.round().astype(int), annot=True, fmt="d", cmap="YlOrRd", linewidths=.5,
            cbar_kws={"label": "% of flows in family"})
plt.title("Protocol Usage by Family — % Share (Smart Label)")
plt.xlabel("Smart Protocol Label")
plt.ylabel("Family")
plt.tight_layout()
plt.show()

# --- Binary presence matrix ---
pp_ct_smart_all = pd.crosstab(df_combined["family"], df_combined["presence_smart"])
pp_bin_all = (pp_ct_smart_all > 0).astype(int)
pp_bin_all = pp_bin_all.reindex(index=family_order).fillna(0)
presence_order = pp_bin_all.sum(axis=0).sort_values(ascending=True).index
pp_bin_all = pp_bin_all[presence_order]

# --- Plot ---
plt.figure(figsize=(max(10, 0.3 * len(presence_order)), max(6, 0.6 * len(family_order))))
sns.heatmap(pp_bin_all, annot=True, fmt="d", cmap="Greys", linewidths=.5,
            cbar_kws={"label": "Presence (1=yes, 0=no)"})
plt.title("Protocol Presence by Family — Smart App/Transport")
plt.xlabel("Label")
plt.ylabel("Family")
plt.tight_layout()
plt.show()


# ============================
# Plot 6: Small-multiples — per-family horizontal bars (all protocol pairs)
# ============================
families = list(family_order)
n = len(families)
ncols = 3 if n >= 3 else n
nrows = int(np.ceil(n / ncols)) if n > 0 else 1

fig, axes = plt.subplots(nrows, ncols, figsize=(7*ncols, 4.6*nrows), squeeze=False)

for i, fam in enumerate(families):
    r, c = divmod(i, ncols)
    ax = axes[r][c]
    counts = (df_combined.loc[df_combined["family"] == fam, "protocol_pair"]
              .value_counts().sort_values())  # ascending → longest at top
    ax.barh(counts.index, counts.values)
    ax.set_title(fam)
    ax.set_xlabel("Flow count")
    ax.set_ylabel("app_proto | protocol")

# Hide unused panels
for j in range(i+1, nrows*ncols):
    r, c = divmod(j, ncols)
    axes[r][c].axis("off")

plt.tight_layout()
plt.show()

# ----------------------------
# Boxplots per family — compare Benign vs. Malware (malicious only)
# ----------------------------
box_features = [
    'byte_count', 'packet_count', 'inter_packet_timing_stddev',
    'uri_entropy', 'session_duration', 'burst_count',
    'score', 'dns_entropy', 'dns_sub_entropy', 'domain_entropy'
]

# Filter: keep all Benign, and only malicious from other families
benign_df = df_combined[df_combined['family'] == 'Benign']
malicious_df = df_combined[(df_combined['family'] != 'Benign') & (df_combined['malicious'] == 1)]
filtered_df = pd.concat([benign_df, malicious_df], ignore_index=True)

# Recalculate family_order based on filtered data
_counts = filtered_df['family'].value_counts()
if 'Benign' in _counts.index:
    others = _counts.drop('Benign').index
    filtered_family_order = ['Benign'] + list(others)
else:
    filtered_family_order = list(_counts.index)

for feature in box_features:
    if feature not in filtered_df.columns:
        continue
    plt.figure(figsize=(max(12, 1.2*filtered_df['family'].nunique()), 6))
    sns.boxplot(data=filtered_df, x='family', y=feature, order=filtered_family_order)
    plt.title(f'Boxplot of {feature} by Malware Family (Malicious Only)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    sns.kdeplot(data=filtered_df, x=feature, hue='family', fill=True, common_norm=False)
    plt.title(f"Distribution of {feature} (Benign vs Malware Malicious Only)")
    plt.tight_layout()
    plt.show()

# ----------------------------
# Correlation heatmaps
# ----------------------------
plt.figure(figsize=(10, 8))
sns.heatmap(df_benign[columns_to_convert].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap - Benign")
plt.tight_layout()
plt.show()

# Use only malicious rows 
df_spyware_mal = df_spyware[df_spyware['malicious'] == 1]
if not df_spyware_mal.empty:
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_spyware_mal[columns_to_convert].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap - Spyware (malicious only)")
    plt.tight_layout()
    plt.show()

# ----------------------------
# Simple model + importance
# ----------------------------
X = df_combined[columns_to_convert]
y = df_combined['malicious']
X = SimpleImputer().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

importances = clf.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
sorted_features = np.array(columns_to_convert)[sorted_idx]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[sorted_idx], y=sorted_features, orient='h')
plt.title("Feature Importance from Random Forest")
plt.tight_layout()
plt.show()

# ----------------------------
# t-SNE by family
# ----------------------------
features = df_combined[columns_to_convert].dropna()
labels_family = df_combined.loc[features.index, 'family']
X_scaled = StandardScaler().fit_transform(features)

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=labels_family, palette='tab10', alpha=0.7, legend='full')
plt.title('t-SNE Visualization of Flows by Family')
plt.tight_layout()
plt.show()

# ----------------------------
# PCA explained variance
# ----------------------------
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance')
plt.title('PCA Explained Variance')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ----------------------------
# Pairplot by malicious (optional; can be heavy)
# ----------------------------
pp_df = df_combined[columns_to_convert + ['malicious']].dropna()
if len(pp_df) > 0 and len(pp_df) <= 5000:  # guard to avoid huge pairplots
    sns.pairplot(pp_df, hue='malicious', diag_kind='kde')
    plt.show()

# ----------------------------
# t-SNE by label type & KMeans
# ----------------------------
df_combined['label_type'] = df_combined.apply(
    lambda row: f"{row['family']}_{int(row['malicious'])}", axis=1
)

X2 = df_combined[columns_to_convert].dropna()
labels = df_combined.loc[X2.index, 'label_type']

X2_scaled = StandardScaler().fit_transform(X2)
X2_tsne = TSNE(n_components=2, random_state=42).fit_transform(X2_scaled)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X2_tsne[:, 0], y=X2_tsne[:, 1], hue=labels, alpha=0.6, legend=False)
plt.title('t-SNE of Flows by Label Type')
plt.tight_layout()
plt.show()

n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
cluster_labels = kmeans.fit_predict(X2_scaled)
df_combined['cluster'] = -1
df_combined.loc[X2.index, 'cluster'] = cluster_labels

print("\nCluster distribution by label type:")
print(pd.crosstab(df_combined['cluster'], df_combined['label_type']))
