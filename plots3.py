import os
from glob import glob
from pathlib import Path

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

# ----------------------------
# Load all benign CSVs (flat folder)  — drops VSSMONITORING
# ----------------------------
def load_csvs_flat(folder_path, label='Benign'):
    all_files = glob(os.path.join(folder_path, '*.csv'))
    dfs = []
    for file in all_files:
        try:
            df = pd.read_csv(file)
            # drop artifact
            if 'protocol' in df.columns:
                df = df[df['protocol'].astype(str).str.upper() != "VSSMONITORING"]
            df['family'] = label
            dfs.append(df)
        except Exception as e:
            print(f"[ERROR] loading benign {file}: {e}")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

# ----------------------------
# Load all subfolders (NO malicious filter) 
# ----------------------------
def load_csvs_nested_all(parent_folder):
    family_frames = []
    family_folders = [f.path for f in os.scandir(parent_folder) if f.is_dir()]
    for folder in family_folders:
        family_name = os.path.basename(folder)
        all_files = glob(os.path.join(folder, '*.csv'))
        family_dfs = []
        for file in all_files:
            try:
                df = pd.read_csv(file)
                df['family'] = family_name
                family_dfs.append(df)
                print(f"{file} → loaded {len(df)} rows")
            except Exception as e:
                print(f"[ERROR] loading {file}: {e}")
        if family_dfs:
            family_frames.append(pd.concat(family_dfs, ignore_index=True))
    if not family_frames:
        print("[WARNING] No spyware flows found")
        return pd.DataFrame()
    return pd.concat(family_frames, ignore_index=True)

# ----------------------------
# Paths / load
# ----------------------------
benign_path = 'analysis_output_benign_baseline_labelled'
spyware_path = 'Spyware'  # malware families in subfolders

df_benign = load_csvs_flat(benign_path)
df_spyware = load_csvs_nested_all(spyware_path)

assert not df_benign.empty, "Benign data is empty"
assert not df_spyware.empty, "Spyware data is empty"

# Combine
df_combined = pd.concat([df_benign, df_spyware], ignore_index=True)

# ----------------------------
# Cleaning / typing
# ----------------------------
columns_to_convert = [
    'byte_count', 'packet_count', 'inter_packet_timing_stddev',
    'uri_entropy', 'session_duration', 'burst_count',
    'score', 'dns_entropy', 'dns_sub_entropy', 'domain_entropy'
]
for c in columns_to_convert:
    if c not in df_combined.columns:
        df_combined[c] = np.nan

# numeic
df_combined[columns_to_convert] = df_combined[columns_to_convert].apply(pd.to_numeric, errors='coerce')
df_combined = df_combined.dropna(subset=columns_to_convert, how='all')

# malicious to int 0/1
df_combined['malicious'] = pd.to_numeric(df_combined.get('malicious', 0), errors='coerce').fillna(0).astype(int)

# normalize strings for protocol fields
for col in ['protocol', 'app_proto', 'transport']:
    if col not in df_combined.columns:
        df_combined[col] = ''
    df_combined[col] = df_combined[col].astype(str).fillna('').str.strip()

# ----------------------------
# Normalization logic 
# ----------------------------
def normalize_row(r):
    proto      = str(r.get('protocol', '')).upper()         
    app_proto  = str(r.get('app_proto', '')).upper()        
    transport  = str(r.get('transport', '')).upper()        
    dport      = str(r.get('dport', ''))
    sport      = str(r.get('sport', ''))
    tls_sni    = str(r.get('tls_sni', '') or '')
    http_host  = str(r.get('http_host', '') or '')
    http_meth  = str(r.get('http_method', '') or '')
    has_dns    = bool(str(r.get('dns_qname', '') or ''))

    if proto == "VSSMONITORING":
        proto = ""   


    def as_int(x):
        try: return int(x)
        except: return None
    sp, dp = as_int(sport), as_int(dport)

    def is_transport(x): return x in ('TCP','UDP','ICMP','SCTP','ARP','OTHER','UNKNOWN')
    def is_specific_layer(x): return (x and not is_transport(x))

    # 1) QUIC → HTTP/3
    if proto == 'QUIC' or (transport == 'UDP' and (sp == 443 or dp == 443) and app_proto in ('HTTPS','HTTP-ALT','HTTPS-ALT','TLS')):
        return 'HTTP/3', transport, 'prefer QUIC → HTTP/3'

    # 2) DHCP
    if proto in ('DHCP','BOOTP'):
        return 'DHCP', transport, 'Wireshark DHCP'

    # 3) DNS (incl. over TCP)
    if proto in ('DNS','MDNS','LLMNR','NBNS') or app_proto in ('DNS','MDNS','LLMNR','NBNS') or has_dns:
        return 'DNS', transport, 'DNS by layer/port/feature'

    # 4) HTTP(S)
    if proto in ('HTTP','HTTP2') or http_host or http_meth:
        return 'HTTP', transport, 'HTTP by layer/feature'
    if proto in ('TLS','SSL') or app_proto in ('HTTPS','HTTPS-ALT','HTTP-ALT'):
        if (sp == 443 or dp == 443) or tls_sni or app_proto.startswith('HTTPS'):
            return 'HTTPS', transport, 'TLS/SSL on 443 or SNI'
        if proto in ('TLS','SSL'):
            return 'TLS', transport, 'generic TLS'

    # 5) Common L7s
    for fam in [
        'SSH','FTP','FTP-DATA','FTPS','FTPS-DATA','SMTP','SMTPS','POP3','POP3S',
        'IMAP','IMAPS','RDP','VNC','SMB','NTP','BGP','LDAP','LDAPS','SNMP','RTSP'
    ]:
        if proto == fam or app_proto == fam:
            return fam, transport, 'common L7 by layer/port'

    # 6) fallback to specific highest layer if present
    if is_specific_layer(proto):
        return proto, transport, 'Wireshark highest layer'

    # 7) else specific app_proto if present
    if is_specific_layer(app_proto):
        return app_proto, transport, 'port map heuristic'

    # 8) final fallback: transport only
    return transport, transport, 'fallback transport'

norm = df_combined.apply(lambda r: pd.Series(normalize_row(r), index=['unified_app','carrier','norm_reason']), axis=1)
df_combined = pd.concat([df_combined, norm], axis=1)
df_combined['app_over_transport'] = df_combined['unified_app'] + '|' + df_combined['carrier']

# Reuse existing plotting column name to minimize changes below:
df_combined["protocol_pair"] = df_combined["app_over_transport"]

# ----------------------------
# Port analysis helpers
# ----------------------------
def port_category(p):
    """Return Well-known (0-1023) / Registered (1024-49151) / Dynamic (49152-65535) / Unknown."""
    try:
        p = int(p)
        if 0 <= p <= 1023: return "Well-known"
        if 1024 <= p <= 49151: return "Registered"
        if 49152 <= p <= 65535: return "Dynamic"
        return "Other"
    except:
        return "Unknown"

for col in ["sport","dport"]:
    if col in df_combined.columns:
        df_combined[f"{col}_category"] = df_combined[col].apply(port_category)
    else:
        df_combined[f"{col}_category"] = "Unknown"

# Annotated label (only when dst port is unusual)
def protocol_port_label(row):
    base = row["app_over_transport"]
    try:
        dp = int(row.get("dport", -1))
        if dp not in (80,443,25,53,67,68,110,143,993,995,22):  # annotate notable non-standard ports
            return f"{base}:{dp}"
    except:
        pass
    return base

df_combined["app_transport_port"] = df_combined.apply(protocol_port_label, axis=1)

# ----------------------------
# DEBUG: quick summary
# ----------------------------
print("\n==== DEBUGGING COMBINED DF ====")
print(df_combined[['family', 'malicious']].groupby(['family', 'malicious']).size())

print("\n==== FINAL CHECK ====")
print("Benign samples:", df_benign.shape)
print("Spyware samples:", df_spyware.shape)
print("Combined shape:", df_combined.shape)
print("Families:\n", df_combined['family'].value_counts())

# Consistent family order (by total size)
family_order = df_combined['family'].value_counts().index

# ============================
# Plot 1: Flow count per family (sanity)
# ============================
plt.figure(figsize=(max(12, 1.2*len(family_order)), 6))
sns.countplot(data=df_combined, x='family', order=family_order)
plt.title("Total Flow Count per Family")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# ============================
# Plot 2: Stacked bar — Malicious (1) vs Benign (0) per family
# ============================
stack_counts = (
    pd.crosstab(df_combined["family"], df_combined["malicious"])
      .rename(columns={0: "Benign", 1: "Malicious"})
      .loc[family_order]
)

fig, ax = plt.subplots(figsize=(max(12, 1.2*len(stack_counts)), 6))
stack_counts.plot(kind="bar", stacked=True, ax=ax)
ax.set_title("Malicious vs Benign Flows per Family")
ax.set_xlabel("Family")
ax.set_ylabel("Flow Count")
ax.legend(title="Label", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# ============================
# Plot 3: Protocol usage by family — COUNTS heatmap (using unified app|carrier)
# ============================
pp_ct = pd.crosstab(df_combined["family"], df_combined["protocol_pair"])
pp_ct = pp_ct.loc[family_order]
pp_ct = pp_ct[pp_ct.sum().sort_values(ascending=False).index]  # order columns by global frequency

plt.figure(figsize=(max(10, 0.28*pp_ct.shape[1]), max(6, 0.6*pp_ct.shape[0])))
sns.heatmap(pp_ct, annot=True, fmt="g", cmap="YlGnBu", linewidths=.5,
            cbar_kws={"label": "Flow count"})
plt.title("Protocol Usage by Family — Counts (unified_app | carrier)")
plt.xlabel("App|Transport")
plt.ylabel("Family")
plt.tight_layout()
plt.show()

# ============================
# Plot 4: Protocol usage by family — % within family heatmap
# ============================
pp_pct = pp_ct.div(pp_ct.sum(axis=1).replace(0, np.nan), axis=0) * 100
plt.figure(figsize=(max(10, 0.28*pp_pct.shape[1]), max(6, 0.6*pp_pct.shape[0])))
sns.heatmap(pp_pct, annot=True, fmt=".1f", cmap="YlOrRd", linewidths=.5,
            cbar_kws={"label": "% of flows in family"})
plt.title("Protocol Usage by Family — % Share (unified_app | carrier)")
plt.xlabel("App|Transport")
plt.ylabel("Family")
plt.tight_layout()
plt.show()

# ============================
# Plot 5: Protocol PRESENCE — Binary heatmap
# ============================
pp_bin = (pp_ct > 0).astype(int)
presence_order = pp_bin.sum(axis=0).sort_values(ascending=True).index  # rare → common
pp_bin = pp_bin[presence_order]

plt.figure(figsize=(max(10, 0.28*pp_bin.shape[1]), max(6, 0.6*pp_bin.shape[0])))
sns.heatmap(pp_bin, annot=True, fmt="d", cmap="Greys", linewidths=.5,
            cbar_kws={"label": "Presence (1=yes, 0=no)"})
plt.title("Protocol Presence by Family — Binary (unified_app | carrier)")
plt.xlabel("App|Transport")
plt.ylabel("Family")
plt.tight_layout()
plt.show()

# ============================
# Plot 6: Small-multiples — per-family horizontal bars
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
              .value_counts().sort_values())  
    ax.barh(counts.index, counts.values)
    ax.set_title(fam)
    ax.set_xlabel("Flow count")
    ax.set_ylabel("App|Transport")

# Hide unused panels
for j in range(i+1, nrows*ncols):
    r, c = divmod(j, ncols)
    axes[r][c].axis("off")

plt.tight_layout()
plt.show()

# ============================
# Port category distribution per family (stacked)
# ============================
port_counts = pd.crosstab(df_combined["family"], df_combined["dport_category"]).loc[family_order]

plt.figure(figsize=(max(12, 1.2*len(port_counts)), 6))
port_counts.plot(kind="bar", stacked=True, colormap="tab20c")
plt.title("Destination Port Category Distribution per Family")
plt.ylabel("Flow count")
plt.xlabel("Family")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# ============================
# Top N non-standard destination ports (not well-known)
# ============================
nonstandard = df_combined.loc[df_combined["dport_category"] != "Well-known", "dport"]
top_nonstd = (pd.to_numeric(nonstandard, errors='coerce')
              .dropna()
              .astype(int)
              .value_counts()
              .head(15))

if not top_nonstd.empty:
    plt.figure(figsize=(10, 5))
    sns.barplot(x=top_nonstd.index.astype(str), y=top_nonstd.values, palette="magma")
    plt.title("Top 15 Non-Standard Destination Ports (all families)")
    plt.xlabel("Port")
    plt.ylabel("Flow count")
    plt.tight_layout()
    plt.show()

# ----------------------------
# Boxplots per family 
# ----------------------------
box_features = ['byte_count', 'packet_count', 'inter_packet_timing_stddev',
                'uri_entropy', 'session_duration', 'burst_count',
                'score', 'dns_entropy', 'dns_sub_entropy', 'domain_entropy']

for feature in box_features:
    if feature not in df_combined.columns:
        continue
    plt.figure(figsize=(max(12, 1.2*df_combined['family'].nunique()), 6))
    sns.boxplot(data=df_combined, x='family', y=feature, order=family_order)
    plt.title(f'Boxplot of {feature} by Malware Family')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    sns.kdeplot(data=df_combined, x=feature, hue='malicious', fill=True)
    plt.title(f"Distribution of {feature} (Malicious vs Benign)")
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

# Use only malicious rows for a spyware-only heatmap (optional)
df_spyware_mal = df_spyware[df_spyware['malicious'] == 1] if 'malicious' in df_spyware.columns else pd.DataFrame()
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
