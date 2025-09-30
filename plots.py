import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from glob import glob
import numpy as np

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans
# ----------------------------
# Load all benign CSVs (flat folder)
# ----------------------------
def load_csvs_flat(folder_path, label='Benign'):
    all_files = glob(os.path.join(folder_path, '*.csv'))
    dfs = []
    for file in all_files:
        try:
            df = pd.read_csv(file)
            df['family'] = label
            dfs.append(df)
        except Exception as e:
            print(f"[ERROR] loading benign {file}: {e}")
    return pd.concat(dfs, ignore_index=True)

# ----------------------------
# Load all ransomware subfolders, keep only malicious == 1
# ----------------------------
def load_csvs_nested_ransomware(parent_folder):
    ransomware_dfs = []
    family_folders = [f.path for f in os.scandir(parent_folder) if f.is_dir()]
    for folder in family_folders:
        family_name = os.path.basename(folder)
        all_files = glob(os.path.join(folder, '*.csv'))
        family_dfs = []
        for file in all_files:
            try:
                df = pd.read_csv(file)
                if 'malicious' not in df.columns:
                    print(f"[WARNING] No 'malicious' column in {file}")
                    continue
                df = df[df['malicious'] == 1]
                if df.empty:
                    continue
                df['family'] = family_name
                family_dfs.append(df)
                print(f"{file} → kept {len(df)} rows")
            except Exception as e:
                print(f"[ERROR] loading ransomware {file}: {e}")
        if family_dfs:
            family_df = pd.concat(family_dfs, ignore_index=True)
            ransomware_dfs.append(family_df)
    if ransomware_dfs:
        return pd.concat(ransomware_dfs, ignore_index=True)
    else:
        print("[WARNING] No ransomware flows found")
        return pd.DataFrame()

# ----------------------------
# Load and combine
# ----------------------------
benign_path = 'analysis_output_benign_baseline_labelled'
ransomware_path = 'Spyware'

df_benign = load_csvs_flat(benign_path)
df_ransomware = load_csvs_nested_ransomware(ransomware_path)

# Confirm nothing is empty
assert not df_benign.empty, "Benign data is empty"
assert not df_ransomware.empty, "Ransomware data is empty"

# Combine properly without overwriting
df_combined = pd.concat([df_benign, df_ransomware], ignore_index=True)

# Convert necessary columns
columns_to_convert = [
    'byte_count', 'packet_count', 'inter_packet_timing_stddev',
    'uri_entropy', 'session_duration', 'burst_count',
    'score', 'dns_entropy', 'dns_sub_entropy', 'domain_entropy'
]
df_combined[columns_to_convert] = df_combined[columns_to_convert].apply(pd.to_numeric, errors='coerce')
df_combined = df_combined.dropna(subset=columns_to_convert, how='all')

print("\n==== DEBUGGING COMBINED DF ====")
print(df_combined[['family', 'malicious']].groupby(['family', 'malicious']).size())


# Final verification
print("\n==== FINAL CHECK ====")
print("Benign samples:", df_benign.shape)
print("Ransomware samples:", df_ransomware.shape)
print("Combined shape:", df_combined.shape)
print("Families:\n", df_combined['family'].value_counts())

plt.figure(figsize=(12, 6))
sns.countplot(data=df_combined, x='family', order=df_combined['family'].value_counts().index)
plt.title("Flow Count per Family")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# ============================================================
# Stacked bar = #benign vs #malicious per family

# ============================================================
family_mal_counts = (
    df_combined
    .groupby(['family', 'malicious'])
    .size()
    .unstack(fill_value=0)
    .rename(columns={0: 'Benign (0)', 1: 'Malicious (1)'})
    .loc[df_combined['family'].value_counts().index]  # same order as earlier plot
)

ax = family_mal_counts.plot(kind='bar', stacked=True, figsize=(12, 6))
ax.set_title("Malicious (1) vs Benign (0) Flows per Family")
ax.set_xlabel("Family")
ax.set_ylabel("Flow Count")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# ============================================================
# 2: Pie charts of protocol usage (by app_proto) per family
# ============================================================
families = list(df_combined['family'].value_counts().index)
n = len(families)
ncols = 3 if n >= 3 else n
nrows = int(np.ceil(n / ncols)) if n > 0 else 1

fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
if nrows * ncols == 1:
    axes = np.array([axes])
axes = axes.ravel()

for ax, fam in zip(axes, families):
    proto_counts = (
        df_combined.loc[df_combined['family'] == fam, 'app_proto']
        .fillna('Unknown').replace('', 'Unknown')
        .value_counts()
    )
    ax.pie(proto_counts.values, labels=proto_counts.index, autopct='%1.1f%%', startangle=140)
    ax.set_title(f"Protocol Distribution (app_proto) — {fam}")

# hide any unused subplots
for j in range(len(families), len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()


# Final inspection plot
plt.figure(figsize=(12, 6))
sns.countplot(data=df_combined, x='family', order=df_combined['family'].value_counts().index)
plt.title("Flow Count per Family")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ----------------------------
# Plot boxplots per family
# ----------------------------
box_features = ['byte_count', 'packet_count', 'inter_packet_timing_stddev',
    'uri_entropy', 'session_duration', 'burst_count',
    'score', 'dns_entropy', 'dns_sub_entropy', 'domain_entropy']

for feature in box_features:
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_combined, x='family', y=feature)
    plt.title(f'Boxplot of {feature} by Malware Family')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    sns.kdeplot(data=df_combined, x=feature, hue='malicious', fill=True)
    plt.title(f"Distribution of {feature} (Malicious vs Benign)")
    plt.show()

    # Correlation heatmap for benign
plt.figure(figsize=(10, 8))
sns.heatmap(df_benign[columns_to_convert].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap - Benign")
plt.show()

# Prepare data
X = df_combined[columns_to_convert]
y = df_combined['malicious']
X = SimpleImputer().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Feature importance plot
importances = clf.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
sorted_features = np.array(columns_to_convert)[sorted_idx]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[sorted_idx], y=sorted_features)
plt.title("Feature Importance from Random Forest")
plt.show()

# Correlation heatmap for ransomware
plt.figure(figsize=(10, 8))
sns.heatmap(df_ransomware[columns_to_convert].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap - Ransomware")
plt.show()

features = df_combined[columns_to_convert].dropna()
labels = df_combined.loc[features.index, 'family']

X_scaled = StandardScaler().fit_transform(features)

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=labels, palette='tab10', alpha=0.7)
plt.title('t-SNE Visualization of Flows by Family')
plt.show()


pca = PCA()
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance')
plt.title('PCA Explained Variance')
plt.grid()
plt.show()

sns.pairplot(df_combined[columns_to_convert + ['malicious']].dropna(), hue='malicious', diag_kind='kde')
plt.show()


df_combined['label_type'] = df_combined.apply(
    lambda row: f"{row['family']}_{int(row['malicious'])}", axis=1
)

X = df_combined[columns_to_convert].dropna()
labels = df_combined.loc[X.index, 'label_type']

X_scaled = StandardScaler().fit_transform(X)
X_tsne = TSNE(n_components=2, random_state=42).fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=labels, alpha=0.6)
plt.title('t-SNE of Flows by Label Type')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

n_clusters = 4  # Try a few values (3-6)
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

df_combined['cluster'] = -1
df_combined.loc[X.index, 'cluster'] = cluster_labels


pd.crosstab(df_combined['cluster'], df_combined['label_type'])