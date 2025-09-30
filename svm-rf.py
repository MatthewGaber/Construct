import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Paths 
MAZE_PATH = 'Ransomware/analysis_output_maze_baseline_labelled'     
BENIGN_PATH = 'analysis_output_benign_baseline_labelled' 
EXCLUDE_SUBSTR = "-with-ipinfo-"


# Columns to drop (non-useful or identifiers)
NON_FEATURE_COLUMNS = [
    'flow_id', 'src', 'dst', 'domain', 'protocol', 'transport',
    'reasons', 'timestamp', 'http_uri', 'http_user_agent', 'http_method',
    'dns_qname', 'dns_tld'
]

def load_data(path):
    all_data = []
    for file in os.listdir(path):
        if file.endswith(".csv"):
            if EXCLUDE_SUBSTR in file:
                continue
            df = pd.read_csv(os.path.join(path, file), low_memory=False)
            all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

# Load and combine data
maze_df = load_data(MAZE_PATH)
benign_df = load_data(BENIGN_PATH)

df = pd.concat([maze_df, benign_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Preserve 'malicious' as target label
if 'malicious' not in df.columns:
    raise ValueError("'malicious' column is required and not found.")

# Drop non-feature columns
features_df = df.drop(columns=NON_FEATURE_COLUMNS, errors='ignore')

# Handle missing values based on data type
for col in features_df.columns:
    if features_df[col].dtype == 'object':
        features_df[col] = features_df[col].fillna('missing')
    elif features_df[col].dtype == 'bool':
        features_df[col] = features_df[col].fillna(False).astype(int)
    else:
        features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
        features_df[col] = features_df[col].fillna(features_df[col].mean())

# Ensure 'malicious' exists and is numeric
features_df['malicious'] = features_df['malicious'].astype(int)

# One-hot encode remaining categorical columns
categorical_cols = features_df.select_dtypes(include=['object']).columns
features_df = pd.get_dummies(features_df, columns=categorical_cols)

# Final features and labels
X = features_df.drop(columns=['malicious'], errors='ignore')
y = features_df['malicious']

# Split dataset
print(f"Shape of final dataset: {features_df.shape}")
print(f"Number of malicious samples: {features_df['malicious'].sum()}")
print(f"Number of benign samples: {(features_df['malicious'] == 0).sum()}")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Train SVM
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Evaluation function
def evaluate_model(name, y_true, y_pred):
    print(f"\n--- {name} Evaluation ---")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(y_true, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Benign (0)", "Malicious (1)"],
                yticklabels=["Benign (0)", "Malicious (1)"])
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

# Evaluate both models
evaluate_model("SVM", y_test, y_pred_svm)
evaluate_model("Random Forest", y_test, y_pred_rf)