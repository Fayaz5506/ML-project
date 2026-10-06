import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib
import os

# Create figures directory
os.makedirs('figures', exist_ok=True)

# 1. Load Data
df = pd.read_csv('insurance_claims.csv')
print(f"Dataset shape: {df.shape}")

# Drop empty column if present
if '_c39' in df.columns:
    df.drop(columns=['_c39'], inplace=True)

# 2. Preprocessing & Feature Engineering
# Handle '?' values
for col in ['collision_type', 'property_damage', 'police_report_available']:
    df[col] = df[col].replace('?', 'Unknown')

df['authorities_contacted'] = df['authorities_contacted'].fillna('None')

# Feature engineering: Days between policy bind date and incident date
df['policy_bind_date'] = pd.to_datetime(df['policy_bind_date'])
df['incident_date'] = pd.to_datetime(df['incident_date'])
df['days_between_bind_and_incident'] = (df['incident_date'] - df['policy_bind_date']).dt.days

# Drop date and ID columns that are not generalizable
drop_cols = ['policy_number', 'insured_zip', 'incident_location', 'policy_bind_date', 'incident_date']
df_clean = df.drop(columns=drop_cols)

# Encode Target
df_clean['fraud_reported'] = df_clean['fraud_reported'].map({'Y': 1, 'N': 0})

# Identify categorical and numerical columns
cat_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
num_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns.tolist()
num_cols.remove('fraud_reported')

print(f"Categorical features: {len(cat_cols)}, Numerical features: {len(num_cols)}")

# Fit LabelEncoders for categorical columns and keep mapping
encoders = {}
df_encoded = df_clean.copy()
for col in cat_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_clean[col].astype(str))
    encoders[col] = le

# Separate X and y
X = df_encoded.drop(columns=['fraud_reported'])
y = df_encoded['fraud_reported']
feature_names = X.columns.tolist()

# Train/Test Split (80/20 Stratified)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# Scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Resampling with SMOTE for handling class imbalance
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

print(f"Original train class distribution: {np.bincount(y_train)}")
print(f"SMOTE train class distribution: {np.bincount(y_train_res)}")
print(f"Test class distribution: {np.bincount(y_test)}")

# Define Models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

# Train & Evaluate
results = []
cv_results = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    # 1. Train on SMOTE data
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC-AUC': auc,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'TP': tp
    })
    
    # 5-Fold Stratified CV on full feature scaled X
    X_scaled_full = scaler.transform(X)
    cv_acc = cross_validate(model, X_scaled_full, y, cv=skf, scoring=['accuracy', 'precision', 'recall', 'f1'])
    cv_results.append({
        'Model': name,
        'CV Mean Accuracy': cv_acc['test_accuracy'].mean(),
        'CV Mean Precision': cv_acc['test_precision'].mean(),
        'CV Mean Recall': cv_acc['test_recall'].mean(),
        'CV Mean F1-Score': cv_acc['test_f1'].mean()
    })

results_df = pd.DataFrame(results)
cv_df = pd.DataFrame(cv_results)

print("\n--- Model Test Performance (SMOTE) ---")
print(results_df.to_string())

print("\n--- 5-Fold Stratified Cross Validation ---")
print(cv_df.to_string())

# Save Pipeline with Best Model (Random Forest or Gradient Boosting)
best_model_name = results_df.sort_values(by='F1-Score', ascending=False).iloc[0]['Model']
best_model = models[best_model_name]
print(f"\nBest Performing Model selected: {best_model_name}")

pipeline_data = {
    'model': best_model,
    'scaler': scaler,
    'encoders': encoders,
    'cat_cols': cat_cols,
    'num_cols': num_cols,
    'feature_names': feature_names,
    'model_name': best_model_name,
    'df_sample': df_clean.head(10)
}
joblib.dump(pipeline_data, 'fraud_detection_pipeline.pkl')
print("Saved pipeline artifact to fraud_detection_pipeline.pkl")
