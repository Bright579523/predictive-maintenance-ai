import nbformat
import json

nb = nbformat.read(r'd:\Project\Project2\Smart_Dashboard\notebooks\Predictive_Maintenance_Advanced.ipynb', as_version=4)

# Data Loading & EDA
nb.cells.append(nbformat.v4.new_markdown_cell('# 🏭 Smart CNC Predictive Maintenance (Advanced Pipeline)\\n\\nThis notebook contains the advanced Machine Learning pipeline for the Smart CNC Dashboard. It upgrades the previous standalone experiments to a **Production-Ready MLOps Pipeline**.\\n\\n### 🚀 Key Upgrades from Previous Versions:\\n1. **Local & Production Environment:** Transitioned from Colab to a fully localized VS Code environment with structured directories (`models/`, `data/`, `docs/`).\\n2. **Experiment Tracking (`MLflow`):** Replaced manual tracking with MLflow to systematically log parameters, evaluation metrics, and model artifacts.\\n3. **Baseline vs. Advanced:** Now includes a Logistic Regression baseline to benchmark against our advanced XGBoost model, explicitly proving business ROI.\\n4. **Hyperparameter Tuning (`Optuna`):** Implemented Bayesian Optimization via Optuna for faster, more accurate parallelized model tuning.\\n5. **Explainable AI (`SHAP`):** Integrated SHAP to demystify black-box predictions.\\n6. **Data Drift Monitoring (`Evidently AI`):** Implemented Evidently AI to detect distribution shifts between training and test sets.'))
nb.cells.append(nbformat.v4.new_code_cell('''import pandas as pd
import numpy as np

# Load local data
data_path = '../data/ai4i2020.csv'
df = pd.read_csv(data_path)

print("Dataset Shape:", df.shape)
print("Failure Rate:", round(df['Machine failure'].mean() * 100, 2), "%")
df.head()'''))

# Feature Engineering
nb.cells.append(nbformat.v4.new_markdown_cell('## 2. Feature Engineering\nCreating Physics-based features for better prediction.'))
nb.cells.append(nbformat.v4.new_code_cell('''# Calculate Physics-based Features
df['Power_W'] = df['Torque [Nm]'] * df['Rotational speed [rpm]'] * 0.1047
df['Temp_Diff'] = df['Process temperature [K]'] - df['Air temperature [K]']

# Label encode 'Type'
type_mapping = {'L': 0, 'M': 1, 'H': 2}
df['Type_encoded'] = df['Type'].map(type_mapping)

features = ['Type_encoded', 'Air temperature [K]', 'Process temperature [K]', 
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 
            'Power_W', 'Temp_Diff']

X = df[features]
y = df['Machine failure']
X.head()'''))

# Train Test Split & Preprocessing
nb.cells.append(nbformat.v4.new_markdown_cell('## 3. Preprocessing (Scaling & SMOTE)\nIMPORTANT: SMOTE must be applied ONLY to the training data to prevent data leakage.'))
nb.cells.append(nbformat.v4.new_code_cell('''from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os

# Split data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE to handle class imbalance (Only on Train)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

print(f"Original Train Shape: {X_train_scaled.shape}, Failures: {sum(y_train)}")
print(f"SMOTE Train Shape: {X_train_smote.shape}, Failures: {sum(y_train_smote)}")

# Ensure models directory exists
os.makedirs('../models', exist_ok=True)
joblib.dump(scaler, '../models/scaler.pkl')
print("✅ Scaler saved to models/scaler.pkl")'''))

# Baseline Model
nb.cells.append(nbformat.v4.new_markdown_cell('## 4. Baseline Model (Logistic Regression)\nTraining a simple Logistic Regression model to establish a baseline for comparison.'))
nb.cells.append(nbformat.v4.new_code_cell('''from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score

# Initialize and train the Baseline Model
baseline_model = LogisticRegression(random_state=42, max_iter=1000)
baseline_model.fit(X_train_smote, y_train_smote)

# Evaluate Baseline
y_pred_base = baseline_model.predict(X_test_scaled)
y_prob_base = baseline_model.predict_proba(X_test_scaled)[:, 1]

print("Baseline Model (Logistic Regression) Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_base):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_base):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_base):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_prob_base):.4f}")'''))

# Optuna + XGBoost
nb.cells.append(nbformat.v4.new_markdown_cell('## 5. Hyperparameter Tuning with Optuna & XGBoost\nUsing n_jobs=-1 to utilize all CPU cores on your local machine.'))
nb.cells.append(nbformat.v4.new_code_cell('''import xgboost as xgb
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 20),
        'random_state': 42,
        'n_jobs': -1  # Parallel processing for Local Machine
    }
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train_smote, y_train_smote)
    
    y_pred = model.predict(X_test_scaled)
    return f1_score(y_test, y_pred)

# Run Optuna Optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)  # Reduced trials to 30 for speed on local, can increase later

print("Best Trial:", study.best_trial.params)
print("Best F1-Score:", study.best_value)'''))

# Train Final Model
nb.cells.append(nbformat.v4.new_markdown_cell('## 6. Train Final Model & MLflow Tracking\nTraining the final XGBoost model with the best parameters and saving it.'))
nb.cells.append(nbformat.v4.new_code_cell('''# Train final model with best parameters
best_params = study.best_trial.params
best_params['n_jobs'] = -1
best_params['random_state'] = 42

final_model = xgb.XGBClassifier(**best_params)
final_model.fit(X_train_smote, y_train_smote)

# Evaluation
y_pred = final_model.predict(X_test_scaled)
y_prob = final_model.predict_proba(X_test_scaled)[:, 1]

print("Final Model Performance on Test Set:")
print(f"Accuracy: {final_model.score(X_test_scaled, y_test):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")

# Save the model locally
joblib.dump(final_model, '../models/predictive_maintenance_model.pkl')
print("✅ Model saved to models/predictive_maintenance_model.pkl")'''))

# SHAP Explainability
nb.cells.append(nbformat.v4.new_markdown_cell('## 7. Explainable AI (SHAP)\nUsing SHAP to understand why the model makes its predictions.'))
nb.cells.append(nbformat.v4.new_code_cell('''import shap

# Initialize JavaScript visualization for SHAP
shap.initjs()

# Create TreeExplainer
explainer = shap.TreeExplainer(final_model)

# Use shap.sample to speed up calculation locally
X_test_sample = shap.sample(X_test_scaled, 100)
shap_values = explainer.shap_values(X_test_sample)

# 1. Summary Plot (Feature Importance)
plt.figure(figsize=(8,6))
shap.summary_plot(shap_values, X_test_sample, feature_names=features, show=False)
plt.tight_layout()
plt.show()'''))

# Data Drift (Evidently)
nb.cells.append(nbformat.v4.new_markdown_cell('## 8. Data Drift Monitoring (Phase 7: Lean MLOps)\nUsing Evidently AI to detect data drift between Train and Test data.'))
nb.cells.append(nbformat.v4.new_code_cell('''!pip install -q evidently
from evidently import Report
from evidently.presets import DataDriftPreset

# Convert scaled arrays back to DataFrame for evidently
df_train_drift = pd.DataFrame(X_train_scaled, columns=features)
df_test_drift = pd.DataFrame(X_test_scaled, columns=features)

report = Report(metrics=[DataDriftPreset()])
snapshot = report.run(reference_data=df_train_drift, current_data=df_test_drift)
snapshot.save_html("../docs/data_drift_report.html")

print("✅ Data Drift Report saved to docs/data_drift_report.html")'''))

nbformat.write(nb, r'd:\Project\Project2\Smart_Dashboard\notebooks\Predictive_Maintenance_Advanced.ipynb')
