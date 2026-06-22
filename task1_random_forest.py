# ================================================================
# TASK 1: Random Forest Classifier - Breast Cancer Dataset
# Following the 12-Step ML Workflow
# ================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, roc_auc_score, roc_curve
)

# ================================================================
# STEP 1: PROBLEM UNDERSTANDING
# ================================================================
print("=" * 60)
print("STEP 1: PROBLEM UNDERSTANDING")
print("=" * 60)
print("""
  Type      : Binary Classification
  Target    : Tumor is Malignant (0) or Benign (1)
  Algorithm : Random Forest Classifier
  Dataset   : Breast Cancer Wisconsin (sklearn built-in)
""")

# ================================================================
# STEP 2: DATA COLLECTION
# ================================================================
print("=" * 60)
print("STEP 2: DATA COLLECTION")
print("=" * 60)

data = load_breast_cancer()
df   = pd.DataFrame(data.data, columns=data.feature_names)
df['Target'] = data.target   # 0 = malignant, 1 = benign

print(f"  Dataset loaded successfully!")
print(f"  Shape  : {df.shape}  (rows x columns)")
print(f"  Features: {len(data.feature_names)}")
print(f"  Target classes: {list(data.target_names)}")
print(f"\nFirst 5 rows:\n{df.head()}")

# ================================================================
# STEP 3: DATA CLEANING
# ================================================================
print("\n" + "=" * 60)
print("STEP 3: DATA CLEANING")
print("=" * 60)

# Check missing values
print(f"\n  Missing values per column:\n{df.isnull().sum().to_string()}")
print(f"\n  Total missing values : {df.isnull().sum().sum()}")

# Check duplicates
dupes = df.duplicated().sum()
print(f"  Duplicate rows       : {dupes}")

# Check outliers using IQR
Q1  = df.drop('Target', axis=1).quantile(0.25)
Q3  = df.drop('Target', axis=1).quantile(0.75)
IQR = Q3 - Q1
outliers = ((df.drop('Target', axis=1) < (Q1 - 1.5 * IQR)) |
            (df.drop('Target', axis=1) > (Q3 + 1.5 * IQR))).sum().sum()
print(f"  Outlier data points  : {outliers}")
print("\n  → No missing values or duplicates found. Dataset is clean.")

# ================================================================
# STEP 4: DATA PREPROCESSING
# ================================================================
print("\n" + "=" * 60)
print("STEP 4: DATA PREPROCESSING")
print("=" * 60)

X = df.drop('Target', axis=1)
y = df['Target']

# Feature Scaling will be applied AFTER train-test split (Step 7)
# to prevent data leakage
print("  Features (X) shape :", X.shape)
print("  Target  (y) shape  :", y.shape)
print("  Scaling method     : StandardScaler (applied after split)")

# ================================================================
# STEP 5: EXPLORATORY DATA ANALYSIS (EDA)
# ================================================================
print("\n" + "=" * 60)
print("STEP 5: EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 60)

print("\nClass Distribution:")
class_counts = y.value_counts().rename({0: "Malignant", 1: "Benign"})
print(class_counts.to_string())
print(f"\n  Malignant : {class_counts['Malignant']} ({class_counts['Malignant']/len(y)*100:.1f}%)")
print(f"  Benign    : {class_counts['Benign']}  ({class_counts['Benign']/len(y)*100:.1f}%)")

print("\nBasic Statistics (first 5 features):")
print(df[list(data.feature_names[:5]) + ['Target']].describe().round(2))

# EDA Plots
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("STEP 5 – Exploratory Data Analysis (EDA)\nBreast Cancer Dataset",
             fontsize=14, fontweight='bold', y=1.01)

# Plot 1: Class Distribution
class_counts.plot(kind='bar', ax=axes[0, 0], color=['crimson', 'steelblue'],
                  edgecolor='black', width=0.5)
axes[0, 0].set_title("Class Distribution")
axes[0, 0].set_xlabel("Class"); axes[0, 0].set_ylabel("Count")
axes[0, 0].set_xticklabels(["Malignant", "Benign"], rotation=0)
for i, v in enumerate(class_counts):
    axes[0, 0].text(i, v + 3, str(v), ha='center', fontweight='bold')

# Plot 2: Distribution of 'mean radius' by class
sns.histplot(data=df, x='mean radius', hue='Target', kde=True,
             ax=axes[0, 1], palette={0: 'crimson', 1: 'steelblue'}, bins=30)
axes[0, 1].set_title("mean radius Distribution by Class\n(0=Malignant, 1=Benign)")

# Plot 3: Distribution of 'mean texture' by class
sns.histplot(data=df, x='mean texture', hue='Target', kde=True,
             ax=axes[0, 2], palette={0: 'crimson', 1: 'steelblue'}, bins=30)
axes[0, 2].set_title("mean texture Distribution by Class")

# Plot 4: Boxplot - mean radius vs class
sns.boxplot(data=df, x='Target', y='mean radius', ax=axes[1, 0],
            palette={'0': 'crimson', '1': 'steelblue'})
axes[1, 0].set_title("Boxplot: mean radius vs Class")
axes[1, 0].set_xticklabels(["Malignant (0)", "Benign (1)"])

# Plot 5: Correlation Heatmap (top 10 features)
top_features = df.drop('Target', axis=1).corrwith(df['Target']).abs().nlargest(10).index
corr_matrix  = df[list(top_features)].corr()
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', ax=axes[1, 1],
            linewidths=0.5, square=True)
axes[1, 1].set_title("Correlation Heatmap\n(Top 10 Features)")

# Plot 6: Pairplot-style scatter (mean radius vs mean perimeter)
scatter = axes[1, 2].scatter(df['mean radius'], df['mean perimeter'],
                              c=df['Target'], cmap='RdBu', alpha=0.6, edgecolors='k', linewidth=0.3)
axes[1, 2].set_title("mean radius vs mean perimeter\n(Red=Malignant, Blue=Benign)")
axes[1, 2].set_xlabel("mean radius"); axes[1, 2].set_ylabel("mean perimeter")
plt.colorbar(scatter, ax=axes[1, 2])

plt.tight_layout()
plt.savefig("outputs/task1_step5_eda.png", dpi=150, bbox_inches='tight')
plt.close()
print("\n  EDA plot saved → task1_step5_eda.png")

# ================================================================
# STEP 6: FEATURE ENGINEERING
# ================================================================
print("\n" + "=" * 60)
print("STEP 6: FEATURE ENGINEERING")
print("=" * 60)

# Find top correlated features with target
correlation_with_target = df.drop('Target', axis=1).corrwith(df['Target']).abs().sort_values(ascending=False)
print("\nTop 10 Features correlated with Target:")
print(correlation_with_target.head(10).round(4).to_string())

# Keep features with correlation > 0.1 (remove near-zero importance ones)
selected_features = correlation_with_target[correlation_with_target > 0.1].index.tolist()
print(f"\n  Features with correlation > 0.1 : {len(selected_features)}")
print(f"  Features dropped               : {len(data.feature_names) - len(selected_features)}")

X = df[selected_features]
print(f"  Final feature count            : {X.shape[1]}")

# ================================================================
# STEP 7: TRAIN-TEST SPLIT
# ================================================================
print("\n" + "=" * 60)
print("STEP 7: TRAIN-TEST SPLIT")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature Scaling (after split to avoid data leakage)
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)   # fit on train only
X_test  = scaler.transform(X_test)        # transform test using train's stats

print(f"  Train set : {X_train.shape}  ({len(y_train)} samples)")
print(f"  Test set  : {X_test.shape}   ({len(y_test)} samples)")
print(f"  Scaling   : StandardScaler applied (fit on train, transform on test)")

# ================================================================
# STEP 8: MODEL SELECTION
# ================================================================
print("\n" + "=" * 60)
print("STEP 8: MODEL SELECTION")
print("=" * 60)
print("""
  Selected Algorithm : Random Forest Classifier
  Why Random Forest?
    → Handles high-dimensional data well (30 features)
    → Robust to outliers
    → Provides feature importance
    → Less prone to overfitting vs single Decision Tree
    → Works well for binary classification
""")

# ================================================================
# STEP 9: MODEL TRAINING
# ================================================================
print("=" * 60)
print("STEP 9: MODEL TRAINING (Default Parameters)")
print("=" * 60)

rf_default = RandomForestClassifier(n_estimators=100, random_state=42)
rf_default.fit(X_train, y_train)

y_pred_default = rf_default.predict(X_test)
acc_default    = accuracy_score(y_test, y_pred_default)
print(f"\n  Default Model Accuracy : {acc_default:.4f}")

# ================================================================
# STEP 10: HYPERPARAMETER OPTIMIZATION
# ================================================================
print("\n" + "=" * 60)
print("STEP 10: HYPERPARAMETER OPTIMIZATION (GridSearchCV)")
print("=" * 60)

param_grid = {
    'n_estimators' : [50, 100, 200],
    'max_depth'    : [None, 5, 10],
    'min_samples_split': [2, 5],
    'max_features' : ['sqrt', 'log2']
}

print("\n  Running GridSearchCV (5-fold cross-validation)...")
print(f"  Parameter grid: {param_grid}")

grid_search = GridSearchCV(
    estimator  = RandomForestClassifier(random_state=42),
    param_grid = param_grid,
    cv         = 5,
    scoring    = 'accuracy',
    n_jobs     = -1,
    verbose    = 0
)
grid_search.fit(X_train, y_train)

print(f"\n  Best Parameters : {grid_search.best_params_}")
print(f"  Best CV Score   : {grid_search.best_score_:.4f}")

# Use best model
rf_best = grid_search.best_estimator_

# ================================================================
# STEP 11: MODEL EVALUATION
# ================================================================
print("\n" + "=" * 60)
print("STEP 11: MODEL EVALUATION")
print("=" * 60)

y_pred      = rf_best.predict(X_test)
y_pred_prob = rf_best.predict_proba(X_test)[:, 1]

acc     = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)
cm      = confusion_matrix(y_test, y_pred)

print(f"\n  Accuracy     : {acc:.4f}")
print(f"  ROC-AUC      : {roc_auc:.4f}")
print(f"\n  Improvement over default model: {(acc - acc_default)*100:+.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Malignant", "Benign"]))

print("Confusion Matrix:")
print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
print(f"  FN={cm[1,0]}  TP={cm[1,1]}")
print(f"\n  → True Negatives  (Correctly predicted Malignant) : {cm[0,0]}")
print(f"  → False Positives (Malignant predicted as Benign)  : {cm[0,1]}")
print(f"  → False Negatives (Benign predicted as Malignant)  : {cm[1,0]}")
print(f"  → True Positives  (Correctly predicted Benign)     : {cm[1,1]}")

# Evaluation Plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("STEP 11 – Model Evaluation\nRandom Forest Classifier (Breast Cancer)",
             fontsize=14, fontweight='bold')

# Plot 1: Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=["Malignant", "Benign"],
            yticklabels=["Malignant", "Benign"],
            annot_kws={"size": 16})
axes[0].set_title(f"Confusion Matrix\n(Accuracy: {acc:.4f})")
axes[0].set_xlabel("Predicted Label")
axes[0].set_ylabel("Actual Label")

# Plot 2: ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
axes[1].plot(fpr, tpr, color='steelblue', lw=2.5, label=f"ROC Curve (AUC = {roc_auc:.4f})")
axes[1].fill_between(fpr, tpr, alpha=0.1, color='steelblue')
axes[1].plot([0, 1], [0, 1], 'k--', lw=1.5, label="Random Classifier")
axes[1].set_title("ROC Curve")
axes[1].set_xlabel("False Positive Rate (FPR)")
axes[1].set_ylabel("True Positive Rate (TPR)")
axes[1].legend(loc='lower right')
axes[1].grid(True, alpha=0.3)

# Plot 3: Top 15 Feature Importances
feat_imp = pd.Series(rf_best.feature_importances_, index=selected_features)
feat_imp.nlargest(15).sort_values().plot(kind='barh', ax=axes[2], color='steelblue',
                                          edgecolor='black', linewidth=0.5)
axes[2].set_title("Top 15 Feature Importances")
axes[2].set_xlabel("Importance Score")
axes[2].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig("outputs/task1_step11_evaluation.png", dpi=150, bbox_inches='tight')
plt.close()
print("\n  Evaluation plot saved → task1_step11_evaluation.png")

# ================================================================
# STEP 12: MODEL DEPLOYMENT
# ================================================================
print("\n" + "=" * 60)
print("STEP 12: MODEL DEPLOYMENT")
print("=" * 60)

# Save the trained model and scaler
joblib.dump(rf_best, "outputs/task1_rf_model.pkl")
joblib.dump(scaler,  "outputs/task1_scaler.pkl")

print("  Model saved  → task1_rf_model.pkl")
print("  Scaler saved → task1_scaler.pkl")
print("""
  How to load and use later:
  ─────────────────────────────────────────────
  import joblib
  model  = joblib.load('task1_rf_model.pkl')
  scaler = joblib.load('task1_scaler.pkl')

  # For new data:
  new_data_scaled = scaler.transform(new_data)
  prediction      = model.predict(new_data_scaled)
  # 0 = Malignant, 1 = Benign
  ─────────────────────────────────────────────
""")

print("=" * 60)
print("  ALL 12 STEPS COMPLETED SUCCESSFULLY!")
print(f"  Final Model Accuracy : {acc:.4f} ({acc*100:.2f}%)")
print(f"  Final ROC-AUC Score  : {roc_auc:.4f}")
print("=" * 60)