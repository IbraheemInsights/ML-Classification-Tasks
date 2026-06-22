# ================================================================
# TASK 2: Logistic Regression - Pima Indians Diabetes Dataset
# Following the 12-Step ML Workflow
# ================================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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
  Target    : Patient has Diabetes (1) or Not (0)
  Algorithm : Logistic Regression
  Dataset   : Pima Indians Diabetes Dataset
  Source    : National Institute of Diabetes (USA)
""")

# ================================================================
# STEP 2: DATA COLLECTION
# ================================================================
print("=" * 60)
print("STEP 2: DATA COLLECTION")
print("=" * 60)

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

col_names = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

df = pd.read_csv(url, header=None, names=col_names)

# Ensure outputs directory exists (avoid errors when saving figures/models)
os.makedirs("outputs", exist_ok=True)

print(f"  Dataset loaded successfully!")
print(f"  Shape    : {df.shape}  (rows x columns)")
print(f"  Features : {len(col_names) - 1}")
print(f"  Target   : Outcome (0 = No Diabetes, 1 = Diabetes)")
print(f"\nFirst 5 rows:\n{df.head()}")

# ================================================================
# STEP 3: DATA CLEANING
# ================================================================
print("\n" + "=" * 60)
print("STEP 3: DATA CLEANING")
print("=" * 60)

# Check missing values
print(f"\n  Missing values per column:")
print(df.isnull().sum().to_string())
print(f"\n  Total missing values : {df.isnull().sum().sum()}")

# Check duplicates
dupes = df.duplicated().sum()
print(f"  Duplicate rows       : {dupes}")

# Check zero values in columns where 0 is medically impossible
zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
print(f"\n  Unrealistic Zero Values (medically impossible):")
print((df[zero_cols] == 0).sum().to_string())

# Replace zeros with NaN
df[zero_cols] = df[zero_cols].replace(0, np.nan)

# Fill NaN with median of each column
df[zero_cols] = df[zero_cols].fillna(df[zero_cols].median())

print(f"\n  → Zeros replaced with column median values")
print(f"  → Null count after fixing : {df.isnull().sum().sum()}")
print(f"\n  Dataset is now clean ✓")

# ================================================================
# STEP 4: DATA PREPROCESSING
# ================================================================
print("\n" + "=" * 60)
print("STEP 4: DATA PREPROCESSING")
print("=" * 60)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

print(f"  Features (X) shape : {X.shape}")
print(f"  Target   (y) shape : {y.shape}")
print(f"  No categorical variables to encode (all numeric)")
print(f"  Scaling method     : StandardScaler (applied after split)")

# ================================================================
# STEP 5: EXPLORATORY DATA ANALYSIS (EDA)
# ================================================================
print("\n" + "=" * 60)
print("STEP 5: EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 60)

print("\n  Class Distribution:")
class_counts = y.value_counts().rename({0: "No Diabetes", 1: "Diabetes"})
print(class_counts.to_string())
print(f"\n  No Diabetes : {class_counts['No Diabetes']} ({class_counts['No Diabetes']/len(y)*100:.1f}%)")
print(f"  Diabetes    : {class_counts['Diabetes']}  ({class_counts['Diabetes']/len(y)*100:.1f}%)")

print("\n  Basic Statistics:")
print(df.describe().round(2).to_string())

# EDA Plots
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("STEP 5 – Exploratory Data Analysis (EDA)\nPima Indians Diabetes Dataset",
             fontsize=14, fontweight='bold')

# Plot 1: Class Distribution
class_counts.plot(kind='bar', ax=axes[0, 0],
                  color=['steelblue', 'crimson'],
                  edgecolor='black', width=0.5)
axes[0, 0].set_title("Class Distribution")
axes[0, 0].set_xlabel("Class")
axes[0, 0].set_ylabel("Count")
axes[0, 0].set_xticklabels(["No Diabetes", "Diabetes"], rotation=0)
for i, v in enumerate(class_counts):
    axes[0, 0].text(i, v + 5, str(v), ha='center', fontweight='bold')

# Plot 2: Glucose Distribution by class
sns.histplot(data=df, x='Glucose', hue='Outcome', kde=True,
             ax=axes[0, 1], palette={0: 'steelblue', 1: 'crimson'}, bins=30)
axes[0, 1].set_title("Glucose Distribution by Class\n(0=No Diabetes, 1=Diabetes)")

# Plot 3: BMI Distribution by class
sns.histplot(data=df, x='BMI', hue='Outcome', kde=True,
             ax=axes[0, 2], palette={0: 'steelblue', 1: 'crimson'}, bins=30)
axes[0, 2].set_title("BMI Distribution by Class")

# Plot 4: Age Distribution by class
sns.histplot(data=df, x='Age', hue='Outcome', kde=True,
             ax=axes[1, 0], palette={0: 'steelblue', 1: 'crimson'}, bins=30)
axes[1, 0].set_title("Age Distribution by Class")

# Plot 5: Correlation Heatmap
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            ax=axes[1, 1], linewidths=0.5, square=True,
            annot_kws={"size": 7})
axes[1, 1].set_title("Correlation Heatmap")

# Plot 6: Boxplot - Glucose vs Outcome
sns.boxplot(data=df, x='Outcome', y='Glucose', ax=axes[1, 2],
            palette={'0': 'steelblue', '1': 'crimson'})
axes[1, 2].set_title("Boxplot: Glucose vs Class")
axes[1, 2].set_xticklabels(["No Diabetes (0)", "Diabetes (1)"])

plt.tight_layout()
plt.savefig("outputs/task2_step5_eda.png", dpi=150, bbox_inches='tight')
plt.close()
print("\n  EDA plot saved → outputs/task2_step5_eda.png")

# ================================================================
# STEP 6: FEATURE ENGINEERING
# ================================================================
print("\n" + "=" * 60)
print("STEP 6: FEATURE ENGINEERING")
print("=" * 60)

# Correlation of each feature with target
correlation_with_target = df.drop("Outcome", axis=1).corrwith(df["Outcome"]).abs().sort_values(ascending=False)
print("\n  Feature Correlation with Target (Outcome):")
print(correlation_with_target.round(4).to_string())

# Keep features with correlation > 0.1
selected_features = correlation_with_target[correlation_with_target > 0.1].index.tolist()
print(f"\n  Features with correlation > 0.1 : {len(selected_features)}")
print(f"  Features dropped               : {len(col_names) - 1 - len(selected_features)}")
print(f"  Selected features              : {selected_features}")

X = df[selected_features]

# ================================================================
# STEP 7: TRAIN-TEST SPLIT
# ================================================================
print("\n" + "=" * 60)
print("STEP 7: TRAIN-TEST SPLIT")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature Scaling
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

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
  Selected Algorithm : Logistic Regression
  Why Logistic Regression?
    → Simple and interpretable model
    → Works well for binary classification
    → Gives probability scores (not just 0 or 1)
    → Model coefficients tell us which features matter most
    → Fast to train even on large datasets
""")

# ================================================================
# STEP 9: MODEL TRAINING
# ================================================================
print("=" * 60)
print("STEP 9: MODEL TRAINING (Default Parameters)")
print("=" * 60)

lr_default = LogisticRegression(max_iter=1000, random_state=42)
lr_default.fit(X_train, y_train)

y_pred_default = lr_default.predict(X_test)
acc_default    = accuracy_score(y_test, y_pred_default)
print(f"\n  Default Model Accuracy : {acc_default:.4f}")

# ================================================================
# STEP 10: HYPERPARAMETER OPTIMIZATION
# ================================================================
print("\n" + "=" * 60)
print("STEP 10: HYPERPARAMETER OPTIMIZATION (GridSearchCV)")
print("=" * 60)

param_grid = {
    'C'        : [0.01, 0.1, 1, 10, 100],   # regularization strength
    'solver'   : ['lbfgs', 'liblinear'],      # optimization algorithm
    'penalty'  : ['l2'],                      # regularization type
    'max_iter' : [1000]
}

print(f"\n  Running GridSearchCV (5-fold cross-validation)...")
print(f"  Parameter grid: {param_grid}")

grid_search = GridSearchCV(
    estimator  = LogisticRegression(random_state=42),
    param_grid = param_grid,
    cv         = 5,
    scoring    = 'accuracy',
    n_jobs     = -1,
    verbose    = 0
)
grid_search.fit(X_train, y_train)

print(f"\n  Best Parameters : {grid_search.best_params_}")
print(f"  Best CV Score   : {grid_search.best_score_:.4f}")

lr_best = grid_search.best_estimator_

# ================================================================
# STEP 11: MODEL EVALUATION
# ================================================================
print("\n" + "=" * 60)
print("STEP 11: MODEL EVALUATION")
print("=" * 60)

y_pred      = lr_best.predict(X_test)
y_pred_prob = lr_best.predict_proba(X_test)[:, 1]

acc     = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)
cm      = confusion_matrix(y_test, y_pred)

print(f"\n  Accuracy     : {acc:.4f}")
print(f"  ROC-AUC      : {roc_auc:.4f}")
print(f"\n  Improvement over default model: {(acc - acc_default)*100:+.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=["No Diabetes", "Diabetes"]))

print("Confusion Matrix:")
print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
print(f"  FN={cm[1,0]}  TP={cm[1,1]}")
print(f"\n  → True Negatives  (Correctly predicted No Diabetes) : {cm[0,0]}")
print(f"  → False Positives (No Diabetes predicted as Diabetes): {cm[0,1]}")
print(f"  → False Negatives (Diabetes predicted as No Diabetes): {cm[1,0]}")
print(f"  → True Positives  (Correctly predicted Diabetes)     : {cm[1,1]}")

# ── Model Coefficients Interpretation ────────────────────────
coef_df = pd.DataFrame({
    "Feature"    : selected_features,
    "Coefficient": lr_best.coef_[0]
}).sort_values("Coefficient", ascending=False)

print("\n  Model Coefficients (what influences diabetes risk):")
print(f"  Positive = increases diabetes risk")
print(f"  Negative = decreases diabetes risk")
print(coef_df.to_string(index=False))

# Evaluation Plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("STEP 11 – Model Evaluation\nLogistic Regression (Diabetes)",
             fontsize=14, fontweight='bold')

# Plot 1: Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axes[0],
            xticklabels=["No Diabetes", "Diabetes"],
            yticklabels=["No Diabetes", "Diabetes"],
            annot_kws={"size": 16})
axes[0].set_title(f"Confusion Matrix\n(Accuracy: {acc:.4f})")
axes[0].set_xlabel("Predicted Label")
axes[0].set_ylabel("Actual Label")

# Plot 2: ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
axes[1].plot(fpr, tpr, color='green', lw=2.5,
             label=f"ROC Curve (AUC = {roc_auc:.4f})")
axes[1].fill_between(fpr, tpr, alpha=0.1, color='green')
axes[1].plot([0, 1], [0, 1], 'k--', lw=1.5, label="Random Classifier")
axes[1].set_title("ROC Curve")
axes[1].set_xlabel("False Positive Rate (FPR)")
axes[1].set_ylabel("True Positive Rate (TPR)")
axes[1].legend(loc='lower right')
axes[1].grid(True, alpha=0.3)

# Plot 3: Coefficients Bar Chart
colors = ['crimson' if c > 0 else 'steelblue' for c in coef_df["Coefficient"]]
axes[2].barh(coef_df["Feature"], coef_df["Coefficient"],
             color=colors, edgecolor='black', linewidth=0.5)
axes[2].axvline(0, color='black', linewidth=1)
axes[2].set_title("Feature Coefficients\n(Red = ↑ Diabetes Risk, Blue = ↓ Risk)")
axes[2].set_xlabel("Coefficient Value")
axes[2].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig("outputs/task2_step11_evaluation.png", dpi=150, bbox_inches='tight')
plt.close()
print("\n  Evaluation plot saved → outputs/task2_step11_evaluation.png")

# ================================================================
# STEP 12: MODEL DEPLOYMENT
# ================================================================
print("\n" + "=" * 60)
print("STEP 12: MODEL DEPLOYMENT")
print("=" * 60)

joblib.dump(lr_best, "outputs/task2_lr_model.pkl")
joblib.dump(scaler,  "outputs/task2_scaler.pkl")

print("  Model saved  → outputs/task2_lr_model.pkl")
print("  Scaler saved → outputs/task2_scaler.pkl")
print("""
  How to load and use later:
  ─────────────────────────────────────────────
  import joblib
  model  = joblib.load('outputs/task2_lr_model.pkl')
  scaler = joblib.load('outputs/task2_scaler.pkl')

  # For new patient data:
  new_data_scaled = scaler.transform(new_data)
  prediction      = model.predict(new_data_scaled)
  # 0 = No Diabetes, 1 = Diabetes
  ─────────────────────────────────────────────
""")

print("=" * 60)
print("  ALL 12 STEPS COMPLETED SUCCESSFULLY!")
print(f"  Final Model Accuracy : {acc:.4f} ({acc*100:.2f}%)")
print(f"  Final ROC-AUC Score  : {roc_auc:.4f}")
print("=" * 60)