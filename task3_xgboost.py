# ================================================================
# TASK 3: XGBoost Classifier - Titanic Survival Prediction
# Following the 12-Step ML Workflow
# ================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, roc_auc_score, roc_curve
)
from xgboost import XGBClassifier

# ================================================================
# STEP 1: PROBLEM UNDERSTANDING
# ================================================================
print("=" * 60)
print("STEP 1: PROBLEM UNDERSTANDING")
print("=" * 60)
print("""
  Type      : Binary Classification
  Target    : Passenger Survived (1) or Not (0)
  Algorithm : XGBoost Classifier
  Dataset   : Titanic Passenger Dataset
  Source    : Real RMS Titanic data (April 15, 1912)
""")

# ================================================================
# STEP 2: DATA COLLECTION
# ================================================================
print("=" * 60)
print("STEP 2: DATA COLLECTION")
print("=" * 60)

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df  = pd.read_csv(url)

print(f"  Dataset loaded successfully!")
print(f"  Shape    : {df.shape}  (rows x columns)")
print(f"  Columns  : {df.columns.tolist()}")
print(f"\nFirst 5 rows:\n{df.head()}")

# ================================================================
# STEP 3: DATA CLEANING
# ================================================================
print("\n" + "=" * 60)
print("STEP 3: DATA CLEANING")
print("=" * 60)

print(f"\n  Missing values per column:")
print(df.isnull().sum().to_string())

print(f"\n  Total missing values : {df.isnull().sum().sum()}")
print(f"  Duplicate rows       : {df.duplicated().sum()}")

# Handle missing values
# Age → fill with median
df["Age"].fillna(df["Age"].median(), inplace=True)

# Embarked → fill with mode (most common port)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

# Fare → fill with median
df["Fare"].fillna(df["Fare"].median(), inplace=True)

# Cabin → too many missing (77%), drop the column
df.drop(columns=["Cabin"], inplace=True)

print(f"\n  → Age missing values filled with median ({df['Age'].median():.1f} years)")
print(f"  → Embarked missing values filled with mode ({df['Embarked'].mode()[0]})")
print(f"  → Cabin column dropped (77% missing — not useful)")
print(f"  → Null count after cleaning : {df.isnull().sum().sum()}")
print(f"\n  Dataset is now clean ✓")

# ================================================================
# STEP 4: DATA PREPROCESSING
# ================================================================
print("\n" + "=" * 60)
print("STEP 4: DATA PREPROCESSING")
print("=" * 60)

# Drop columns not useful for prediction
df.drop(columns=["PassengerId", "Name", "Ticket"], inplace=True)
print("  Dropped columns: PassengerId, Name, Ticket (no predictive value)")

# Encode categorical columns (text → numbers)
le = LabelEncoder()
df["Sex"]      = le.fit_transform(df["Sex"])       # male=1, female=0
df["Embarked"] = le.fit_transform(df["Embarked"])  # C=0, Q=1, S=2

print("  Encoded: Sex      → male=1, female=0")
print("  Encoded: Embarked → C=0, Q=1, S=2")
print(f"\n  Final columns: {df.columns.tolist()}")
print(f"  Shape after preprocessing: {df.shape}")

# ================================================================
# STEP 5: EXPLORATORY DATA ANALYSIS (EDA)
# ================================================================
print("\n" + "=" * 60)
print("STEP 5: EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 60)

survival_counts = df["Survived"].value_counts().rename({0: "Did Not Survive", 1: "Survived"})
print(f"\n  Survival Distribution:")
print(survival_counts.to_string())
print(f"\n  Did Not Survive : {survival_counts['Did Not Survive']} ({survival_counts['Did Not Survive']/len(df)*100:.1f}%)")
print(f"  Survived        : {survival_counts['Survived']} ({survival_counts['Survived']/len(df)*100:.1f}%)")

print(f"\n  Basic Statistics:")
print(df.describe().round(2).to_string())

# EDA Plots
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("STEP 5 – Exploratory Data Analysis (EDA)\nTitanic Dataset",
             fontsize=14, fontweight='bold')

# Plot 1: Survival Count
survival_counts.plot(kind='bar', ax=axes[0, 0],
                     color=['crimson', 'steelblue'],
                     edgecolor='black', width=0.5)
axes[0, 0].set_title("Survival Distribution")
axes[0, 0].set_xlabel("Class")
axes[0, 0].set_ylabel("Count")
axes[0, 0].set_xticklabels(["Did Not Survive", "Survived"], rotation=0)
for i, v in enumerate(survival_counts):
    axes[0, 0].text(i, v + 5, str(v), ha='center', fontweight='bold')

# Plot 2: Survival by Sex (0=female, 1=male)
survival_by_sex = df.groupby("Sex")["Survived"].mean().rename({0: "Female", 1: "Male"})
survival_by_sex.plot(kind='bar', ax=axes[0, 1],
                     color=['steelblue', 'crimson'],
                     edgecolor='black', width=0.5)
axes[0, 1].set_title("Survival Rate by Gender\n(Female=0, Male=1)")
axes[0, 1].set_xlabel("Gender")
axes[0, 1].set_ylabel("Survival Rate")
axes[0, 1].set_xticklabels(["Female", "Male"], rotation=0)
axes[0, 1].set_ylim(0, 1)
for i, v in enumerate(survival_by_sex):
    axes[0, 1].text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')

# Plot 3: Survival by Pclass
survival_by_class = df.groupby("Pclass")["Survived"].mean()
survival_by_class.plot(kind='bar', ax=axes[0, 2],
                       color=['gold', 'silver', 'peru'],
                       edgecolor='black', width=0.5)
axes[0, 2].set_title("Survival Rate by Passenger Class")
axes[0, 2].set_xlabel("Class (1=First, 2=Second, 3=Third)")
axes[0, 2].set_ylabel("Survival Rate")
axes[0, 2].set_xticklabels(["1st Class", "2nd Class", "3rd Class"], rotation=0)
axes[0, 2].set_ylim(0, 1)
for i, v in enumerate(survival_by_class):
    axes[0, 2].text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')

# Plot 4: Age Distribution by Survival
sns.histplot(data=df, x='Age', hue='Survived', kde=True,
             ax=axes[1, 0], palette={0: 'crimson', 1: 'steelblue'}, bins=30)
axes[1, 0].set_title("Age Distribution by Survival\n(0=Did Not Survive, 1=Survived)")

# Plot 5: Fare Distribution by Survival
sns.histplot(data=df, x='Fare', hue='Survived', kde=True,
             ax=axes[1, 1], palette={0: 'crimson', 1: 'steelblue'}, bins=30)
axes[1, 1].set_title("Fare Distribution by Survival")

# Plot 6: Correlation Heatmap
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            ax=axes[1, 2], linewidths=0.5, square=True,
            annot_kws={"size": 8})
axes[1, 2].set_title("Correlation Heatmap")

plt.tight_layout()
plt.savefig("outputs/task3_step5_eda.png", dpi=150, bbox_inches='tight')

print("\n  EDA plot saved → outputs/task3_step5_eda.png")

# ================================================================
# STEP 6: FEATURE ENGINEERING
# ================================================================
print("\n" + "=" * 60)
print("STEP 6: FEATURE ENGINEERING")
print("=" * 60)

# Correlation with target
correlation_with_target = df.drop("Survived", axis=1).corrwith(df["Survived"]).abs().sort_values(ascending=False)
print("\n  Feature Correlation with Target (Survived):")
print(correlation_with_target.round(4).to_string())

# Keep features with correlation > 0.1
selected_features = correlation_with_target[correlation_with_target > 0.1].index.tolist()
print(f"\n  Features with correlation > 0.1 : {len(selected_features)}")
print(f"  Features dropped               : {len(df.columns) - 1 - len(selected_features)}")
print(f"  Selected features              : {selected_features}")

X = df[selected_features]
y = df["Survived"]

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
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"  Train set : {X_train_scaled.shape}  ({len(y_train)} samples)")
print(f"  Test set  : {X_test_scaled.shape}   ({len(y_test)} samples)")
print(f"  Scaling   : StandardScaler applied (fit on train, transform on test)")

# ================================================================
# STEP 8: MODEL SELECTION
# ================================================================
print("\n" + "=" * 60)
print("STEP 8: MODEL SELECTION")
print("=" * 60)
print("""
  Selected Algorithm : XGBoost Classifier
  Why XGBoost?
    → Builds trees sequentially (each fixes previous errors)
    → More accurate than Random Forest in most cases
    → Handles missing values internally
    → Built-in regularization to prevent overfitting
    → One of the most winning algorithms in Kaggle competitions
""")

# ================================================================
# STEP 9: MODEL TRAINING
# ================================================================
print("=" * 60)
print("STEP 9: MODEL TRAINING (Default Parameters)")
print("=" * 60)

xgb_default = XGBClassifier(
    n_estimators     = 100,
    max_depth        = 4,
    learning_rate    = 0.1,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    eval_metric      = 'logloss',
    random_state     = 42
)
xgb_default.fit(X_train_scaled, y_train)

y_pred_default = xgb_default.predict(X_test_scaled)
acc_default    = accuracy_score(y_test, y_pred_default)
print(f"\n  Default Model Accuracy : {acc_default:.4f}")

# ================================================================
# STEP 10: HYPERPARAMETER OPTIMIZATION
# ================================================================
print("\n" + "=" * 60)
print("STEP 10: HYPERPARAMETER OPTIMIZATION (GridSearchCV)")
print("=" * 60)

param_grid = {
    'n_estimators' : [100, 200],
    'max_depth'    : [3, 4, 5],
    'learning_rate': [0.05, 0.1],
    'subsample'    : [0.8, 1.0]
}

print(f"\n  Running GridSearchCV (5-fold cross-validation)...")
print(f"  Parameter grid: {param_grid}")

grid_search = GridSearchCV(
    estimator  = XGBClassifier(eval_metric='logloss', random_state=42),
    param_grid = param_grid,
    cv         = 5,
    scoring    = 'accuracy',
    n_jobs     = -1,
    verbose    = 0
)
grid_search.fit(X_train_scaled, y_train)

print(f"\n  Best Parameters : {grid_search.best_params_}")
print(f"  Best CV Score   : {grid_search.best_score_:.4f}")

xgb_best = grid_search.best_estimator_

# ================================================================
# STEP 11: MODEL EVALUATION
# ================================================================
print("\n" + "=" * 60)
print("STEP 11: MODEL EVALUATION")
print("=" * 60)

y_pred      = xgb_best.predict(X_test_scaled)
y_pred_prob = xgb_best.predict_proba(X_test_scaled)[:, 1]

acc     = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)
cm      = confusion_matrix(y_test, y_pred)

print(f"\n  Accuracy     : {acc:.4f}")
print(f"  ROC-AUC      : {roc_auc:.4f}")
print(f"\n  Improvement over default model: {(acc - acc_default)*100:+.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=["Did Not Survive", "Survived"]))

print("Confusion Matrix:")
print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
print(f"  FN={cm[1,0]}  TP={cm[1,1]}")
print(f"\n  → True Negatives  (Correctly predicted Did Not Survive) : {cm[0,0]}")
print(f"  → False Positives (Did Not Survive predicted as Survived): {cm[0,1]}")
print(f"  → False Negatives (Survived predicted as Did Not Survive): {cm[1,0]}")
print(f"  → True Positives  (Correctly predicted Survived)         : {cm[1,1]}")

# Feature Importances
feat_imp = pd.Series(xgb_best.feature_importances_, index=selected_features)
feat_imp_sorted = feat_imp.sort_values(ascending=False)
print(f"\n  Feature Importances (XGBoost):")
print(feat_imp_sorted.round(4).to_string())

# Evaluation Plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("STEP 11 – Model Evaluation\nXGBoost Classifier (Titanic)",
             fontsize=14, fontweight='bold')

# Plot 1: Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=axes[0],
            xticklabels=["Not Survived", "Survived"],
            yticklabels=["Not Survived", "Survived"],
            annot_kws={"size": 16})
axes[0].set_title(f"Confusion Matrix\n(Accuracy: {acc:.4f})")
axes[0].set_xlabel("Predicted Label")
axes[0].set_ylabel("Actual Label")

# Plot 2: ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
axes[1].plot(fpr, tpr, color='darkorange', lw=2.5,
             label=f"ROC Curve (AUC = {roc_auc:.4f})")
axes[1].fill_between(fpr, tpr, alpha=0.1, color='darkorange')
axes[1].plot([0, 1], [0, 1], 'k--', lw=1.5, label="Random Classifier")
axes[1].set_title("ROC Curve")
axes[1].set_xlabel("False Positive Rate (FPR)")
axes[1].set_ylabel("True Positive Rate (TPR)")
axes[1].legend(loc='lower right')
axes[1].grid(True, alpha=0.3)

# Plot 3: Feature Importances
feat_imp_sorted.sort_values().plot(kind='barh', ax=axes[2],
                                   color='darkorange',
                                   edgecolor='black', linewidth=0.5)
axes[2].set_title("Feature Importances (XGBoost)")
axes[2].set_xlabel("Importance Score")
axes[2].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig("outputs/task3_step11_evaluation.png", dpi=150, bbox_inches='tight')

print("\n  Evaluation plot saved → outputs/task3_step11_evaluation.png")

# ================================================================
# STEP 12: MODEL DEPLOYMENT
# ================================================================
print("\n" + "=" * 60)
print("STEP 12: MODEL DEPLOYMENT")
print("=" * 60)

joblib.dump(xgb_best, "outputs/task3_xgb_model.pkl")
joblib.dump(scaler,   "outputs/task3_scaler.pkl")

print("  Model saved  → outputs/task3_xgb_model.pkl")
print("  Scaler saved → outputs/task3_scaler.pkl")
print("""
  How to load and use later:
  ─────────────────────────────────────────────
  import joblib
  model  = joblib.load('outputs/task3_xgb_model.pkl')
  scaler = joblib.load('outputs/task3_scaler.pkl')

  # For new passenger data:
  new_data_scaled = scaler.transform(new_data)
  prediction      = model.predict(new_data_scaled)
  # 0 = Did Not Survive, 1 = Survived
  ─────────────────────────────────────────────
""")

print("=" * 60)
print("  ALL 12 STEPS COMPLETED SUCCESSFULLY!")
print(f"  Final Model Accuracy : {acc:.4f} ({acc*100:.2f}%)")
print(f"  Final ROC-AUC Score  : {roc_auc:.4f}")
print("=" * 60)