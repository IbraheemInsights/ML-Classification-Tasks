# ================================================================
# TASK 4: Decision Tree Classifier - Pima Indians Diabetes Dataset
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
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
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
  Algorithm : Decision Tree Classifier
  Dataset   : Pima Indians Diabetes Dataset
  Source    : National Institute of Diabetes (USA)
  Extra     : Compare Full Tree vs Restricted Tree (max_depth=3)
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

print(f"\n  Missing values per column:")
print(df.isnull().sum().to_string())
print(f"\n  Total missing values : {df.isnull().sum().sum()}")
print(f"  Duplicate rows       : {df.duplicated().sum()}")

# Check unrealistic zero values
zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
print(f"\n  Unrealistic Zero Values (medically impossible):")
print((df[zero_cols] == 0).sum().to_string())

# Replace zeros with NaN then fill with median
df[zero_cols] = df[zero_cols].replace(0, np.nan)
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

class_counts = y.value_counts().rename({0: "No Diabetes", 1: "Diabetes"})
print(f"\n  Class Distribution:")
print(class_counts.to_string())
print(f"\n  No Diabetes : {class_counts['No Diabetes']} ({class_counts['No Diabetes']/len(y)*100:.1f}%)")
print(f"  Diabetes    : {class_counts['Diabetes']}  ({class_counts['Diabetes']/len(y)*100:.1f}%)")

print(f"\n  Basic Statistics:")
print(df.describe().round(2).to_string())

# EDA Plots
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("STEP 5 – Exploratory Data Analysis (EDA)\nPima Indians Diabetes Dataset (Decision Tree)",
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

# Plot 6: Boxplot Glucose vs Outcome
sns.boxplot(data=df, x='Outcome', y='Glucose', ax=axes[1, 2],
            palette={'0': 'steelblue', '1': 'crimson'})
axes[1, 2].set_title("Boxplot: Glucose vs Class")
axes[1, 2].set_xticklabels(["No Diabetes (0)", "Diabetes (1)"])

plt.tight_layout()
plt.savefig("outputs/task4_step5_eda.png", dpi=150, bbox_inches='tight')
plt.close()
print("\n  EDA plot saved → outputs/task4_step5_eda.png")

# ================================================================
# STEP 6: FEATURE ENGINEERING
# ================================================================
print("\n" + "=" * 60)
print("STEP 6: FEATURE ENGINEERING")
print("=" * 60)

correlation_with_target = df.drop("Outcome", axis=1).corrwith(df["Outcome"]).abs().sort_values(ascending=False)
print("\n  Feature Correlation with Target (Outcome):")
print(correlation_with_target.round(4).to_string())

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
print(f"  Scaling   : StandardScaler applied")
print(f"  random_state = 42")

# ================================================================
# STEP 8: MODEL SELECTION
# ================================================================
print("\n" + "=" * 60)
print("STEP 8: MODEL SELECTION")
print("=" * 60)
print("""
  Selected Algorithm : Decision Tree Classifier
  Why Decision Tree?
    → Easy to understand and visualize
    → Shows exactly which features it used
    → No need for feature scaling (but we apply it anyway)
    → Works well for both numeric and categorical data
    → Can be restricted (max_depth) to avoid overfitting

  Two models will be built and compared:
    → Model 1 : Full Tree    (no depth limit)
    → Model 2 : Restricted   (max_depth=3)
""")

# ================================================================
# STEP 9: MODEL TRAINING
# ================================================================
print("=" * 60)
print("STEP 9: MODEL TRAINING")
print("=" * 60)

# Model 1: Full Decision Tree
dt_full = DecisionTreeClassifier(random_state=42)
dt_full.fit(X_train, y_train)

y_pred_full = dt_full.predict(X_test)
acc_full    = accuracy_score(y_test, y_pred_full)

print(f"\n  Model 1 - Full Tree:")
print(f"    Accuracy   : {acc_full:.4f}")
print(f"    Tree Depth : {dt_full.get_depth()}")
print(f"    Num Leaves : {dt_full.get_n_leaves()}")

# Model 2: Restricted Decision Tree
dt_limited = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_limited.fit(X_train, y_train)

y_pred_limited = dt_limited.predict(X_test)
acc_limited    = accuracy_score(y_test, y_pred_limited)

print(f"\n  Model 2 - Restricted Tree (max_depth=3):")
print(f"    Accuracy   : {acc_limited:.4f}")
print(f"    Tree Depth : {dt_limited.get_depth()}")
print(f"    Num Leaves : {dt_limited.get_n_leaves()}")

# ================================================================
# STEP 10: HYPERPARAMETER OPTIMIZATION
# ================================================================
print("\n" + "=" * 60)
print("STEP 10: HYPERPARAMETER OPTIMIZATION (GridSearchCV)")
print("=" * 60)

param_grid = {
    'max_depth'        : [3, 4, 5, 6, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf' : [1, 2, 4],
    'criterion'        : ['gini', 'entropy']
}

print(f"\n  Running GridSearchCV (5-fold cross-validation)...")
print(f"  Parameter grid: {param_grid}")

grid_search = GridSearchCV(
    estimator  = DecisionTreeClassifier(random_state=42),
    param_grid = param_grid,
    cv         = 5,
    scoring    = 'accuracy',
    n_jobs     = -1,
    verbose    = 0
)
grid_search.fit(X_train, y_train)

print(f"\n  Best Parameters : {grid_search.best_params_}")
print(f"  Best CV Score   : {grid_search.best_score_:.4f}")

dt_best = grid_search.best_estimator_

y_pred_best = dt_best.predict(X_test)
acc_best    = accuracy_score(y_test, y_pred_best)
print(f"  Best Model Accuracy on Test Set : {acc_best:.4f}")

# ================================================================
# STEP 11: MODEL EVALUATION
# ================================================================
print("\n" + "=" * 60)
print("STEP 11: MODEL EVALUATION & COMPARISON")
print("=" * 60)

# --- Full Tree Evaluation ---
cm_full    = confusion_matrix(y_test, y_pred_full)
roc_full   = roc_auc_score(y_test, dt_full.predict_proba(X_test)[:, 1])

print(f"\n  ── Model 1: Full Decision Tree ──")
print(f"  Accuracy  : {acc_full:.4f}")
print(f"  ROC-AUC   : {roc_full:.4f}")
print(f"  Depth     : {dt_full.get_depth()}")
print(f"\nClassification Report (Full Tree):")
print(classification_report(y_test, y_pred_full,
      target_names=["No Diabetes", "Diabetes"]))

# --- Restricted Tree Evaluation ---
cm_limited  = confusion_matrix(y_test, y_pred_limited)
roc_limited = roc_auc_score(y_test, dt_limited.predict_proba(X_test)[:, 1])

print(f"\n  ── Model 2: Restricted Tree (max_depth=3) ──")
print(f"  Accuracy  : {acc_limited:.4f}")
print(f"  ROC-AUC   : {roc_limited:.4f}")
print(f"  Depth     : {dt_limited.get_depth()}")
print(f"\nClassification Report (Restricted Tree):")
print(classification_report(y_test, y_pred_limited,
      target_names=["No Diabetes", "Diabetes"]))

# --- Best Model Evaluation ---
cm_best  = confusion_matrix(y_test, y_pred_best)
roc_best = roc_auc_score(y_test, dt_best.predict_proba(X_test)[:, 1])

print(f"\n  ── Model 3: Best Model (GridSearchCV) ──")
print(f"  Accuracy  : {acc_best:.4f}")
print(f"  ROC-AUC   : {roc_best:.4f}")
print(f"  Depth     : {dt_best.get_depth()}")
print(f"\nClassification Report (Best Model):")
print(classification_report(y_test, y_pred_best,
      target_names=["No Diabetes", "Diabetes"]))

# --- Comparison Summary ---
print("\n" + "=" * 60)
print("  MODEL COMPARISON SUMMARY")
print("=" * 60)
print(f"  {'Model':<30} {'Accuracy':>10} {'ROC-AUC':>10} {'Depth':>8}")
print(f"  {'-'*58}")
print(f"  {'Full Tree':<30} {acc_full:>10.4f} {roc_full:>10.4f} {dt_full.get_depth():>8}")
print(f"  {'Restricted Tree (max_depth=3)':<30} {acc_limited:>10.4f} {roc_limited:>10.4f} {dt_limited.get_depth():>8}")
print(f"  {'Best Model (GridSearchCV)':<30} {acc_best:>10.4f} {roc_best:>10.4f} {dt_best.get_depth():>8}")
print(f"\n  → Full Tree has more depth → risk of OVERFITTING")
print(f"  → Restricted Tree (depth=3) is simpler → better generalization")
print(f"  → Best Model found by GridSearchCV gives optimal balance")

# --- Feature Importances ---
feat_imp_full    = pd.Series(dt_full.feature_importances_,    index=selected_features)
feat_imp_limited = pd.Series(dt_limited.feature_importances_, index=selected_features)
feat_imp_best    = pd.Series(dt_best.feature_importances_,    index=selected_features)

print(f"\n  Feature Importances – Full Tree:")
print(feat_imp_full.sort_values(ascending=False).round(4).to_string())

print(f"\n  Feature Importances – Restricted Tree (max_depth=3):")
print(feat_imp_limited.sort_values(ascending=False).round(4).to_string())

print(f"\n  Feature Importances – Best Model:")
print(feat_imp_best.sort_values(ascending=False).round(4).to_string())

# Evaluation Plots
fig = plt.figure(figsize=(20, 14))
fig.suptitle("STEP 11 – Model Evaluation & Comparison\nDecision Tree Classifier (Diabetes)",
             fontsize=14, fontweight='bold')

# Plot 1: Confusion Matrix - Full Tree
ax1 = fig.add_subplot(3, 3, 1)
sns.heatmap(cm_full, annot=True, fmt='d', cmap='Purples', ax=ax1,
            xticklabels=["No DM", "DM"],
            yticklabels=["No DM", "DM"],
            annot_kws={"size": 14})
ax1.set_title(f"Confusion Matrix\nFull Tree (Acc: {acc_full:.4f})")
ax1.set_xlabel("Predicted"); ax1.set_ylabel("Actual")

# Plot 2: Confusion Matrix - Restricted Tree
ax2 = fig.add_subplot(3, 3, 2)
sns.heatmap(cm_limited, annot=True, fmt='d', cmap='Purples', ax=ax2,
            xticklabels=["No DM", "DM"],
            yticklabels=["No DM", "DM"],
            annot_kws={"size": 14})
ax2.set_title(f"Confusion Matrix\nRestricted Tree (Acc: {acc_limited:.4f})")
ax2.set_xlabel("Predicted"); ax2.set_ylabel("Actual")

# Plot 3: Confusion Matrix - Best Model
ax3 = fig.add_subplot(3, 3, 3)
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Purples', ax=ax3,
            xticklabels=["No DM", "DM"],
            yticklabels=["No DM", "DM"],
            annot_kws={"size": 14})
ax3.set_title(f"Confusion Matrix\nBest Model (Acc: {acc_best:.4f})")
ax3.set_xlabel("Predicted"); ax3.set_ylabel("Actual")

# Plot 4: ROC Curves - All 3 models
ax4 = fig.add_subplot(3, 3, 4)
fpr_f, tpr_f, _ = roc_curve(y_test, dt_full.predict_proba(X_test)[:, 1])
fpr_l, tpr_l, _ = roc_curve(y_test, dt_limited.predict_proba(X_test)[:, 1])
fpr_b, tpr_b, _ = roc_curve(y_test, dt_best.predict_proba(X_test)[:, 1])
ax4.plot(fpr_f, tpr_f, color='purple',    lw=2, label=f"Full Tree (AUC={roc_full:.3f})")
ax4.plot(fpr_l, tpr_l, color='mediumpurple', lw=2, label=f"Restricted (AUC={roc_limited:.3f})")
ax4.plot(fpr_b, tpr_b, color='indigo',    lw=2, label=f"Best Model (AUC={roc_best:.3f})")
ax4.plot([0, 1], [0, 1], 'k--', lw=1)
ax4.set_title("ROC Curves - All Models")
ax4.set_xlabel("FPR"); ax4.set_ylabel("TPR")
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# Plot 5: Accuracy Comparison Bar
ax5 = fig.add_subplot(3, 3, 5)
models = ['Full Tree', 'Restricted\n(depth=3)', 'Best Model\n(GridSearch)']
accs   = [acc_full, acc_limited, acc_best]
bars   = ax5.bar(models, accs,
                 color=['purple', 'mediumpurple', 'indigo'],
                 edgecolor='black', width=0.5)
ax5.set_ylim(0.5, 1.0)
ax5.set_title("Accuracy Comparison")
ax5.set_ylabel("Accuracy")
ax5.grid(True, alpha=0.3, axis='y')
for bar, acc in zip(bars, accs):
    ax5.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.005,
             f"{acc:.4f}", ha='center', va='bottom', fontweight='bold')

# Plot 6: Feature Importances - Best Model
ax6 = fig.add_subplot(3, 3, 6)
feat_imp_best.sort_values().plot(kind='barh', ax=ax6,
                                  color='indigo',
                                  edgecolor='black', linewidth=0.5)
ax6.set_title("Feature Importances\n(Best Model)")
ax6.set_xlabel("Importance Score")
ax6.grid(True, alpha=0.3, axis='x')

# Plot 7: Feature Importances - Full Tree
ax7 = fig.add_subplot(3, 3, 7)
feat_imp_full.sort_values().plot(kind='barh', ax=ax7,
                                  color='purple',
                                  edgecolor='black', linewidth=0.5)
ax7.set_title("Feature Importances\n(Full Tree)")
ax7.set_xlabel("Importance Score")
ax7.grid(True, alpha=0.3, axis='x')

# Plot 8: Feature Importances - Restricted Tree
ax8 = fig.add_subplot(3, 3, 8)
feat_imp_limited.sort_values().plot(kind='barh', ax=ax8,
                                     color='mediumpurple',
                                     edgecolor='black', linewidth=0.5)
ax8.set_title("Feature Importances\n(Restricted Tree, depth=3)")
ax8.set_xlabel("Importance Score")
ax8.grid(True, alpha=0.3, axis='x')

# Plot 9: Decision Tree Visualization (Restricted - max_depth=3)
ax9 = fig.add_subplot(3, 3, 9)
plot_tree(dt_limited,
          feature_names=selected_features,
          class_names=["No DM", "DM"],
          filled=True, ax=ax9, fontsize=6,
          impurity=False, proportion=False)
ax9.set_title("Decision Tree Visualization\n(Restricted Tree, max_depth=3)")

plt.tight_layout()
plt.savefig("outputs/task4_step11_evaluation.png", dpi=150, bbox_inches='tight')
plt.close()
print("\n  Evaluation plot saved → outputs/task4_step11_evaluation.png")

# ================================================================
# STEP 12: MODEL DEPLOYMENT
# ================================================================
print("\n" + "=" * 60)
print("STEP 12: MODEL DEPLOYMENT")
print("=" * 60)

joblib.dump(dt_best,   "outputs/task4_dt_model.pkl")
joblib.dump(dt_limited,"outputs/task4_dt_restricted.pkl")
joblib.dump(scaler,    "outputs/task4_scaler.pkl")

print("  Best Model saved     → outputs/task4_dt_model.pkl")
print("  Restricted Tree saved→ outputs/task4_dt_restricted.pkl")
print("  Scaler saved         → outputs/task4_scaler.pkl")
print("""
  How to load and use later:
  ─────────────────────────────────────────────
  import joblib
  model  = joblib.load('outputs/task4_dt_model.pkl')
  scaler = joblib.load('outputs/task4_scaler.pkl')

  # For new patient data:
  new_data_scaled = scaler.transform(new_data)
  prediction      = model.predict(new_data_scaled)
  # 0 = No Diabetes, 1 = Diabetes
  ─────────────────────────────────────────────
""")

print("=" * 60)
print("  ALL 12 STEPS COMPLETED SUCCESSFULLY!")
print(f"  Full Tree Accuracy       : {acc_full:.4f} ({acc_full*100:.2f}%)")
print(f"  Restricted Tree Accuracy : {acc_limited:.4f} ({acc_limited*100:.2f}%)")
print(f"  Best Model Accuracy      : {acc_best:.4f} ({acc_best*100:.2f}%)")
print(f"  Best Model ROC-AUC       : {roc_best:.4f}")
print("=" * 60)