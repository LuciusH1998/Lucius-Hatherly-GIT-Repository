# ===============================================================
# 🧠 MODEL TRAINING + EVALUATION (with per-model hyperparameter tuning)
# ===============================================================
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ---------------------------------------------------------------
# Preprocessing setup
# ---------------------------------------------------------------
numeric_cols = df_train.select_dtypes(include=['float64', 'int64']).columns.drop('quality', errors='ignore')
categorical_cols = ['beer_style']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
])

preprocessor_nb = ColumnTransformer([
    ('num', MinMaxScaler(), numeric_cols),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
])

# ---------------------------------------------------------------
# Define hyperparameter grids for each model
# ---------------------------------------------------------------
param_grids = {
    "KNN": {
        "clf__n_neighbors": [5, 10, 15, 20, 25, 30],
        "clf__weights": ["uniform", "distance"],
        "clf__p": [1, 2]
    },
    "Logistic Regression": {
        "clf__C": [0.1, 1.0, 5.0],
        "clf__penalty": ["l2"],
        "clf__solver": ["lbfgs", "saga"],
        "clf__max_iter": [1000],
        "clf__class_weight": ["balanced"]
    },
    "Gaussian Naive Bayes": {},  # no hyperparams to tune
    "Complement Naive Bayes": {},  # minimal tuning
    "Random Forest": {
        "clf__n_estimators": [100, 200, 400],
        "clf__max_depth": [10, 20, None],
        "clf__min_samples_split": [2, 4],
        "clf__min_samples_leaf": [1, 2],
        "clf__max_features": ["sqrt", "log2"]
    },
    "Linear SVM": {
        "clf__C": [0.5, 1.0, 2.0, 5.0],
        "clf__kernel": ["linear"],
        "clf__class_weight": ["balanced"]
    },
    "Gaussian Process": {
        "clf__max_iter_predict": [100, 200]
    }
}

# ---------------------------------------------------------------
# Define base models
# ---------------------------------------------------------------
base_models = {
    "KNN": Pipeline([('preprocess', preprocessor), ('clf', KNeighborsClassifier())]),
    "Logistic Regression": Pipeline([('preprocess', preprocessor), ('clf', LogisticRegression(random_state=42))]),
    "Gaussian Naive Bayes": Pipeline([('preprocess', preprocessor_nb), ('clf', GaussianNB())]),
    "Complement Naive Bayes": Pipeline([('preprocess', preprocessor_nb), ('clf', ComplementNB())]),
    "Random Forest": Pipeline([('preprocess', preprocessor), ('clf', RandomForestClassifier(random_state=42))]),
    "Linear SVM": Pipeline([('preprocess', preprocessor), ('clf', SVC(probability=True, random_state=42))]),
    "Gaussian Process": Pipeline([('preprocess', preprocessor), ('clf', GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42))])
}

# ---------------------------------------------------------------
# Train, tune, and evaluate
# ---------------------------------------------------------------
Best_Models = []

for name, model in base_models.items():
    print(f"\n🔹 Evaluating {name} across hyperparameter grid...")
    grid = param_grids.get(name, {})
    best_f1 = -1
    best_params = None
    best_model = None

    for params in ParameterGrid(grid) if grid else [{}]:
        model.set_params(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)
        acc = accuracy_score(y_val, y_pred)

        if f1 > best_f1:
            best_f1 = f1
            best_params = params
            best_model = model

    # Save best model
    Best_Models.append({
        "Model": name,
        "Best Params": best_params,
        "Best F1": best_f1,
        "Best Model": best_model
    })

    # --- Evaluate best version ---
    y_pred = best_model.predict(X_val)
    print(f"\n✅ Best {name} Params: {best_params}")
    print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print(f"Precision: {precision_score(y_val, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_val, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"F1-Score: {best_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, zero_division=0))

    # Confusion matrix
    labels = np.unique(np.concatenate([y_val, y_pred]))
    ConfusionMatrixDisplay(confusion_matrix(y_val, y_pred, labels=labels),
                           display_labels=labels).plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

# ---------------------------------------------------------------
# Compare best models across classifiers
# ---------------------------------------------------------------
results_df = pd.DataFrame([{
    "Model": bm["Model"],
    "Best Params": bm["Best Params"],
    "Best F1": bm["Best F1"]
} for bm in Best_Models]).sort_values(by="Best F1", ascending=False)

print("\n🏆 Best Models by F1-Score:")
print(results_df)

# ---------------------------------------------------------------
# Feature Importance (only for supported models)
# ---------------------------------------------------------------
top_model = Best_Models[0]["Best Model"]
print(f"\n🌟 Feature Importance for {Best_Models[0]['Model']}")

if hasattr(top_model.named_steps['clf'], "feature_importances_"):
    importances = top_model.named_steps['clf'].feature_importances_
    feature_names = (
        numeric_cols.tolist() +
        list(top_model.named_steps['preprocess']
             .named_transformers_['cat']
             .get_feature_names_out(categorical_cols))
    )
    fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    fi_df = fi_df.sort_values(by="Importance", ascending=False)
    print(fi_df.head(10))
    fi_df.plot(kind='barh', x='Feature', y='Importance', legend=False)
    plt.title(f"Top 10 Important Features - {Best_Models[0]['Model']}")
    plt.show()
else:
    print("⚠️ Feature importances not available for this model.")

# ---------------------------------------------------------------
# Error Analysis for Best Model
# ---------------------------------------------------------------
print("\n🔍 Error Analysis on Validation Set")
y_pred_val = top_model.predict(X_val)
val_df = X_val.copy()
val_df["true_quality"] = y_val
val_df["pred_quality"] = y_pred_val
errors = val_df[val_df["true_quality"] != val_df["pred_quality"]]
print(f"Number of misclassifications: {len(errors)} / {len(val_df)}")
print(errors.sample(min(10, len(errors))))
