# %% [markdown]
# # Competition 1: Beer Quality Prediction 🍺
# 
# This notebook provides starter code for the Beer Quality Prediction competition. For full instructions, problem statement, and grading criteria, please refer to the **README.md** file.
# 
# **Quick Summary**: You will build a classification model to predict beer quality (scores 1-10) based on chemical properties. Submit your predictions to [Kaggle](https://www.kaggle.com/competitions/ift-6390-ift-3395-beer-quality-prediction/).

# %% [markdown]
# ## Import Dependencies

# %%
# Import required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown]
# ## Load and Explore Data

# %%
# Load the training data
train_df = pd.read_csv('data/train.csv', delimiter=';', index_col="id")

# Training Data 
print(f"Training data shape: {train_df.shape}")
print(f"Number of samples (n): {train_df.shape[0]}")
print(f"Number of features (d): {train_df.shape[1] - 1}")  # Subtract 1 for target column
print(f"Columns: {list(train_df.columns)}")
print("\nFirst few rows:")
print(train_df.head())


# %%
# Check data types and basic info
print("\nData types and info for train df:")
print(train_df.info())

# %%
test_df = pd.read_csv('data/test.csv', delimiter=';', index_col="id")

# Test Data 
print(f"Test data shape: {test_df.shape}")
print(f"Number of samples (n): {test_df.shape[0]}")
print(f"Columns: {list(test_df.columns)}")
print("\nFirst few rows:")
print(test_df.head())

# %%
# Check data types and basic info
print("\nData types and info for test df:")
print(test_df.info())

# %%
## Data Exploration and Visualization Continued 
# Listing column names for train and test data 

# Printing summary statistics 
print("\n Summary Statistics")
display(train_df.describe())

# Data Type per feature 
print("\n Data Type per feature")
print(train_df.dtypes.value_counts)


# %%
## Data Visualization and Exploring
## Visualizing Target variables
plt.figure(figsize=(9,3))
sns.countplot(x="quality",data=train_df)
plt.title("Beer Quality Scores Distribution")
plt.xlabel("Beer Quality Score")
plt.ylabel("Count of Beer Quality per Score")
plt.show()

## Plotting Correlation Matrix for numerical Features 
features = train_df.select_dtypes(include=["float64", "int64"]).columns
sns.heatmap(train_df[features].corr(), annot=False, cmap="coolwarm", center=0)
plt.title("Correlation Matrix of Training Data for numerical Features")
plt.show()

# %%
## Detecting Outliers for feature columns 
plt.figure(figsize=(12,4))
sns.boxplot(data=np.log1p(train_df[features.drop(["quality"], errors='ignore')]))
plt.xticks(rotation=90)
plt.title("Numeric Features Boxplot")
plt.show()

# %%
## Analyzing Categorial feature - Beer Style column 
plt.figure(figsize=(10,5))
train_df["beer_style"].value_counts().head().plot(kind="bar", color="blue")
plt.title("Most popular beer styles")
plt.xlabel("Beer Style Type")
plt.ylabel("Beer Style Count")
plt.show()

## Analyzing Beer Stlye relationship with Target 
plt.figure(figsize=(12,4))
sns.boxplot(x="beer_style", y="quality", data=train_df)
plt.xticks(rotation=90)
plt.title("Beer Style and Quality Analysis")
plt.show()

# %% [markdown]
# ## Data Preprocessing
# 
# Implement your preprocessing pipeline (see README.md for suggestions).

# %%
# First copy the data frames to avoid modifying the original data frames
df_train = train_df.copy()
df_test = test_df.copy()

# Investigate missing values
print("Investigating the missing values before imputation.")
print(train_df.isnull().sum().sort_values(ascending=False))

# Fill numeric columns
# Identify numeric columns (exclude 'quality' if present)
numeric_cols = [col for col in df_train.select_dtypes(include=["float64", "int64"]).columns if col != "quality"]

for col in numeric_cols:
    df_train[col] = df_train[col].fillna(df_train[col].median())
    df_test[col] = df_test[col].fillna(df_train[col].median())

# Fill categorical column
df_train["beer_style"] = df_train["beer_style"].fillna(df_train["beer_style"].mode()[0])
df_test["beer_style"] = df_test["beer_style"].fillna(df_train["beer_style"].mode()[0])

# Print results
print("\n Missing values after imputations for Training data:")
print(df_train.isnull().sum().sum())

print("\n Missing values after imputations for Testing data:")
print(df_test.isnull().sum().sum())











# %%
## Removing outlying data based on summary statistics results 
# Outlier clipping per feature distribution
for col in numeric_cols:
    if col in ['free_CO2', 'dissolved_oxygen', 'final_gravity']:
        lower, upper = np.percentile(df_train[col], [5, 95])
    elif col in ['alcohol_ABV', 'fermentation_strength']:
        lower, upper = np.percentile(df_train[col], [1, 99])
    else:
        lower, upper = np.percentile(df_train[col], [2, 98])
    df_train[col] = np.clip(df_train[col], lower, upper)


# %%
## Correlation:
## Printing the Top Correlated Features with quality 
correlation = df_train.drop(columns=["beer_style"]).corr()["quality"].sort_values(ascending=False)
print("\nTop Correlated Featurs: ")
print(correlation.head(15))

# Seeing relationship between quality and beer style
quality_bstyle = df_train.groupby('beer_style')['quality'].mean().sort_values(ascending=False)
print(quality_bstyle)

# %%
# Seeing relationship between alcohol ABV and Fermentation Strength 
df_train[["alcohol_ABV", "fermentation_strength"]].corr()


# %%
# Seeing relationship between bitterness IBU and original gravity
df_train[["bitterness_IBU", "original_gravity"]].corr()

# %%
# Seeing relationship between lactic acid, gypsum levels, and free CO2
df_train[["lactic_acid", "gypsum_level", "free_CO2"]].corr()

# %%
# Dropping Fermentation strength 
#df_train = df_train.drop(columns=['fermentation_strength'])
#df_test = df_test.drop(columns=['fermentation_strength'])

# %%
## Final Analysis of Results after preprocessing 
print("\nTraining data shape")
print(df_train.shape)
print("\nTraining data types")
print(df_train.dtypes)
print("\nTest data shape")
print(df_test.shape)

# %% [markdown]
# ## Prepare Training and Validation Data
# 
# Split your data into training and validation sets.

# %%
## Splitting Data into Train and Test Validation Sets: 
from sklearn.model_selection import train_test_split
# Defining training and validation data
X = df_train.drop(columns="quality")
y= df_train["quality"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

##Printing Training and Training and Validation Set Sizes: 
print(f"Training set sizes for X_train (features) are: {X_train.shape[0]}, Training set sizes for y_train (labels): {y_train.shape}")
print(f"Validation set sizes for X_val (features) are:   {X_val.shape[0]}, Training set sizes for y_val (labels):   {y_val.shape}")


# %%
y_train.value_counts(normalize=True).sort_index()


# %% [markdown]
# ## Train Models
# 
# Train and compare multiple machine learning models.

# %%
# Importing all relevant packages and functions
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Preprocessing and defining numeric vs categorical columns 
numeric_cols = df_train.select_dtypes(include=['float64', 'int64']).columns.drop('quality', errors='ignore')
categorical_cols = ['beer_style']

# Changing beer_style column to numeric via OneHotEncoder and defining preprocessors that will be used for classifiers 
# First classifier uses Standard Scaler and the second uses naive bayes 
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
])

preprocessor_nb = ColumnTransformer([
    ('num', MinMaxScaler(), numeric_cols),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
])

# Hyperparameter grids (with proper clf__ prefixes)
param_options = {
    # Defining KNN hyperparameters 
    "KNN": {
        "clf__n_neighbors": [4, 8, 12, 16, 20, 24],
        "clf__weights": ["uniform", "distance"],
        "clf__p": [1, 2]
    },
    # Defining Logistic Regression
    "Logistic Regression": {
        "clf__C": [0.2, 1.0, 4.5],
        "clf__penalty": ["l2"],
        "clf__solver": ["lbfgs", "saga"],
        "clf__max_iter": [500],
        "clf__class_weight": ["balanced"]
    },
    # Defining Gaussian Naive Bayes
    "Gaussian Naive Bayes": {},  # no hyperparameter tuning
    "Complement Naive Bayes": {},  # no hyper parameter tuning
    # Defining Random Forest Hyperparameters
    "Random Forest": {
        "clf__n_estimators": [100, 250, 400],
        "clf__max_depth": [15, 20, None],
        "clf__min_samples_split": [2, 4],
        "clf__min_samples_leaf": [1, 2],
        "clf__max_features": ["log2", "sqrt"]
    },
    # Defining Linear SVM hyperparameters 
    "Linear SVM": {
        "clf__C": [0.5, 1.0, 2.0, 5.0],
        "clf__kernel": ["linear"],
        "clf__class_weight": ["balanced"]
    },
    # defining Gaussian Process 
    "Gaussian Process": {
        "clf__max_iter_predict": [100, 200]
    }
}

# Foundation Models 
foundation_models = {
    "KNN": Pipeline([('preprocess', preprocessor), ('clf', KNeighborsClassifier())]),
    "Logistic Regression": Pipeline([('preprocess', preprocessor), ('clf', LogisticRegression(random_state=42))]),
    "Gaussian Naive Bayes": Pipeline([('preprocess', preprocessor_nb), ('clf', GaussianNB())]),
    "Complement Naive Bayes": Pipeline([('preprocess', preprocessor_nb), ('clf', ComplementNB())]),
    "Random Forest": Pipeline([('preprocess', preprocessor), ('clf', RandomForestClassifier(random_state=42))]),
    "Linear SVM": Pipeline([('preprocess', preprocessor), ('clf', SVC(probability=True, random_state=42))]),
    "Gaussian Process": Pipeline([('preprocess', preprocessor), ('clf', GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42))])
}

# Training, Hyperparameter Tuning, and Evaluation
optimal_models = []

# Iterating over m_name, and model in foundation_models.items()
for m_name, model in foundation_models.items():
    print(f"\n🔹 Evaluating {m_name} across its hyperparameter space...")
    space = param_options.get(m_name, {})
    # Defining f1, model, and params for now
    best_f1 = -1
    best_model = None
    best_params = None

    # Iterating over ParameterGrid 
    for params in ParameterGrid(space) if space else [{}]:
        model.set_params(**params)
        # Fitting and predicting the models 
        model.fit(X_train, y_train)
        y_predict = model.predict(X_val)

        f1 = f1_score(y_val, y_predict, average="weighted", zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_params = params
            best_model = model
    # Appending Optimal Models 
    optimal_models.append({
        "Model": m_name,
        "Best Params": best_params,
        "Best F1": best_f1,
        "Best Model": best_model
    })

    # --- Evaluate best version ---
    y_pred = best_model.predict(X_val)
    print(f"\nBest {m_name} Params: {best_params}")
    print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print(f"Precision: {precision_score(y_val, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_val, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"F1-Score: {best_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, zero_division=0))

    labels = np.unique(np.concatenate([y_val, y_pred]))
    ConfusionMatrixDisplay(confusion_matrix(y_val, y_pred, labels=labels),
                           display_labels=labels).plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {m_name}")
    plt.show()

# Compare Best Models
results_df = pd.DataFrame([{
    "Model": om["Model"],
    "Best Params": om["Best Params"],
    "Best F1": om["Best F1"]
} for om in optimal_models]).sort_values(by="Best F1", ascending=False)

print("\n Best Models by F1-Score:")
print(results_df)

# Feature Importance (if available)
top_model = optimal_models[0]["Best Model"]
print(f"\nFeature Importance for {optimal_models[0]['Model']}")

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
    plt.title(f"Top 10 Important Features - {optimal_models[0]['Model']}")
    plt.show()
else:
    print("Feature importances not available for this model.")

# Error Analysis for Best Model
print("\nError Analysis on Validation Set")
y_pred_val = top_model.predict(X_val)
val_df = X_val.copy()
val_df["true_quality"] = y_val
val_df["pred_quality"] = y_pred_val
errors = val_df[val_df["true_quality"] != val_df["pred_quality"]]
print(f"Number of misclassifications: {len(errors)} / {len(val_df)}")
print(errors.sample(min(10, len(errors))))





# %%
# ===============================================================
# 🧠 MODEL TRAINING (with integrated preprocessing)
# ===============================================================
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier, KernelDensity
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
import numpy as np

# ---------------------------------------------------------------
# Define Preprocessor (Numeric + Categorical)
# ---------------------------------------------------------------
numeric_cols = df_train.select_dtypes(include=['float64', 'int64']).columns.drop('quality', errors='ignore')
categorical_cols = ['beer_style']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ]
)

preprocessor_nb = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), numeric_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ]
)

# ---------------------------------------------------------------
# Define Models with Preprocessing Pipelines
# ---------------------------------------------------------------
models = {
    "Logistic Regression": Pipeline([
        ('preprocess', preprocessor),
        ('clf', LogisticRegression(max_iter=2000,
        solver='lbfgs',
        C=1.0,
        penalty='l2',
        class_weight='balanced',
        random_state=42))
    ]),
    "Linear SVM": Pipeline([
        ('preprocess', preprocessor),
        ('clf', SVC(kernel="linear", probability=True, random_state=42))
    ]),
    "Random Forest": Pipeline([
        ('preprocess', preprocessor),
        ('clf', RandomForestClassifier(n_estimators=200, max_depth=20,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        class_weight='balanced_subsample'))
    ]),
    "KNN (k=15)": Pipeline([
        ('preprocess', preprocessor),
        ('clf', KNeighborsClassifier(n_neighbors=15,
        weights='distance',
        metric='minkowski',
        p=2))
    ]),
    "KNN (k=30)": Pipeline([
        ('preprocess', preprocessor),
        ('clf', KNeighborsClassifier(n_neighbors=30, weights='distance',
        metric='minkowski',
        p=2))
    ]),
    "Gaussian Naive Bayes": Pipeline([
        ('preprocess', preprocessor_nb),
        ('clf', GaussianNB())
    ]),
    "Complement Naive Bayes": Pipeline([
        ('preprocess', preprocessor_nb),
        ('clf', ComplementNB())
    ]),
    "Polynomial Regression (deg=3)": Pipeline([
        ('preprocess', preprocessor),
        ('poly', PolynomialFeatures(degree=3)),
        ('clf', LinearRegression())
    ]),
    # Optional experimental:
    "Gradient Boosting (experimental)": Pipeline([
        ('preprocess', preprocessor),
        ('clf', HistGradientBoostingClassifier(
    max_depth=6,
    learning_rate=0.1,
    max_iter=600,
    class_weight='balanced',
    random_state=42))
    ]), 
    "Gaussian Process": Pipeline([
        ('preprocess', preprocessor),
        ('clf',  GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42))])
}

# ---------------------------------------------------------------
# Custom Parzen Window Classifier (kept separate)
# ---------------------------------------------------------------
class ParzenWindowClassifier:
    def __init__(self, bandwidth=0.5, preprocessor=None):
        self.bandwidth = bandwidth
        self.models = {}
        self.preprocessor = preprocessor

    def fit(self, X, y):
        if self.preprocessor is not None:
            X = self.preprocessor.fit_transform(X)
        for c in np.unique(y):
            kde = KernelDensity(kernel="gaussian", bandwidth=self.bandwidth)
            kde.fit(X[y == c])
            self.models[c] = kde

    def predict(self, X):
        if self.preprocessor is not None:
            X = self.preprocessor.transform(X)
        log_probs = np.array([self.models[c].score_samples(X) for c in self.models])
        return np.argmax(log_probs, axis=0)


# ---------------------------------------------------------------
# Train Models
# ---------------------------------------------------------------
trained_models = {}
for name, model in models.items():
    print(f"\n🔹 Training {name}...")
    if isinstance(model, ParzenWindowClassifier):
        # Manually encode for Parzen Window
        X_train_encoded = preprocessor.fit_transform(X_train)
        model.fit(X_train_encoded, y_train.values)
    else:
        model.fit(X_train, y_train)
    trained_models[name] = model

print("\n✅ Training completed for all models!")



# %%
## Evaluating Gaussian Process
# ===============================================================
# 🎯 Quick Evaluation for Gaussian Process Model
# ===============================================================
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Select the Gaussian Process model
gp_model = trained_models["Gaussian Process"]

# Make predictions on validation set
print("\n📈 Evaluating Gaussian Process Classifier...")
y_pred_gp = gp_model.predict(X_val)

# Compute metrics
acc = accuracy_score(y_val, y_pred_gp)
print(f"\n✅ Gaussian Process Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_pred_gp, zero_division=0))

# Display confusion matrix
labels = np.unique(np.concatenate([y_val, y_pred_gp]))
cm = confusion_matrix(y_val, y_pred_gp, labels=labels)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(cmap="Blues")
plt.title("Confusion Matrix - Gaussian Process")
plt.show()

# Predict on Kaggle test data
y_test_pred = gp_model.predict(df_test)

# Ensure the output format matches the Kaggle submission template
submission = pd.DataFrame({
    "id": df_test.index if "id" in df_test.columns else np.arange(len(df_test)),
    "quality": y_test_pred.astype(int)
})

# Save CSV for Kaggle
output_path = "predicted_quality_gaussian_process.csv"
submission.to_csv(output_path, index=False)

print(f"✅ Submission file saved as '{output_path}'")
print(submission.head())

# Optional: check prediction distribution
print("\nPrediction Distribution on Test Set:")
print(submission["quality"].value_counts(normalize=True).round(3))



# %% [markdown]
# ## Evaluate Models
# 
# Analyze your best model's performance (see README.md for evaluation guidelines).

# %%
# ===============================================================
# 📊 MODEL EVALUATION (Fixed + Safe for Regression Outputs)
# ===============================================================
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

results = []

for name, model in trained_models.items():
    print(f"\n📈 Evaluating {name}...")

    if isinstance(model, ParzenWindowClassifier):
        y_pred = model.predict(X_val.values)
    elif "Polynomial Regression" in name:
        y_pred = np.round(model.predict(X_val)).astype(int)
        y_pred = np.clip(y_pred, y_val.min(), y_val.max())  # keep in valid range
    else:
        y_pred = model.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    results.append((name, acc))
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_val, y_pred, zero_division=0))

    # --- Fixed Confusion Matrix Display ---
    labels = np.unique(np.concatenate([y_val, y_pred]))
    ConfusionMatrixDisplay(confusion_matrix(y_val, y_pred, labels=labels),
                           display_labels=labels).plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

# Summary
results_df = pd.DataFrame(results, columns=["Model", "Accuracy"]).sort_values(by="Accuracy", ascending=False)
print("\n🏆 Model Performance Summary:")
print(results_df)



# %% [markdown]
# ## Generate Predictions for Kaggle Submission
# 
# Create a CSV file with columns: `id` and `quality` (see README.md for format details).

# %%
# ===============================================================
# 🏆 AUTO-GENERATE KAGGLE SUBMISSION (Best Model Automatically Selected)
# ===============================================================
import pandas as pd
import numpy as np

# 1️⃣ Load the Kaggle sample submission template
submission = pd.read_csv("data/sample_submission.csv")
print("📄 Sample submission loaded:")
print(submission.head())

# 2️⃣ Automatically select best model from results_df
best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]
print(f"\n🏆 Best model selected automatically: {best_model_name}")

# 3️⃣ Check test data format
print("🔍 Test data shape before prediction:", df_test.shape)

# 4️⃣ Generate predictions
# (Since models are trained as pipelines, they handle preprocessing automatically)
test_predictions = best_model.predict(df_test)

# 5️⃣ Prepare submission DataFrame
submission["quality"] = test_predictions

# 6️⃣ Sanity checks
print("\n✅ Submission preview:")
print(submission.head())
print("\nUnique predicted quality values:", np.unique(submission["quality"]))
print("Submission shape:", submission.shape)

# 7️⃣ Save final submission file
output_filename = f"submission_{best_model_name.replace(' ', '_').lower()}.csv"
submission.to_csv(output_filename, index=False)
print(f"\n💾 Kaggle submission file saved as: {output_filename}")




# %% [markdown]
# ---
# Good luck with the competition! 🍺
# 


