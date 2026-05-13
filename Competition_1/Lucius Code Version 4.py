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
# Load the training and test data
train_df = pd.read_csv('data/train.csv', delimiter=';')

# Training Data 
print(f"Training data shape: {train_df.shape}")
print(f"Number of samples (n): {train_df.shape[0]}")
print(f"Number of features (d): {train_df.shape[1] - 1}")  # Subtract 1 for target column
print(f"Columns: {list(train_df.columns)}")
print("\nFirst few rows:")
print(train_df.head())

# Check data types and basic info
print("\nData types and info for train df:")
print(train_df.info())


# %%
test_df = pd.read_csv('data/test.csv', delimiter=';')

# Test Data 
print(f"Test data shape: {test_df.shape}")
print(f"Number of samples (n): {test_df.shape[0]}")
print(f"Columns: {list(test_df.columns)}")
print("\nFirst few rows:")
print(test_df.head())


# Check data types and basic info
print("\nData types and info for test df:")
print(test_df.info())

# %%
## Data Exploration and Visualization Continued 
# Listing column names for train and test data 
print(train_df.columns)

# Printing summary statistics 
print("\n Summary Statistics")
display(train_df.describe())

# Data Type per feature 
print("\n Data Type per feature")
print(train_df.dtypes.value_counts)

# Defining Missing value per feature 
print("\n Missing Values summary statistics")
display(train_df.isnull().sum().sort_values(ascending=False))


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
sns.boxplot(data=train_df[features.drop(["id", "quality"], errors='ignore')])
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
# Function to safely set 'id' as index for any DataFrame
def set_id_index(df, name="DataFrame"):
    print(f"\nProcessing {name}...")
    first_col = df.columns[0]
    print(f"First column in {name}: {first_col}")

    # Drop unnecessary unnamed/index columns
    if first_col.lower() in ["unnamed: 0", "index"]:
        df = df.drop(columns=[first_col])
        print(f"Dropped redundant column '{first_col}'.")

    # Set 'id' as index if it exists
    if 'id' in df.columns:
        df = df.set_index('id')
        print(f" 'id' set as index for {name}.")
    else:
        print(f"Warning: 'id' column not found in {name}. It may already be the index.")

    print(f"{name} shape: {df.shape}")
    print(f"Index name: {df.index.name}\n")
    return df


# Apply to both training and test DataFrames
df_train = set_id_index(df_train, "Training DataFrame")
df_test = set_id_index(df_test, "Test DataFrame")


# %%
## Encoding Categorical Variables 
label_enc = LabelEncoder()
df_train["beer_style"] = label_enc.fit_transform(df_train["beer_style"])
df_test["beer_style"] = label_enc.transform(df_test["beer_style"])

print("\nBeer Styles encoded as integers")
print(label_enc.classes_)

# %%
#numeric_cols = [col for col in df_train.select_dtypes(include=["float64", "int64"]).columns if col != "quality"]

# Scaling Columns for both Train and Test 
#scaling_factor = StandardScaler()

#df_train[numeric_cols] = scaling_factor.fit_transform(df_train[numeric_cols])
#df_test[numeric_cols] = scaling_factor.transform(df_test[numeric_cols])

#print("\nScaling Completed Successfully")


# %%
## Correlation:
## Printing the Top Correlated Features with quality 
correlation = df_train.corr()["quality"].sort_values(ascending=False)
print("\nTop Correlated Featurs: ")
print(correlation.head(15))

# %%
## Dropping columns with negative correlation with quality from training and test set 
drop_cols = []

df_train.drop(columns=drop_cols, inplace=True, errors="ignore")
df_test.drop(columns=drop_cols, inplace=True, errors="ignore")


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
# Defining features and labels
# Features will be all retained columns except quality, and labels column is quality
X = df_train.drop(columns=["quality"], errors="ignore")
y = df_train["quality"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

##Printing Training and Training and Validation Set Sizes: 
print(f"Training set sizes for X_train (features) are: {X_train.shape}, Training set sizes for y_train (labels): {y_train.shape}")
print(f"Validation set sizes for X_val (features) are:   {X_val.shape}, Training set sizes for y_val (labels):   {y_val.shape}")


# %%
# ===============================================================
# ⚡️ L U C I U S  C O D E  V 5  —  F A S T  +  M O R E  A C C U R A T E
# ===============================================================
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import ComplementNB, GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.ensemble import VotingClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===============================================================
# 🔹 MODEL DEFINITIONS (Enhanced Accuracy + Fast Execution)
# ===============================================================

classifiers = [
    # --- Bayes Models (low variance, fast) ---
    ComplementNB(alpha=0.5),               # slightly reduced alpha = less bias
    GaussianNB(var_smoothing=1e-4),        # stronger smoothing improves generalization

    # --- Linear Models ---
    RidgeClassifier(alpha=0.3, solver='lsqr'),   # lower alpha = higher capacity
    SGDClassifier(loss="modified_huber", penalty="elasticnet",
                  alpha=5e-5, l1_ratio=0.3, max_iter=800,
                  tol=1e-3, random_state=42),

    # --- Quadratic Discriminant (Bayes boundary) ---
    QDA(reg_param=0.05),                   # lighter regularization for more flexible fit

    # --- Parzen Window (Gaussian-weighted KNN variants) ---
    KNeighborsClassifier(n_neighbors=4, weights='distance', p=2),
    KNeighborsClassifier(n_neighbors=8, weights='distance', p=2),
    KNeighborsClassifier(n_neighbors=16, weights='distance', p=2),  # Parzen-like
    KNeighborsClassifier(n_neighbors=20, weights='uniform', p=2),   # Histogram
]

classifier_names = [
    "Complement Naive Bayes",
    "Gaussian Naive Bayes",
    "Ridge Classifier (alpha=0.3)",
    "SGD Classifier (Elastic Net)",
    "QDA (reg=0.05)",
    "KNN (k=4, dist)",
    "KNN (k=8, dist)",
    "Parzen Window (k=16)",
    "Histogram Approximation (k=20)"
]

# ===============================================================
# 🔹 TRAINING + EVALUATION
# ===============================================================
results, trained_models = [], {}
print("\n🚀 Starting optimized model training loop...\n")

for name, clf in zip(classifier_names, classifiers):
    print(f"🧩 Training {name}...")
    try:
        # Adaptive scaler per model
        if "Naive Bayes" in name:
            model = make_pipeline(MinMaxScaler(), clf)
        elif "SGD" in name:
            model = make_pipeline(RobustScaler(), clf)
        else:
            model = make_pipeline(StandardScaler(), clf)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_val, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)

        trained_models[name] = model
        results.append({
            "Classifier": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1
        })
        print(f"✅ {name}: Accuracy={acc:.3f}, F1={f1:.3f}")

    except Exception as e:
        print(f"⚠️ Skipped {name}: {e}")

# ===============================================================
# 🔹 SUMMARIZE RESULTS
# ===============================================================
results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
print("\n📊 Final Results Summary:")
display(results_df)

# ===============================================================
# 🔹 ENSEMBLE (Weighted Soft Voting of Top 3 Models)
# ===============================================================
top3 = results_df.head(3)["Classifier"].tolist()
print(f"\n🤝 Building soft-voting ensemble of: {top3}")

estimators = [(n, trained_models[n]) for n in top3]
weights = [3, 2, 1]  # higher weight to top performer
ensemble = VotingClassifier(estimators=estimators, voting='soft', weights=weights, n_jobs=-1)

ensemble.fit(X_train, y_train)
y_pred_ens = ensemble.predict(X_val)

acc_ens = accuracy_score(y_val, y_pred_ens)
f1_ens = f1_score(y_val, y_pred_ens, average='weighted', zero_division=0)
print(f"🏆 Ensemble Accuracy={acc_ens:.3f}, F1={f1_ens:.3f}")

# ===============================================================
# 🔹 CONFUSION MATRIX
# ===============================================================
cm = confusion_matrix(y_val, y_pred_ens)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title(f"Confusion Matrix – Ensemble ({', '.join(top3)})")
plt.show()

# ===============================================================
# 🔹 SUBMISSION
# ===============================================================
best_model = ensemble if acc_ens > results_df.iloc[0]["Accuracy"] else trained_models[results_df.iloc[0]["Classifier"]]
print("\n📤 Generating Kaggle submission...")

y_test_pred = best_model.predict(df_test[X_train.columns])
submission = pd.DataFrame({"id": df_test.index, "quality": y_test_pred})
submission.to_csv("predicted_quality_V5.csv", index=False)
print("✅ Saved → predicted_quality_V5.csv")


# %% [markdown]
# ## Train Models
# 
# Train and compare multiple machine learning models.

# %%
# ===============================================================
# 🧠 MODEL TRAINING AND EVALUATION
# ===============================================================
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.linear_model import LogisticRegression, LinearRegression, BayesianRidge
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------
# Define classifiers and names
# -----------------------------------------
classifiers = [
    #GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
    #GaussianNB(),                    # Standard Gaussian Naive Bayes
    ComplementNB(),                  # Additional Bayesian variant
    LogisticRegression(max_iter=1000, random_state=42),
    #LinearRegression(),              # Regression model mapped to class labels
    #BayesianRidge(),                 # Bayesian Linear Regression
    KNeighborsClassifier(n_neighbors=2),
    KNeighborsClassifier(n_neighbors=4),
    KNeighborsClassifier(n_neighbors=6),
    KNeighborsClassifier(n_neighbors=8),
    KNeighborsClassifier(n_neighbors=22, weights='uniform'),  # Histogram Approximation
    KNeighborsClassifier(n_neighbors=16, weights='distance')  # Parzen Window Approximation
]

classifier_names = [
    #"Gaussian Process Classifier",
    #"Gaussian Naive Bayes",
    "Complement Naive Bayes",
    "Logistic Regression",
    #"Linear Regression",
    #"Bayesian Ridge Regression",
    "KNN (k=2)",
    "KNN (k=4)",
    "KNN (k=6)",
    "KNN (k=8)",
    "Histogram Approximation",
    "Parzen Window Approximation"
]

# -----------------------------------------
# Initialize results containers
# -----------------------------------------
results = []
trained_models = {}

# -----------------------------------------
# Train and evaluate each classifier
# -----------------------------------------
for name, clf in zip(classifier_names, classifiers):
    print(f"\n🚀 Training {name}...")
    try:
        model = make_pipeline(StandardScaler(), clf)
        model.fit(X_train, y_train)

        # Predict on validation data
        y_pred = model.predict(X_val)

        # Handle regression-based models (round to nearest class)
        if name in ["Linear Regression", "Bayesian Ridge Regression"]:
            y_pred = np.clip(np.round(y_pred), y_train.min(), y_train.max())

        # Compute metrics
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_val, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)

        # Store model and metrics
        trained_models[name] = model
        results.append({
            "Classifier": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1
        })

        print(f"✅ {name}: "
              f"Accuracy={acc:.3f}, Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}")

    except Exception as e:
        print(f"⚠️ Skipped {name} — error: {e}")

# -----------------------------------------
# Summarize model performance
# -----------------------------------------
results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
print("\n📊 Final Results Summary:")
display(results_df)

# -----------------------------------------
# Identify best model and visualize confusion matrix
# -----------------------------------------
best_model_name = results_df.iloc[0]["Classifier"]
best_model = trained_models[best_model_name]




# %% [markdown]
# ## Evaluate Models
# 
# Analyze your best model's performance (see README.md for evaluation guidelines).

# %%


# %% [markdown]
# ## Generate Predictions for Kaggle Submission
# 
# Create a CSV file with columns: `id` and `quality` (see README.md for format details).

# %%
# Generating Parzen Window Approximation for Kaggle 
best_m_name = "Parzen Window Approximation"
best_model = trained_models[best_m_name]

features = X_train.columns
predicted_data = df_test[features]



# Generate Predictions
y_predictions = best_model.predict(predicted_data)
predictions = pd.DataFrame({"id":predicted_data.index, "quality":y_predictions})
output = "predicted_quality_parzen_window.csv"
predictions.to_csv(output, index=False)




# %% [markdown]
# ---
# Good luck with the competition! 🍺
# 


