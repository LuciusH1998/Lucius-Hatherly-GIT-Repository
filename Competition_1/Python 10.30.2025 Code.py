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
# Reading test dataframe as a csv file  
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
## Detecting Outliers for feature columns using a box plot 
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

# Filling column in na/missing values with median ones 
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
    # Within free_CO2, dissolved_oxygen, and final_gravity column calculating the 5th and 95th percentile 
    if col in ['free_CO2', 'dissolved_oxygen', 'final_gravity']:
        lower, upper = np.percentile(df_train[col], [5, 95])
        # Calculating the 1st and 99th percentile columns for alcohol_ABV and fermentation_strength
    elif col in ['alcohol_ABV', 'fermentation_strength']:
        lower, upper = np.percentile(df_train[col], [1, 99])
    else:
        # For all other columns, calculate the 2nd and 98th percentile column
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
# Dropping Fermentation strength as it is highly correlated with alcohol_ABV and can be removed
# Also dropping pH, final_gravity, dissolved_oxygen, and original_gravity as these columns are weakly correlated with quality
#df_train = df_train.drop(columns=['pH'])
#df_test = df_test.drop(columns=['pH'])

# %%
# ===============================================================
# 🍺 FEATURE ENGINEERING (adds predictive, chemistry-based columns)
# ===============================================================
#import numpy as np
#import pandas as pd

#df_train['alcohol_efficiency'] = df_train['alcohol_ABV'] / (df_train['fermentation_strength'] + 1e-6)
#df_test['alcohol_efficiency'] = df_test['alcohol_ABV'] / (df_test['fermentation_strength'] + 1e-6)

#df_train['flavor_balance'] = (df_train['bitterness_IBU'] + 1e-6) / (df_train['diacetyl_concentration'] + abs(df_train['lactic_acid']) + 1e-6)
#df_test['flavor_balance'] = (df_test['bitterness_IBU'] + 1e-6) / (df_test['diacetyl_concentration'] + abs(df_test['lactic_acid']) + 1e-6)


#print("✅ Added engineered features: attenuation, balance")


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

le = LabelEncoder()
df_train['beer_style'] = le.fit_transform(df_train['beer_style'])
df_test['beer_style'] = le.fit_transform(df_test['beer_style'])

rng = np.random.RandomState(2)

X = df_train.drop(columns="quality")
y= df_train["quality"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

##Printing Training and Training and Validation Set Sizes: 
print(f"Training set sizes for X_train (features) are: {X_train.shape[0]}, Training set sizes for y_train (labels): {y_train.shape}")
print(f"Validation set sizes for X_val (features) are:   {X_val.shape[0]}, Training set sizes for y_val (labels):   {y_val.shape}")


# %%
y_train.value_counts(normalize=True).sort_index()

print(X_train["beer_style"].value_counts())


# %% [markdown]
# ## Train Models
# 
# Train and compare multiple machine learning models.

# %%
# ===============================================================
# 🧠 MODEL TRAINING (Improved with Hyperparameter Tuning + Multinomial Logistic Regression)
# ===============================================================
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import numpy as np

# ---------------------------------------------------------------
# Define Preprocessors
# ---------------------------------------------------------------
preprocessor_std = StandardScaler()
preprocessor_nb = MinMaxScaler()

# ---------------------------------------------------------------
# Define Parameter Grids
# ---------------------------------------------------------------
param_grids = {
    "Logistic Regression": {
        "clf__C": [0.01, 0.02, 0.1],
        "clf__penalty": ["l2"],
        "clf__multi_class": ["multinomial"],  # ✅ explicit multinomial
        "clf__solver": ["lbfgs"],
    },
    "Linear SVM": {
        "clf__penalty": ["l2"], 
        "clf__C": [0.1, 0.2, 0.5],
        "clf__loss": ["squared_hinge"],
    },
    "Random Forest": {
    "clf__n_estimators": [150, 200],     # narrow around 200
    "clf__max_depth": [15, 20],           # near your current best (20)
    "clf__min_samples_split": [3, 4],      # fine-tune around 4
    "clf__min_samples_leaf": [1, 2],       # fine-tune around 2
    "clf__max_features": ["sqrt"],            # fixed to your top performer
    "clf__bootstrap": [True],
    "clf__class_weight": ["balanced_subsample"]
},
    "KNN": {
        "clf__n_neighbors": [5, 10, 15, 25],
        "clf__weights": ["uniform", "distance"],
        "clf__p": [1, 2]
    }
}

# ---------------------------------------------------------------
# Define Base Models
# ---------------------------------------------------------------
base_models = {
    "Logistic Regression": Pipeline([
        ('preprocess', preprocessor_std),
        ('clf', LogisticRegression(
            multi_class='multinomial',  # ✅ multinomial softmax
            solver='lbfgs',
            max_iter=2000,
            random_state=42
        ))
    ]),
    "Linear SVM": Pipeline([
        ('preprocess', preprocessor_std),
        ('clf', LinearSVC(max_iter=5000, random_state=42))
    ]),
    "Random Forest": Pipeline([
        ('preprocess', preprocessor_std),
        ('clf', RandomForestClassifier(random_state=42))
    ]),
    "KNN": Pipeline([
        ('preprocess', preprocessor_std),
        ('clf', KNeighborsClassifier())
    ]),
    "Gaussian Naive Bayes": Pipeline([
        ('preprocess', preprocessor_nb),
        ('clf', GaussianNB())
    ]),
    "Complement Naive Bayes": Pipeline([
        ('preprocess', preprocessor_nb),
        ('clf', ComplementNB())
    ]),
    "Histogram-based Classifier": Pipeline([
        ('preprocess', preprocessor_std),
        ('clf', HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.1,
            max_iter=600,
            class_weight='balanced',
            random_state=42
        ))
    ])
}

# ---------------------------------------------------------------
# Train + Hyperparameter Tuning with Cross-Validation
# ---------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
trained_models = {}
best_params = {}
best_scores = {}

for name, model in base_models.items():
    print(f"\n🔹 Training {name}...")
    if name in param_grids:
        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grids[name],
            scoring='accuracy',
            cv=cv,
            n_jobs=-1
        )
        grid.fit(X_train, y_train)
        trained_models[name] = grid.best_estimator_
        best_params[name] = grid.best_params_
        best_scores[name] = grid.best_score_
        print(f"✅ Best CV Accuracy for {name}: {grid.best_score_:.4f}")
        print(f"   Best Params: {grid.best_params_}")
    else:
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"✅ Trained {name} with default parameters.")

print("\n🏆 All Models Trained Successfully!")




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

    #if isinstance(model, ParzenWindowClassifier):
        #y_pred = model.predict(X_val.values)
    if "Polynomial Regression" in name:
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



# %%
# ===============================================================
# ⚡ Logistic Regression (Yeo-Johnson + StandardScaler + PolyFeatures)
# ===============================================================
from sklearn.preprocessing import PowerTransformer, StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np

# ---------------------------------------------------------------
# Define the pipeline
# ---------------------------------------------------------------
optimized_logreg = Pipeline([
    ("power", PowerTransformer(method='yeo-johnson')),  # handles skewed numeric features
    ("scaler", StandardScaler()),                       # standardize mean=0, std=1
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),  # add feature interactions
    ("clf", LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        C=0.02,  
        max_iter=500,                                      # regularization strength
        random_state=42           # more iterations for convergence
    ))
])

# ---------------------------------------------------------------
# Fit + Predict on validation data
# ---------------------------------------------------------------
optimized_logreg.fit(X_train, y_train)
y_pred = optimized_logreg.predict(X_val)

# ---------------------------------------------------------------
# Evaluate performance
# ---------------------------------------------------------------
print("📈 Accuracy:", accuracy_score(y_val, y_pred))
print("\nClassification Report:")
print(classification_report(y_val, y_pred, zero_division=0))

# ===============================================================
# 🏆 Kaggle Submission File (for Logistic Regression)
# ===============================================================
print("\n🚀 Generating Kaggle submission file for Logistic Regression...")

# Ensure your df_test is properly formatted and aligned
print("🔍 Test data shape before prediction:", df_test.shape)

# Generate predictions
test_predictions = optimized_logreg.predict(df_test)

# If your test set uses an 'id' index, use that for the submission
submission = pd.DataFrame({
    "id": df_test.index,
    "quality": test_predictions
})

# Quick sanity check
print("\n✅ Submission preview:")
print(submission.head())
print("Unique predicted values:", np.unique(submission["quality"]))

# Save the file
output_filename = "submission_logistic_regression.csv"
submission.to_csv(output_filename, index=False)
print(f"\n💾 File saved as: {output_filename}")






# %%
# ===============================================================
# 🤖 MLP Neural Network Regressor (Yeo-Johnson + StandardScaler)
# ===============================================================
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

# ---------------------------------------------------------------
# Define the pipeline
# ---------------------------------------------------------------
optimized_mlp = Pipeline([
    ("power", PowerTransformer(method='yeo-johnson')),   # Normalize skewed numeric features
    ("scaler", StandardScaler()),                        # Standardize mean=0, std=1
    ("mlp", MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),   # 3 hidden layers
        activation='relu',                  # ReLU activation
        solver='adam',                      # Adaptive optimizer
        alpha=0.001,                        # L2 regularization (ridge)
        learning_rate_init=0.001,           # Learning rate
        max_iter=800,                       # Enough epochs for convergence
        random_state=42,
        early_stopping=True,                # Stops when validation score stalls
        n_iter_no_change=20
    ))
])

# ---------------------------------------------------------------
# Fit + Predict on validation data
# ---------------------------------------------------------------
print("🔹 Training MLP Neural Network...")
optimized_mlp.fit(X_train, y_train)
y_pred = optimized_mlp.predict(X_val)

# Round and clip predictions to valid integer range (1–10)
y_pred_int = np.clip(np.round(y_pred), 1, 10).astype(int)

# ---------------------------------------------------------------
# Evaluate performance
# ---------------------------------------------------------------
rmse = np.sqrt(mean_squared_error(y_val, y_pred_int))
r2 = r2_score(y_val, y_pred_int)
print(f"\n📈 RMSE: {rmse:.4f}")
print(f"🔹 R² Score: {r2:.4f}")

# Compare to baseline (predicting mean quality)
baseline_rmse = np.sqrt(mean_squared_error(y_val, np.full_like(y_val, y_train.mean())))
print(f"Baseline RMSE (mean predictor): {baseline_rmse:.4f}")

# ---------------------------------------------------------------
# 🏆 Kaggle Submission File (Rounded Integer Predictions)
# ---------------------------------------------------------------
print("\n🚀 Generating Kaggle submission file for MLP Regressor...")

# Predict on test data
test_predictions = optimized_mlp.predict(df_test)

# Round and clip to integer quality (1–10)
test_predictions = np.clip(np.round(test_predictions), 1, 10).astype(int)

# Build submission DataFrame
submission = pd.DataFrame({
    "id": df_test.index,
    "quality": test_predictions
})

print("\n✅ Submission preview:")
print(submission.head(15))
print("Unique predicted values range:", submission["quality"].min(), "to", submission["quality"].max())

# Save CSV
output_filename = "submission_mlp_regressor_int.csv"
submission.to_csv(output_filename, index=False)
print(f"\n💾 File saved as: {output_filename}")



# %%
# ===============================================================
# 🌲 Random Forest Classifier (StandardScaler + Optimized Params)
# ===============================================================
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np

# ---------------------------------------------------------------
# Define the pipeline
# ---------------------------------------------------------------
optimized_rf = Pipeline([
    ("scaler", StandardScaler()),  # Scale numeric features
    ("clf", RandomForestClassifier(
        n_estimators=400,
        max_depth=20,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        class_weight='balanced_subsample'
    ))
])

# ---------------------------------------------------------------
# Fit + Predict on validation data
# ---------------------------------------------------------------
optimized_rf.fit(X_train, y_train)
y_pred = optimized_rf.predict(X_val)

# ---------------------------------------------------------------
# Evaluate performance
# ---------------------------------------------------------------
print("📈 Accuracy:", accuracy_score(y_val, y_pred))
print("\nClassification Report:")
print(classification_report(y_val, y_pred, zero_division=0))

# ===============================================================
# 🏆 Kaggle Submission File (for Random Forest)
# ===============================================================
print("\n🚀 Generating Kaggle submission file for Random Forest...")

# Ensure df_test has the same columns as training data
print("🔍 Test data shape before prediction:", df_test.shape)

# Generate predictions
test_predictions = optimized_rf.predict(df_test)

# If your test set uses an 'id' index, use that for submission
submission = pd.DataFrame({
    "id": df_test.index,
    "quality": test_predictions
})

# Quick sanity check
print("\n✅ Submission preview:")
print(submission.head())
print("Unique predicted values:", np.unique(submission["quality"]))

# Save the file
output_filename = "submission_random_forest.csv"
submission.to_csv(output_filename, index=False)
print(f"\n💾 File saved as: {output_filename}")


# %%
# ===============================================================
# ⚡ SVM Classifier (Yeo-Johnson + StandardScaler + L1/Lasso Penalty)
# ===============================================================
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np

# ---------------------------------------------------------------
# Define the pipeline
# ---------------------------------------------------------------
optimized_svm_l1 = Pipeline([
    ("power", PowerTransformer(method='yeo-johnson')),  # normalize skewed numeric features
    ("scaler", StandardScaler()),                       # scale features to mean=0, std=1
    ("clf", LinearSVC(
        penalty='l1',              # ✅ Lasso (L1) regularization
        loss='squared_hinge',
        dual=False,                # ⚠️ Required for L1 penalty
        C=0.01,                     # regularization strength (tune if needed)
        max_iter=1000,
        random_state=42
    ))
])

# ---------------------------------------------------------------
# Fit + Predict on validation data
# ---------------------------------------------------------------
optimized_svm_l1.fit(X_train, y_train)
y_pred = optimized_svm_l1.predict(X_val)

# ---------------------------------------------------------------
# Evaluate performance
# ---------------------------------------------------------------
print("📈 Accuracy:", accuracy_score(y_val, y_pred))
print("\nClassification Report:")
print(classification_report(y_val, y_pred, zero_division=0))

# ===============================================================
# 🏆 Kaggle Submission File (for SVM L1/Lasso)
# ===============================================================
print("\n🚀 Generating Kaggle submission file for SVM with L1 penalty...")

# Ensure test data alignment
print("🔍 Test data shape before prediction:", df_test.shape)

# Generate predictions
test_predictions = optimized_svm_l1.predict(df_test)

# Create submission DataFrame
submission = pd.DataFrame({
    "id": df_test.index,
    "quality": test_predictions
})

# Preview results
print("\n✅ Submission preview:")
print(submission.head())
print("Unique predicted values:", np.unique(submission["quality"]))

# Save the file
output_filename = "submission_svm_l1.csv"
submission.to_csv(output_filename, index=False)
print(f"\n💾 File saved as: {output_filename}")


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


