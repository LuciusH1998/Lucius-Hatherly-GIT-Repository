# %% [markdown]
# # Instructions for Running Code
# Run all code cells in the order that you see them here, and markdown cells contain section headers, analysis, and other important observations.

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
# Load the training data and set ; as the delimiter 
train_df = pd.read_csv('data/train.csv', delimiter=';', index_col="id")

# Training Data 
print(f"Training data shape: {train_df.shape}")
print(f"Number of samples (n): {train_df.shape[0]}")
print(f"Number of features (d): {train_df.shape[1] - 1}")  # Subtract 1 for target column
print(f"Columns: {list(train_df.columns)}")
print("\nFirst few rows:")
print(train_df.head())


# %%
# Check data types and basic information 
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

# %% [markdown]
# **Analysis**
# 
# The bar chart indicates that quality is highly centered around scores 5, 6, and to a lesser extent 7. Scores 1 and 2 are not present in the data. This indicates severe class imbalance which may prove problematic for accuracy in the evaluated model. 
# 
# The correlation matrix indicates that alcohol_ABV and fermentation_strength are highly correlated together and the feature pH is poorly correlated with the target variable quality. This could give us some good insight into how I could perform feature engineering later on. 

# %%
## Detecting Outliers for feature columns using a box plot 
plt.figure(figsize=(12,4))
## Applying log transform to features to make the scale smaller and easier to read 
sns.boxplot(data=np.log1p(train_df[features.drop(["quality"], errors='ignore')]))
plt.xticks(rotation=90)
plt.title("Numeric Features Boxplot")
plt.show()

# %% [markdown]
# **Analysis** 
# 
# Boxplot shows outliers for certain numeric features that we can remove later through feature engineering. Outliers can skew accuracy of data. 

# %%
## Analyzing Categorial feature with bar chart - Beer Style column 
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
# **Analysis**
# 
# Beer style is predominantly dominated by Pale and Golden as the most popular styles within the data set. However, all styles have qualities largely ranging from 4 to 7 with 5 and 6 being the most common quality scores. This reiterates that 5 and 6 are the most popular quality scores in the data set and should be remembered in model evaluation due to the effects of class imbalances. 

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

# Fill numeric columns if missing values are present 
# Identify numeric columns (exclude 'quality' if present)
numeric_cols = [col for col in df_train.select_dtypes(include=["float64", "int64"]).columns if col != "quality"]

# Filling column in na/missing values with median ones 
for col in numeric_cols:
    df_train[col] = df_train[col].fillna(df_train[col].median())
    df_test[col] = df_test[col].fillna(df_train[col].median())

# Fill categorical column with mode value
df_train["beer_style"] = df_train["beer_style"].fillna(df_train["beer_style"].mode()[0])
df_test["beer_style"] = df_test["beer_style"].fillna(df_train["beer_style"].mode()[0])

# Print results
print("\n Missing values after imputations performed for Training data:")
print(df_train.isnull().sum().sum())

print("\n Missing values after imputations performed for Testing data:")
print(df_test.isnull().sum().sum())











# %%
## Removing outlying data based on summary statistics results and boxplot distributions
# Outlier clipping per feature distribution
for col in numeric_cols:
    # Within free_CO2, dissolved_oxygen, and final_gravity column calculating the 5th and 95th percentile 
    # Given these columns are less correlated with quality (target variable)
    # I decided to remove a greater range of outliers as impact to model performance will likely be minimal
    if col in ['free_CO2', 'dissolved_oxygen', 'final_gravity']:
        lower, upper = np.percentile(df_train[col], [5, 95])
        # Calculating the 1st and 99th percentile columns for alcohol_ABV and fermentation_strength
        # Given these columns are more correlated with quality, I decided to be more conservative in the range of outliers removed
        # Removing too many points could negatively impact model performance 
    elif col in ['alcohol_ABV', 'fermentation_strength']:
        lower, upper = np.percentile(df_train[col], [1, 99])
    else:
        # For all other columns, calculate the 2nd and 98th percentile column
        lower, upper = np.percentile(df_train[col], [2, 98])
    # Removing these outlier columns with clip 
    df_train[col] = np.clip(df_train[col], lower, upper)


# %% [markdown]
# **Analysis**
# 
# Filling in missing values for both numeric and categorical columns with median and mode respectively. Even though analysis reveals that the data set contains no missing values, the code is now set up to process any future data set with missing values. 
# 
# Moreover, we remove outliers from numerical columns to prevent any distorted impact on model evaluations that could hinder performance. Based on the earlier boxplot of numerical features, we remove all points below the 5th and above the 95th percentile for free_CO2, dissolved_oxygen, and final_gravity. as well as remove all points below the 1st and above the 99th percentile for alcohol_ABV and fermentation_strength. All other columns have outliers below the 2nd and above the 98th percentile removed. 

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
# Feature Engineering (adding and removing unnecessary columns)
import numpy as np
import pandas as pd

# Copying df_train and df_test before feature engineering is applied to use in a downstream experiment 
df_train_v2 = df_train.copy()
df_test_v2 = df_test.copy() 

df_train['alcohol_efficiency'] = df_train['alcohol_ABV'] / (df_train['fermentation_strength'] + 1e-6)
df_test['alcohol_efficiency'] = df_test['alcohol_ABV'] / (df_test['fermentation_strength'] + 1e-6)

# Dropping Fermentation strength as it is highly correlated with alcohol_ABV and can be removed
# Also dropping pH as the column is weakly correlated with quality
df_train = df_train.drop(columns=['pH', 'fermentation_strength'])
df_test = df_test.drop(columns=['pH', 'fermentation_strength'])

print("Added engineered features: alcohol efficiency and removed pH column.")


# %%
## Final Analysis of Results after preprocessing 
print("\nTraining data shape")
print(df_train.shape)
print("\nTraining data types")
print(df_train.dtypes)
print("\nTest data shape")
print(df_test.shape)

# %% [markdown]
# **Analysis**
# 
# Results of correlation matrix show that the alcohol_ABV and fermentation_strength are the most correlated (~0.43 and 0.41 respectively) of the feature columns with quality. Furthermore, alcohol_ABV and fermentation_strength are heavily correlated with one another (~0.95). This indicates that the columns encode similar information. Therefore, I removed fermentation_strength as redundant and added in a column, alcohol_efficiency, which is the quotient of alcohol_ABV and fermentation_strength. The column pH is weakly correlated with quality (~0.01) so it encodes largely irrelevant info towards predicting the target variable so can be safely removed. 

# %% [markdown]
# ## Prepare Training and Validation Data
# 
# Split your data into training and validation sets.

# %%
## Splitting Data into Train and Test Validation Sets: 
from sklearn.model_selection import train_test_split

# Defining training and validation data
# Importing Label Encoder 
le = LabelEncoder()
# Applying label encoding fit transform on beer style for training 
df_train['beer_style'] = le.fit_transform(df_train['beer_style'])
# Applying label encoding transform to beer style 
df_test['beer_style'] = le.transform(df_test['beer_style'])

rng = np.random.RandomState(2)

# Removing quality column from X as quality is the label
X = df_train.drop(columns="quality")
# Assigning quality to y 
y= df_train["quality"]

# Using train test split to create training and validation sets 
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

##Printing Training and Training and Validation Set Sizes: 
print(f"Training set sizes for X_train (features) are: {X_train.shape[0]}, Training set sizes for y_train (labels): {y_train.shape}")
print(f"Validation set sizes for X_val (features) are:   {X_val.shape[0]}, Training set sizes for y_val (labels):   {y_val.shape}")


# %%
# Printing value counts of quality and beer style 
print(y_train.value_counts(normalize=True).sort_index())
print(X_train["beer_style"].value_counts())


# %% [markdown]
# ## Train Models
# 
# Train and compare multiple machine learning models.

# %%
# Importing all necessary libraries and functions for Model Training 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, PolynomialFeatures
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import numpy as np

# Define Preprocessors
preprocessor_std = StandardScaler()
preprocessor_nb = MinMaxScaler()

# Define Parameter Grids for different models 
param_grids = {
    # Defining Logistic Regression Hyperparameters
    # Including C, penalty, multi_class, and solver 
    "Logistic Regression": {
        "clf__C": [0.01, 0.02, 0.1],
        "clf__penalty": ["l2"],
        "clf__multi_class": ["multinomial"],  
        "clf__solver": ["lbfgs"],
    },
    # Defining Linear SVM Hyperparameters 
    # Including penalty, C, and loss
    "Linear SVM": {
        "clf__penalty": ["l2"], 
        "clf__C": [0.01, 0.02, 0.1],
        "clf__loss": ["squared_hinge"],
    },
    # Defining Random Forest Hyperparameters 
    # Including n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, bootstrap, and class_weight
    "Random Forest": {
    "clf__n_estimators": [180, 200],     
    "clf__max_depth": [15, 20],          
    "clf__min_samples_split": [3, 4],      
    "clf__min_samples_leaf": [1, 2],       
    "clf__max_features": ["sqrt"],            
    "clf__bootstrap": [True],
    "clf__class_weight": ["balanced_subsample"]
},
    # Defining KNN hyperparameters 
    # This includes n_neighbors, weights, and p
    "KNN": {
        "clf__n_neighbors": [5, 10, 15, 25],
        "clf__weights": ["uniform", "distance"],
        "clf__p": [1, 2]
    }
}

# Defining the base models for each model algorithm where we have a pipeline, standardization, polynomial features, and other components
# for each model
base_models = {
    # Logistic Regression Classifier
    "Logistic Regression": Pipeline([("power", PowerTransformer(method='yeo-johnson')),
        ('preprocess', preprocessor_std),
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('clf', LogisticRegression(
            multi_class='multinomial',  # multinomial softmax
            solver='lbfgs',
            max_iter=2000,
            random_state=42
        ))
    ]),
    # Defining Linear SVM Pipeline
    "Linear SVM": Pipeline([
        ('preprocess', preprocessor_std),
        ('clf', LinearSVC(max_iter=5000, random_state=42))
    ]),
    # Defining Random Forest Pipeline
    "Random Forest": Pipeline([
        ('preprocess', preprocessor_std),
        ('clf', RandomForestClassifier(random_state=42))
    ]),
    # Defining KNN Pipeline
    "KNN": Pipeline([
        ('preprocess', preprocessor_std),
        ('clf', KNeighborsClassifier())
    ]),
    # Defining Gaussian Naive Bayes 
    "Gaussian Naive Bayes": Pipeline([
        ('preprocess', preprocessor_nb),
        ('clf', GaussianNB())
    ]),
    # Defining Complement Naive Bayes
    "Complement Naive Bayes": Pipeline([
        ('preprocess', preprocessor_nb),
        ('clf', ComplementNB())
    ])
}

# Training and Hyperparameter Tuning with Cross-Validation
# Defining Cross Validation with K Fold 5
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Defining trained models 
trained_models = {}
# Defining best parameters and scores  
best_params = {}
best_scores = {}

# Iterating over base models 
for name, model in base_models.items():
    # Print statement for Training model 
    print(f"\n Training {name}...")
    # Iterating over the parameter grids 
    if name in param_grids:
        # Creating Grid Search Cross Validation for accuracy 
        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grids[name],
            scoring='accuracy',
            cv=cv,
            n_jobs=-1
        )
        # Fitting training data on grid 
        grid.fit(X_train, y_train)
        # Getting the best estimator 
        trained_models[name] = grid.best_estimator_
        # Getting the best parameter name and scores 
        best_params[name] = grid.best_params_
        best_scores[name] = grid.best_score_
        # Printing the best cv accuracy and best parameters amongst the models 
        print(f" Best CV Accuracy for {name}: {grid.best_score_:.4f}")
        print(f"   Best Params: {grid.best_params_}")
    # Otherwise fitting model on X_train and y_train data 
    else:
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f" Trained {name} with default parameters.")
# Models are now all trained successfully 
print("\n All Models are now trained Successfully!")




# %% [markdown]
# ## Evaluate Models
# 
# Analyze your best model's performance (see README.md for evaluation guidelines).

# %%
# Importing all relevant libraries for evaluating models 
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Defining array of results 
results_train = []
results_val = []

# Iterating over trained_models 
for name, model in trained_models.items():
    # Printing evaluating statement 
    print(f"\n Evaluating {name}...")

    # Predictions on the training data 
    y_pred_train = model.predict(X_train)
    # Obtaining accuracy score of predictions on training data 
    acc_train = accuracy_score(y_train, y_pred_train)
    # Printing performance and accuracy for models on training set
    results_train.append((name, acc_train))
    print(f"\n Performance on Training set for {name}:")
    print(f"Training Accuracy: {acc_train:.4f}")
    print(classification_report(y_train, y_pred_train, zero_division=0))


    # Printing Confusion Matrix for Predicted Models on Training Data
    labels = np.unique(np.concatenate([y_train, y_pred_train]))
    ConfusionMatrixDisplay(confusion_matrix(y_train, y_pred_train, labels=labels),
                           display_labels=labels).plot(cmap="Grays")
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

    # Predictions on the Validation Data
    # If Polynomial Regregression in model name, round the predictions to the nearest integer
    if "Polynomial Regression" in name:
        y_pred = np.round(model.predict(X_val)).astype(int)
        y_pred = np.clip(y_pred, y_val.min(), y_val.max()) 
    else:
        y_pred = model.predict(X_val)

    # Defining accuracy score for model's on the validation sets 
    acc = accuracy_score(y_val, y_pred)
    results_val.append((name, acc))
    print(f"\n Validation set performance for {name}:")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_val, y_pred, zero_division=0))

    # Printing confusion matrix for model's predictions on the validation set 
    labels = np.unique(np.concatenate([y_val, y_pred]))
    ConfusionMatrixDisplay(confusion_matrix(y_val, y_pred, labels=labels),
                           display_labels=labels).plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

# Summarizing results for validation and training set 
results_train_df = pd.DataFrame(results_train, columns=["Model", "Accuracy"]).sort_values(by="Accuracy", ascending=False)
results_val_df = pd.DataFrame(results_val, columns=["Model", "Accuracy"]).sort_values(by="Accuracy", ascending=False)
print("\n Training Set Model Performance Summary:")
print(results_train_df)
print("\n Validation Set Model Performance Summary:")
print(results_val_df)



# %%
# Investigating Errors on Each Model 
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
import pandas as pd 
import numpy as np 

# Defining Error Array 
error_table = []

# Iterating over name and models in trained_models 
for name, model in trained_models.items():
    print(f"\n Calculating errors for this model {name}...")

    # Errors on Training Set
    y_pred_train = model.predict(X_train)
    # First provide accuracy for training 
    acc_train = accuracy_score(y_train, y_pred_train)
    # Calculating training error
    train_error = 1-acc_train 
    # Calculating mean absolute error 
    train_mae = mean_absolute_error(y_train, y_pred_train)
    # Calculating rmse 
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))

    # Errors on the Validation Set
    # If model uses polynomial regression in name, round on predictions for quality to complete integers 
    if "Polynomial Regression" in name:
        y_pred_val = np.round(model.predict(X_val)).astype(int)
        y_pred_val = np.clip(y_pred_val, y_val.min(), y_val.max())
    else:
        y_pred_val = model.predict(X_val)
    
    # Computing accuracy, error, MAE, and rmse on validation set 
    acc_val = accuracy_score(y_val, y_pred_val)
    err_val = 1 - acc_val
    mae_val = mean_absolute_error(y_val, y_pred_val)
    rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
    
    # Appending errors for training and validation set to error table 
    error_table.append({
        "Model": name,
        "Train Accuracy": acc_train,
        "Train Error": train_error,
        "Train MAE": train_mae,
        "Train RMSE": train_rmse,
        "Accuracy Validation": acc_val,
        "Error Validation": err_val,
        "MAE Validation": mae_val,
        "RMSE Validation": rmse_val
    })
# Printing error table 
error_table_df = pd.DataFrame(error_table).sort_values("Accuracy Validation", ascending=False)
print("\n Summary of Errors across Model")
print(error_table_df)
    



# %%
## Investigating which features are important for our best model
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 

# Defining the best model which is a random forest 
optimal_rf = trained_models["Random Forest"]["clf"]

# Defining significance of best features 
sig_features = optimal_rf.feature_importances_
# Setting up data frame for significant features 
sig_features_df = pd.DataFrame({"Feature": X_train.columns, "Significance": sig_features}).sort_values(by="Significance", ascending=False)

# Printing sig_features_df head
print(sig_features_df.head(14))

# %% [markdown]
# ## Generate Predictions for Kaggle Submission
# 
# Create a CSV file with columns: `id` and `quality` (see README.md for format details).

# %% [markdown]
# **Generating Kaggle File and Further hyperparameter optimization for Random Forest**

# %%
# Random Forest Classifier (GridSearch + Kaggle Submission)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np

# Define pipeline
rf_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(random_state=42, n_jobs=-1))
])

# Define parameter grid (narrow around the best values)
# Further optimization of Random Forest model
param_grid = {
    "clf__n_estimators": [200, 250, 400],
    "clf__max_depth": [20, 25, 30],
    "clf__min_samples_split": [4, 5, 6],
    "clf__min_samples_leaf": [2, 3, 4],
    "clf__max_features": ["sqrt"],
    "clf__bootstrap": [True],
    "clf__class_weight": ["balanced_subsample"]
}

# Run Grid Search with cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define scoring as accuracy 
grid_search = GridSearchCV(
    estimator=rf_pipe,
    param_grid=param_grid,
    scoring="accuracy",
    cv=cv,
    n_jobs=-1,
    verbose=2
)

print("Starting hyperparameter optimization for Random Forest Model...")
grid_search.fit(X_train, y_train)

print("\n These are the Best parameters found:")
print(grid_search.best_params_)
print(f" Best CV Accuracy: {grid_search.best_score_:.4f}")

# Evaluate best model on validation data
best_rf = grid_search.best_estimator_

y_pred_val = best_rf.predict(X_val)
val_acc = accuracy_score(y_val, y_pred_val)

print(f"\n Validation Accuracy (Best RF): {val_acc:.4f}")
print("\nClassification Report (Validation Set):")
print(classification_report(y_val, y_pred_val, zero_division=0))

# Optional: Training accuracy (to check overfitting)
y_pred_train = best_rf.predict(X_train)
train_acc = accuracy_score(y_train, y_pred_train)
print(f"\n Training Accuracy: {train_acc:.4f}")

# Kaggle Submission (Predictions on Test Data)
print("\n Generating Kaggle submission file for Best Random Forest...")
test_predictions = best_rf.predict(df_test)

# Build submission DataFrame
submission = pd.DataFrame({
    "id": df_test.index,
    "quality": test_predictions
})

# Save to CSV
output_filename = "submission_best_random_forest.csv"
submission.to_csv(output_filename, index=False)

print("\n Submission file created successfully!")
print(submission.head())
print(f" File saved as: {output_filename}")


# %% [markdown]
# **Evaluating Random Forest Model with Feature Engineering performed post train-test split**

# %%
# Performing feature engineering post train test split to test accuracy on Random Forest Model
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

## Splitting Data into Train and Test Validation Sets: 
from sklearn.model_selection import train_test_split
# Defining training and validation data

# Importing Label Encoder 
le = LabelEncoder()
# Applying label encoding fit transform on beer style for training 
df_train_v2['beer_style'] = le.fit_transform(df_train['beer_style'])
# Applying label encoding transform to beer style 
df_test_v2['beer_style'] = le.transform(df_test['beer_style'])

rng = np.random.RandomState(2)

# Removing quality column from X as quality is the label
X_v2 = df_train_v2.drop(columns="quality")
# Assigning quality to y 
y_v2= df_train_v2["quality"]

# Using train test split to create training and validation sets 
X_train_v2, X_val_v2, y_train_v2, y_val_v2 = train_test_split(X_v2, y_v2, test_size=0.2, random_state=42, stratify=y)



# Defining alcohol efficiency as the quotient of alcohol_ABV against fermentation_strength as these two features 
# are the most correlated columns with quality 
X_train_v2["alcohol_eff"] = df_train_v2["alcohol_ABV"]/(df_train_v2["fermentation_strength"] + 1e6)
X_val_v2["alcohol_eff"] = df_test_v2["alcohol_ABV"]/(df_test_v2["fermentation_strength"] + 1e6)
df_test_fe = df_test_v2.copy()
df_test_fe["alcohol_eff"] = df_test_fe["alcohol_ABV"]/(df_test_v2["fermentation_strength"] + 1e6)

# Dropping redundant columns 
# pH has a correlation of around 0.01 with quality and fermentation_strength is highly correlated with alcohol_ABV
# fermentation_strength is 0.41 wrp to quality, while alcohol_ABV is 0.43 wrp to quality
# Thus, fermentation_strength can be removed 
X_train_v2 = X_train_v2.drop(columns=["pH", "fermentation_strength"], errors="ignore")
X_val_v2 = X_val_v2.drop(columns=["pH", "fermentation_strength"], errors="ignore")
df_test_fe = df_test_fe.drop(columns=["pH", "fermentation_strength"], errors="ignore")

# Defining random forest model
opt_rf = Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(
    n_estimators=400, 
    max_depth=20, 
    min_samples_split=4, 
    min_samples_leaf=2, 
    max_features='sqrt', 
    bootstrap=True, 
    random_state=42, 
    n_jobs=-1, 
    class_weight='balanced_subsample')) ])

# Training and evaluating the the Random Forest Model
print("Evaluating the Random Forest Model with feature engineering post train test split: ")
opt_rf.fit(X_train_v2, y_train_v2)
y_pred_v2 = opt_rf.predict(X_val_v2)

# Printing accuracy of model
acc = accuracy_score(y_val_v2, y_pred_v2)
print("Accuracy is: ", acc)
print("\nClassification Report:")
print(classification_report(y_val_v2, y_pred_v2, zero_division=0))

# Generating Kaggle file for the feature engineering performed post train_test split 
print("\n Generating a Kaggle submission file for additional engineering features (rf_additional_feat_eng.csv)...")
predictions_file = opt_rf.predict(df_test_fe)
final_submission = pd.DataFrame({
    "id": df_test.index,         # assumes df_test index matches Kaggle’s "id"
    "quality": predictions_file  # predicted quality values
})

# Final file for Kaggle submission 
output_filename = "rf_additional_feat_eng.csv"
final_submission.to_csv(output_filename, index=False)
print("\n Submission Preview: ")
print(final_submission.head())




# %% [markdown]
# ---
# Good luck with the competition! 🍺
# 


