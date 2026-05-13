# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
file_train = "./data/train.csv"
file_test = "./data/test.csv"

df_train = pd.read_csv(file_train)
df_test = pd.read_csv(file_test)
print(df_train)

# %%
df_train.columns[0]

# %%
df_train[['id', 'beer_style', 'bitterness_IBU', 'diacetyl_concentration', 'lactic_acid','final_gravity', 'sodium', 'free_CO2', 'dissolved_oxygen', 'original_gravity', 'pH', 'gypsum_level', 'alcohol_ABV', 'fermentation_strength', 'quality']] = df_train.iloc[:, 0].str.split(';', expand=True)

# %%
df_test[['id', 'beer_style', 'bitterness_IBU', 'diacetyl_concentration', 'lactic_acid','final_gravity', 'sodium', 'free_CO2', 'dissolved_oxygen', 'original_gravity', 'pH', 'gypsum_level', 'alcohol_ABV', 'fermentation_strength']] = df_test.iloc[:, 0].str.split(';', expand=True)

# %%
df_train = df_train.drop(columns=[df_train.columns[0]]).set_index('id')
print(df_train)

# %%
df_test = df_test.drop(columns=[df_test.columns[0]]).set_index('id')
print(df_test)

# %%
df_train.columns
df_train.to_csv('train.csv')

# %%
df_test.columns

# %%
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

clf_names = [
    "Nearest Neighbors",
    "Nearest Neighbors 5",
    "Nearest Neighbors 7",
    "Linear SVM",
    "Polynomial SVM",
    "RBF SVM",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    #"QDA",
]

classifiers = [
    KNeighborsClassifier(3),
    KNeighborsClassifier(5),
    KNeighborsClassifier(7),
    SVC(kernel="linear", C=0.025, random_state=42),
    SVC(kernel="poly", C=0.025, random_state=42),
    SVC(gamma=2, C=1, random_state=42),
    GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
    DecisionTreeClassifier(max_depth=5, random_state=42),
    RandomForestClassifier(
        max_depth=5, n_estimators=10, max_features=1, random_state=42
    ),
    MLPClassifier(alpha=1, max_iter=1000, random_state=42),
    AdaBoostClassifier(random_state=42),
    GaussianNB(),
    #QuadraticDiscriminantAnalysis(),
]

y_train = df_train['quality']
df_train = df_train.drop(columns=[df_train.columns[-1]])

le = LabelEncoder()
df_train['beer_style'] = le.fit_transform(df_train['beer_style'])
df_test['beer_style'] = le.fit_transform(df_test['beer_style'])

rng = np.random.RandomState(2)

# %%
X_train, X_test, y_train, y_test = train_test_split(
    df_train, y_train, test_size=0.3, random_state=42
)

trained_models = {}
results = []

for name, clf in zip(clf_names, classifiers):
    print(f"Training {name}...")
    try:
        model = make_pipeline(StandardScaler(), clf)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        trained_models[name] = model
        results.append({"Classifier": name, "Accuracy": score})
    except Exception as e:
        print(f"⚠️ Skipped {name} — error: {e}")
        continue

results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
print("\nFinal results:")
print(results_df)

# %%
import pandas as pd


lin_svm = trained_models["Gaussian Process"]
feature_cols = X_train.columns
df_pred = df_test[feature_cols]

y_pred = lin_svm.predict(df_pred)
predictions_df = pd.DataFrame({
    "id": df_pred.index,
    "quality": y_pred
})
predictions_df.to_csv("predicted_quality_gaussian_process.csv", index=False)
print("Saved → predicted_quality_knn5.csv")

# %%



