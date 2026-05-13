# %% [markdown]
# **Importing relevant libraries and Pickel Data**

# %%
## Importing libraries and packages 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
import pickle

# %%
with open('data/train_data.pkl', 'rb') as f:
    data = pickle.load(f)

images_train = data['images']
labels_train = data['labels']

with open('data/test_data.pkl', 'rb') as g:
    data = pickle.load(g)

images_test = data['images']





# %% [markdown]
# **Exploring Pre-November 30th Data**

# %%
## Exploring shape of the data 
print("This is the shape of the images training data: ", images_train.shape)
print("This is the shape of the label training data", labels_train.shape)
print("This is the shape of the images test data: ", images_test.shape)

## Exploring data type 
print("This is the data type of the image training: ", images_train.dtype)
print("This is the number of unique value for labels training: ", np.unique(labels_train))
print("This is the data type of the image test: ", images_test.dtype)

# %%
unique, counts = np.unique(labels_train, return_counts=True)
print("Class distribution:")
for u, c in zip(unique, counts):
    print(f"Label {u}: {c} samples")

# %%
plt.imshow(images_train[0], cmap="gray")
plt.title(f"Label: {labels_train[0]}")
plt.show()

# %%
print(" The minimum pixel for training: ", images_train.min())
print(" The maximum pixel for training: ", images_train.max())
print(" The Mean pixel for training: ", images_train.mean())
print(" The standard deviation for training: ", images_train.std())

print(" The minimum pixel for test: ", images_test.min())
print(" The maximum pixel for test: ", images_test.max())
print(" The Mean pixel for test: ", images_test.mean())
print(" The standard deviation for test: ", images_test.std())

# %% [markdown]
# **Reshaping & Standardizing Data**

# %%
# Reshaping training and test data so we can perform Classification Methods 
images_train_rs = images_train.reshape(images_train.shape[0], -1)
y = labels_train.reshape(-1)
images_test_rs = images_test.reshape(images_test.shape[0], -1)

# Performing Normalization
X = images_train_rs/255.0
X_test = images_test_rs/255.0



# %% [markdown]
# **Train Test Split**

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_val_m, y_train, y_val_m = train_test_split(images_train_stand, labels_train_rs, test_size=0.2, random_state=42)

model_lr = LogisticRegression(max_iter=10000)
model_lr.fit(X_train, y_train)
model_lr.score(X_val_m, y_val_m)

# %%
def train_valid_split(X, y, ratio=0.2, seed=0):
    np.random.seed(seed)
    indx = np.random.permutation(len(X))
    split = int(len(X)*(1-ratio))
    return X[indx[:split]], X[indx[split:]], y[indx[:split]], y[indx[split:]]

X_tr, X_val, y_tr, y_val = train_valid_split(X, y)
print("\n These are the features for X_trained here: ", X_tr)
print("\n These are the labels for y_trained here: ", y_tr)

# %% [markdown]
# **Pre November 30th Model Testing (KNN & Log Reg)**

# %%
# ------------------------------------------
# L2 Normalization (CRITICAL for cosine KNN)
# ------------------------------------------
def l2_normalize(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    return X / norms

X_tr_norm = l2_normalize(X_tr)
X_val_norm = l2_normalize(X_val)
X_test_norm = l2_normalize(images_test_rs)


# ------------------------------------------
# Improved KNN: Cosine + Weighted Voting
# ------------------------------------------
def knn_predict(X_tr_norm, y_train, X_test_norm, k):
    preds = []
    K = len(np.unique(y_train))

    for x in X_test_norm:
        sims = X_tr_norm @ x                     # cosine similarity (vectors normalized)
        idx = np.argsort(-sims)[:k]            # top-k neighbors
        neighbor_labels = y_train[idx]
        weights = sims[idx] + 1e-8

        vote = np.zeros(K)
        for lbl, w in zip(neighbor_labels, weights):
            vote[lbl] += w

        preds.append(np.argmax(vote))

    return np.array(preds)


# ------------------------------------------
# Evaluation loop — find BEST k
# ------------------------------------------
k_values = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
results = {}

for k in k_values:
    y_pred_knn = knn_predict(X_tr_norm, y_tr, X_val_norm, k)
    acc = np.mean(y_pred_knn == y_val)
    results[k] = acc
    print(f"k={k}, accuracy={acc:.4f}")

# ------------------------------------------
# Pick the BEST performing k
# ------------------------------------------
best_k = max(results, key=results.get)
best_acc = results[best_k]

print("\n====================================")
print(f"Best k = {best_k}  (Accuracy = {best_acc:.4f})")
print("====================================\n")

# ------------------------------------------
# Retrain KNN on full training → predict test
# ------------------------------------------
# L2 normalize full datasets for COSINE KNN
X_norm = l2_normalize(X)
X_test_norm = l2_normalize(X_test)
y_pred_test_knn = knn_predict(X_norm, y, X_test_norm, k=best_k)

# ------------------------------------------
# Save Kaggle submission
# ------------------------------------------
df_sub = pd.DataFrame({
    "ID": np.arange(1, len(y_pred_test_knn) + 1),
    "Label": y_pred_test_knn
})

df_sub.to_csv("KNN_submission_FV.csv", index=False)

print(f"Kaggle submission saved using k={best_k} 🎉")


# %%
class LogisticRegression:
  
  def __init__(self, learning_rate=0.01, max_iter=1000):
    self.learning_rate = learning_rate
    self.max_iter = max_iter
    self.weights = None
    self.bias = None
  
  def _sigmoid(self, z):
    return 1 / (1 + np.exp(-z))
  
  def fit(self, X, y):
    
    y = y.reshape(-1)
    
    n_samples, n_features = X.shape
    
    self.weights = np.zeros(n_features)
    self.bias = 0
    
    for _ in range(self.max_iter):
      
      linear_model = np.dot(X, self.weights) + self.bias
      y_pred = self._sigmoid(linear_model)
      
      # Gradients (binary cross-entropy loss)
      error = y_pred - y
      dw = (1 / n_samples) * np.dot(X.T, error)
      db = (1 / n_samples) * np.sum(error)
      
      self.weights -= self.learning_rate * dw
      self.bias -= self.learning_rate * db
  
  def predict_proba(self, X):
    linear_model = np.dot(X, self.weights) + self.bias
    return self._sigmoid(linear_model)
  
  def predict(self, X, threshold=0.5):
    y_pred_proba = self.predict_proba(X)
    return (y_pred_proba >= threshold).astype(int)
  

# %%

class LogisticRegressionOVR:
  
  def __init__(self, learning_rate=0.01, max_iter=10000):
    self.learning_rate = learning_rate
    self.max_iter = max_iter
    self.classifiers = {}
    self.classes_ = None
    
  def fit(self, X, y):
    self.classes_ = np.unique(y)
    self.classifiers = {}
    
    for c in self.classes_:
      y_binary = (y == c).astype(int)
      clf = LogisticRegression(learning_rate=self.learning_rate,
                               max_iter=self.max_iter)
      clf.fit(X, y_binary)
      self.classifiers[c] = clf
  
  def predict_proba(self, X):
    probs = np.column_stack([
      self.classifiers[c].predict_proba(X) for c in self.classes_
    ])
    return probs
  
  def predict(self, X):
    probs = self.predict_proba(X)
    return self.classes_[np.argmax(probs, axis=1)]
  
  def score(self, X, y):
    y_pred = self.predict(X)
    return np.mean(y_pred == y)
  
# ============================================================
# 4. TRAIN THE MODEL
# ============================================================
model = LogisticRegressionOVR(learning_rate=0.01, max_iter=10000)
model.fit(X_tr, y_tr)

val_acc = model.score(X_val, y_val)
print("OVR Logistic Regression Validation Accuracy:", val_acc)

# ============================================================
# 5. PREDICT ON TEST SET (KAGGLE)
# ============================================================
y_pred_test = model.predict(X_test)

# Create submission file
df_sub = pd.DataFrame({
    "ID": np.arange(1, len(y_pred_test) + 1),
    "Label": y_pred_test
})

df_sub.to_csv("logreg_ovr_submission.csv", index=False)
print("Kaggle file saved as logreg_ovr_submission.csv")



# %%
## SKLearn Prediction with LR 
# ============================================================
# 4. TRAIN THE MODEL
# ============================================================
modelsk = LogisticRegressionOVR(learning_rate=0.01, max_iter=10000)
modelsk.fit(X_train, y_train)

val_acc_sk = model.score(X_val_m, y_val_m)
print("OVR Logistic Regression Validation Accuracy with SKLearn:", val_acc_sk)

# ============================================================
# 5. PREDICT ON TEST SET (KAGGLE)
# ============================================================
y_pred_test_sk = model.predict(X_test)

# Create submission file
df_sub = pd.DataFrame({
    "ID": np.arange(1, len(y_pred_test_sk) + 1),
    "Label": y_pred_test_sk
})

df_sub.to_csv("logreg_ovr_submission_sk.csv", index=False)
print("Kaggle file saved as logreg_ovr_submission_sk.csv")


