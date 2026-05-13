# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# %%
with open('data/train_data.pkl', 'rb') as f:
    data = pickle.load(f)

X_train = data['images']
y_train = data['labels']

# %%
# X_train = pd.DataFrame(X_train.reshape(X_train.shape[0], -1))
# X_train.head()

# %%
X_train.shape

# %%
X_train.dtype

# %%
y_train.shape

# %%
print(y_train[0:50])

# %%
print(y_train.min(), y_train.max())

# %%
X_train = X_train.reshape(X_train.shape[0], -1)
y_train = y_train.reshape(-1)

# %%
X_train.max(), X_train.min()

# %%
X_train = X_train / 255.0

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

model_lr = LogisticRegression(max_iter=10000)
model_lr.fit(X_train, y_train)

# %%
model_lr.score(X_val, y_val)

# %%
import pickle
import numpy as np
import pandas as pd

# load test data
with open('data/test_data.pkl', 'rb') as f:
    data = pickle.load(f)

X_test = data['images']

X_test = X_test.reshape(X_test.shape[0], -1)
X_test = X_test / 255.0

y_pred = model_lr.predict(X_test)
y_pred = y_pred.astype(int)


predictions_df = pd.DataFrame({
    "ID": np.arange(1, len(X_test)+1),
    "Label": y_pred
})

predictions_df.to_csv(
    f"predicted_quality_{model_lr.__class__.__name__}.csv",
    index=False
)

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
model.fit(X_train, y_train)

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

df_sub.to_csv("logreg_ovr_submission_main.csv", index=False)
print("Kaggle file saved as logreg_ovr_submission.csv")



