import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression  # Keep this for reference
from sklearn.ensemble import RandomForestClassifier  # Use this model
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

# Read data from CSV
df = pd.read_csv("data_processed.csv")

# Prepare target variable
y = df.pop("cons_general").to_numpy()
y[y < 4] = 0
y[y >= 4] = 1

# Prepare features
X = df.to_numpy()
X = preprocessing.scale(X)  # Standardize features

# Impute missing values using mean strategy
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(X)
X = imp.transform(X)

# Create and train the Random Forest Classifier
clf = RandomForestClassifier()  # Use RandomForestClassifier
yhat = cross_val_predict(clf, X, y, cv=5)

# Calculate evaluation metrics
acc = np.mean(yhat == y)
tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)

# Save metrics to JSON file
with open("metrics.json", 'w') as outfile:
    json.dump({
        "accuracy": acc,
        "specificity": specificity,
        "sensitivity": sensitivity}, outfile)

# Prepare data for visualization
score = yhat == y
score_int = [int(s) for s in score]
df['pred_accuracy'] = score_int

# Create bar plot by region
sns.set_color_codes("dark")
ax = sns.barplot(x="region", y="pred_accuracy", data=df, palette="Greens_d")
ax.set(xlabel="Region", ylabel="Model Accuracy")
plt.savefig("by_region.png", dpi=80)

