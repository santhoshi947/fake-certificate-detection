import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("C:\\Users\\HP\OneDrive\Desktop\certificate_detector\certificate_dataset.csv")

cols_to_drop = ["filename", "folder"]
df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
df = df.dropna(axis=1, how='all')

X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Test Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

cv_scores = cross_val_score(model, X, y, cv=5)

print("\nCross Validation Scores:", cv_scores)
print("Mean CV Accuracy:", np.mean(cv_scores))

feature_importance = pd.Series(model.feature_importances_, index=X.columns)
feature_importance = feature_importance.sort_values(ascending=False)

print("\nTop 10 Important Features:\n")
print(feature_importance.head(10))

plt.figure()
feature_importance.head(10).plot(kind='bar')
plt.tight_layout()
plt.show()

joblib.dump(model, "certificate_detection_model.pkl")
print("\nModel Saved Successfully!")