import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# Load dataset
df = pd.read_csv("../data/IRIS.csv")


# Define features and target
X = df[
    [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
    ]
]

y = df["species"]


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# Train Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)


# Train final model using the complete dataset
final_model = LogisticRegression(max_iter=200)
final_model.fit(X, y)


# Save model
joblib.dump(
    final_model,
    "../models/iris_logistic_regression.pkl",
)

print("Model trained and saved successfully.")