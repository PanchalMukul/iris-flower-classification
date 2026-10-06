import pandas as pd
import joblib


# Load trained model
model = joblib.load("../models/iris_logistic_regression.pkl")


# Create sample flower
sample = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=[
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
    ],
)


# Make prediction
prediction = model.predict(sample)

print("Predicted species:", prediction[0])