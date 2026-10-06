# Iris Flower Classification

A machine learning project that classifies Iris flowers into three species using measurements of their sepals and petals.

## Project Overview

This project uses the classic Iris dataset to build and compare multiple classification models.

The models tested are:

* Decision Tree
* Logistic Regression
* Random Forest

The models are evaluated using a train/test split and 5-fold cross-validation.

## Dataset

The dataset contains 150 Iris flower observations across three species:

* Iris-setosa
* Iris-versicolor
* Iris-virginica

Four measurements are used as input features:

* Sepal length
* Sepal width
* Petal length
* Petal width

The dataset contains 50 observations for each species.

## Machine Learning Workflow

1. Load and inspect the dataset
2. Separate features and target variable
3. Split the data into training and testing sets
4. Train multiple classification models
5. Evaluate model accuracy
6. Analyse the confusion matrix and classification report
7. Perform 5-fold cross-validation
8. Select the best-performing model
9. Train the final model using the complete dataset
10. Save the trained model using Joblib
11. Load the saved model and make predictions

## Model Results

### Test Set Accuracy

| Model               | Accuracy |
| ------------------- | -------: |
| Decision Tree       |   93.33% |
| Logistic Regression |   96.67% |
| Random Forest       |   90.00% |

### 5-Fold Cross-Validation

| Model               | Mean CV Accuracy |
| ------------------- | ---------------: |
| Decision Tree       |           95.33% |
| Logistic Regression |       **97.33%** |
| Random Forest       |           96.67% |

Based on the 5-fold cross-validation results, **Logistic Regression** was selected as the final model.

## Final Model

The final Logistic Regression model was retrained using the complete dataset and saved as:

```text
models/iris_logistic_regression.pkl
```

The saved model can be loaded and used to predict the species of new Iris flowers.

## Project Structure

```text
iris-flower-classification/
│
├── data/
│   └── IRIS.csv
│
├── models/
│   └── iris_logistic_regression.pkl
│
├── notebooks/
│   └── iris_classification.ipynb
│
├── src/
│
├── requirements.txt
│
└── README.md
```

## Technologies Used

* Python
* Pandas
* Scikit-learn
* Joblib
* Jupyter Notebook

## Installation

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

Open the Jupyter Notebook:

```text
notebooks/iris_classification.ipynb
```

Run the notebook cells in order to reproduce the analysis, train the models, evaluate their performance, and generate predictions.

## Example Prediction

The trained model can predict an Iris species from four measurements:

```text
Sepal length: 5.1
Sepal width: 3.5
Petal length: 1.4
Petal width: 0.2
```

The model predicts:

```text
Iris-setosa
```

## Key Takeaway

Logistic Regression achieved the highest mean cross-validation accuracy of **97.33%** among the models tested on this dataset.

This project demonstrates a complete introductory machine learning classification workflow, from data preparation and model comparison through evaluation, model selection, saving, and prediction.
