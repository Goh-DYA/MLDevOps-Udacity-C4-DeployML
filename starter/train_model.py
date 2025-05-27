# Script to train machine learning model.

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model

# Load the data
data = pd.read_csv("../data/census.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=42)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)
print("Training data processed.")

# Process the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)
print("Test data processed.")

# Train and save a model.
model = train_model(X_train, y_train)
print("Model trained.")

# Save model, encoder, and label binarizer as pickle files
pickle.dump(model, open("../model/model.pkl", "wb"))
pickle.dump(encoder, open("../model/encoder.pkl", "wb"))
pickle.dump(lb, open("../model/lb.pkl", "wb"))
