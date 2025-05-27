import pandas as pd
import pickle
from ml.data import process_data
from ml.model import compute_slice_metrics

# Load data and model artifacts
data = pd.read_csv("../data/census.csv")
model = pickle.load(open("../model/model.pkl", "rb"))  # Remove ../
encoder = pickle.load(open("../model/encoder.pkl", "rb"))
lb = pickle.load(open("../model/lb.pkl", "rb"))
print("Model and data loaded.")

# Define categorical features
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

# Compute and save slice metrics for each categorical feature
slice_output = {}
for feature in cat_features:
    slice_metrics = compute_slice_metrics(
        model,
        data,
        feature,
        categorical_features=cat_features,
        encoder=encoder,
        lb=lb
    )
    slice_output[feature] = slice_metrics
print("Slice performance analysis completed.")

# Save results to a file
with open('../model/slice_output.txt', 'w') as f:
    for feature, metrics in slice_output.items():
        f.write(f"\nPerformance on {feature} slices:\n")
        f.write("-" * 40 + "\n")
        for val, (precision, recall, fbeta) in metrics.items():
            f.write(f"{val}:\n")
            f.write(f"  Precision: {precision:.3f}\n")
            f.write(f"  Recall: {recall:.3f}\n")
            f.write(f"  F1: {fbeta:.3f}\n")
print("Saved results to slice_output.txt.")