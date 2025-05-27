import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from .model import train_model, compute_model_metrics, inference

# starter/starter/ml/test_model.py

# Test that train_model returns a fitted RandomForestClassifier and can predict on training data
def test_train_model_returns_fitted_model():
    X = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    y = np.array([0, 1, 1, 0])
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)
    preds = model.predict(X)
    assert len(preds) == len(y)

# Test compute_model_metrics returns correct precision, recall, and fbeta for known inputs
def test_compute_model_metrics_known_values():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    # For these values: precision=1.0, recall=0.666..., fbeta=0.8
    assert np.isclose(precision, 1.0)
    assert np.isclose(recall, 2/3)
    assert np.isclose(fbeta, 0.8)

# Test that inference returns a numpy array of correct shape for predictions
def test_inference_output_shape():
    X = np.array([[0, 1], [1, 0]])
    y = np.array([0, 1])
    model = train_model(X, y)
    preds = inference(model, X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (2,)

