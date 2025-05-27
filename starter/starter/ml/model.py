from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from .data import process_data

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def compute_slice_metrics(model, data, feature, label="salary", categorical_features=None, encoder=None, lb=None):
    """
    Computes model metrics for slices of data based on a categorical feature.

    Inputs
    ------
    model : sklearn.ensemble.RandomForestClassifier
        Trained model
    data : pd.DataFrame
        Data to analyze
    feature : str
        Categorical feature to slice on
    label : str
        Name of label column
    categorical_features : list
        List of categorical feature names
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained encoder
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained label binarizer

    Returns
    -------
    slice_metrics : dict
        Dictionary with slice values as keys and (precision, recall, fbeta) as values
    """
    slice_metrics = {}

    for unique_val in data[feature].unique():
        slice_data = data[data[feature] == unique_val]

        X_slice, y_slice, _, _ = process_data(
            slice_data,
            categorical_features=categorical_features,
            label=label,
            training=False,
            encoder=encoder,
            lb=lb
        )

        preds = inference(model, X_slice)
        precision, recall, fbeta = compute_model_metrics(y_slice, preds)
        slice_metrics[unique_val] = (precision, recall, fbeta)

    return slice_metrics
