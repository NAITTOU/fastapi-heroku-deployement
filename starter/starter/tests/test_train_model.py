import pandas as pd
import pytest
import joblib
from sklearn.ensemble import RandomForestRegressor
from ml.data import process_data
from ml.model import train_model

DATA_PATH = "starter/data/census.csv"
MODEL_PATH = "starter/model/model.joblib"

label = "salary"
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


@pytest.fixture
def model():
    """Fixture to load model for testing."""
    return joblib.load(MODEL_PATH)


@pytest.fixture
def data():
    """Fixture to sample data for testing."""
    df = pd.read_csv(DATA_PATH)
    return df


def test_import_data(data):
    """Test that the data is loaded correctly."""
    assert data.shape[0] > 0, "Data should have rows"
    assert data.shape[1] > 0, "Data should have columns"


def test_data_shape(data):
    """If your data is assumed to have no null values
    then this is a valid test."""
    assert data.shape == data.dropna().shape, "Dropping null changes shape."


def test_process_data_training(data):
    """Test the process_data function in training mode."""

    X_processed, y_processed, encoder, lb = process_data(
        data, categorical_features=cat_features, label=label, training=True
    )

    assert (
        X_processed.shape[0] == data.shape[0]
    ), "Processed X should have the same number of rows as original X"
    assert (
        y_processed.shape[0] == data.shape[0]
    ), "Processed y should have the same number of rows as original y"
    assert encoder is not None, "Encoder should not be None"
    assert lb is not None, "Label binarizer should not be None"


def test_process_data_inference(data):
    """Test the process_data function in inference mode."""

    # Process the data in training mode first
    _, _, encoder, lb = process_data(
        data, categorical_features=cat_features, label=label, training=True
    )

    # Now process the same data in inference mode
    X_processed, y_processed, _, _ = process_data(
        data,
        categorical_features=cat_features,
        label=label,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    assert (
        X_processed.shape[0] == data.shape[0]
    ), "Processed X should have the same number of rows as original X"
    assert (
        y_processed.shape[0] == data.shape[0]
    ), "Processed y should have the same number of rows as original y"


def test_train_model(model):
    """Test the train_model function."""
    assert isinstance(
        model, RandomForestRegressor
    ), "Model should be an instance of RandomForestRegressor"
