# Script to train machine learning model.


from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics

import pandas as pd
import logging
from joblib import dump

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


DATA_PATH = "starter/data/census.csv"
OUTPUT_PATH = "starter/model"

# Add code to load in the data.
logger.info("Loading Data ...")
data = pd.read_csv(DATA_PATH)

# Optional enhancement, use K-fold cross
# validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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

# Proces the test data with the process_data function.

X_test, y_test, encoder, lb = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# Train and save a model.
logger.info("Training the model ...")
model = train_model(X_train, y_train)

logger.info("Running inference on test data")
test_pred = inference(model, X_test)

logger.info("Scoring the predections")
test_pred_binary = (test_pred >= 0.5).astype(
    int
)  # Convert probabilities to binary labels

precision, recall, fbeta = compute_model_metrics(y_test, test_pred_binary)
logger.info(
    f"Scoring results: "
    f"precision: {precision}, "
    f"recall: {recall}, "
    f"fbeta: {fbeta}"
)

logger.info(f"Saving models to disk in {OUTPUT_PATH} folder")
dump(model, f"{OUTPUT_PATH}/model.joblib")
dump(encoder, f"{OUTPUT_PATH}/encoder.joblib")
dump(lb, f"{OUTPUT_PATH}/lb.joblib")
