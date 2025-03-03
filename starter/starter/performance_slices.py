"""
This module outputs the performance of the model
on slices of the data for categorical features.
"""

from pathlib import Path
import joblib
import logging
from typing import List, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from ml.data import process_data
from ml.model import compute_model_metrics

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Configuration
CONFIG = {
    "data_path": Path("starter/data/census.csv"),
    "models_path": Path("starter/model"),
    "output_file": Path("starter/slice_output.txt"),
    "test_size": 0.20,
    "categorical_features": [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ],
}


class ModelPerformanceTester:
    """Class to test model performance on categorical feature slices."""

    def __init__(self, config: dict):
        self.config = config
        self.data = self._load_data()
        self.model, self.encoder, self.lb = self._load_models()

    def _load_data(self) -> pd.DataFrame:
        """Load the dataset."""
        logger.info(f"Loading data from {self.config['data_path']}")
        return pd.read_csv(self.config["data_path"])

    def _load_model(self, model_path):
        """Load model from the specified file path.

        Args:
            model_path (str or Path): The path to the model file to be loaded.

        Returns:
            object: The loaded model, which could be a model, data, etc.

        Raises:
            FileNotFoundError: If the specified model path does not exist.
            ValueError: If the model cannot be loaded due to an invalid format.
        """
        # Convert model_path to a Path object
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"model not found at path: {model_path}")

        try:
            return joblib.load(model_path)
        except Exception as e:
            raise ValueError(f"Could not load model from {model_path}:"
                             f"{str(e)}")

    def _load_models(self) -> Tuple[BaseEstimator,
                                    OneHotEncoder,
                                    LabelBinarizer]:
        """Load model models."""
        models_path = self.config["models_path"]
        logger.info(f"Loading models from {models_path}")
        model = self._load_model(models_path / "model.joblib")
        encoder = self._load_model(models_path / "encoder.joblib")
        lb = self._load_model(models_path / "lb.joblib")
        return model, encoder, lb

    def _evaluate_slice(self,
                        feature: str,
                        value: str,
                        df: pd.DataFrame) -> str:
        """Evaluate model performance on a specific slice."""
        X_test, y_test, _, _ = process_data(
            df,
            self.config["categorical_features"],
            label="salary",
            encoder=self.encoder,
            lb=self.lb,
            training=False,
        )
        y_pred = self.model.predict(X_test)
        test_pred_binary = (y_pred >= 0.5).astype(int)
        precision, recall, fbeta = compute_model_metrics(y_test,
                                                         test_pred_binary)
        return (
            f"{feature} - {value} :: "
            f"Precision: {precision:.2f}. "
            f"Recall: {recall:.2f}. "
            f"Fbeta: {fbeta:.2f}"
            )

    def test_performance(self):
        """Test model performance on categorical feature slices."""
        _, test = train_test_split(self.data,
                                   test_size=self.config["test_size"])
        slice_metrics = []

        for feature in self.config["categorical_features"]:
            for value in test[feature].unique():
                df_temp = test[test[feature] == value]
                metrics_row = self._evaluate_slice(feature, value, df_temp)
                slice_metrics.append(metrics_row)

        self._save_metrics(slice_metrics)
        logger.info(
            f"Performance metrics for slices saved to "
            f"{self.config['output_file']}"
            )

    def _save_metrics(self, metrics: List[str]):
        """Save metrics to a file."""
        self.config["output_file"].parent.mkdir(parents=True, exist_ok=True)
        with open(self.config["output_file"], "w") as file:
            file.write("\n".join(metrics))


if __name__ == "__main__":
    tester = ModelPerformanceTester(CONFIG)
    tester.test_performance()
