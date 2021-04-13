import pandas as pd
import numpy as np


class Dataset:
    def __init__(self, name: str, description=""):
        self.name = name
        self.description = description

        self.dataset, self.labels = self.load()

        (
            self.columns,
            self.target,
            self.real_features,
            self.cat_features,
        ) = self.extract_info()

    def extract_info(self):
        raise Exception("Function not defined")

    def load(self) -> pd.DataFrame:
        raise Exception("Function not defined")

    def transform_to_original(self, X: np.array) -> np.array:
        return X
        # raise Exception("Function not defined")

    """Like one-hot encoding etc."""

    def encode_features(self, X: np.array) -> np.array:
        return X
        # raise Exception("Function not defined")

    def decode_features(self, X: np.array) -> np.array:
        return X
        # raise Exception("Function not defined")

    """Filter out certain variables before etc.
    Also, transform cat to numerical"""

    def preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        return X
        # raise Exception("Function not defined")

    def get_numpy_representation(self) -> np.array:
        # encoded = self.encode_features(self.dataset.copy())
        preprocessed = self.preprocess(self.dataset.copy())
        X = preprocessed.values.astype(float)
        return X

    def numpy_to_df(self, X: np.array) -> pd.DataFrame:
        original_values = self.transform_to_original(X.copy())
        df = pd.DataFrame(original_values, columns=self.columns)
        return df

    def get_optimizer_data(self) -> np.array:
        raise Exception("Function not defined")

    def get_classifier_data(self) -> np.array:
        raise Exception("Function not defined")

    def get_test_data(self) -> np.array:
        return self.dataset.copy()

    def get_processed_orig_data(self, X: np.array) -> pd.DataFrame:
        raise Exception("Function not defined")
