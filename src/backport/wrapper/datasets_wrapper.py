import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

from datasets.dataset import Dataset
from backport.utils import RepresentationTranslator

from competitor.actions.feature import CategoricFeature, Feature


class DatasetWrapper(Dataset):
    def __init__(self, name, legacy_dataset, feature_objects):
        self.legacy_dataset = legacy_dataset

        self.feature_objects = feature_objects.copy()
        self.translator = RepresentationTranslator(self.feature_objects.copy())

        super().__init__(name=name, description="Competitor dataset wrapper.")

    def extract_info(self):
        columns = np.array(list(self.feature_objects.keys()))
        order = np.argsort([f.idx for _, f in self.feature_objects.items()])
        columns = columns[order]
        real_features = []
        cat_features = []
        for i, feature_key in enumerate(columns):
            feature = self.feature_objects[feature_key]
            if isinstance(feature, Feature):
                real_features.append(i)
            elif isinstance(feature, CategoricFeature):
                cat_features.append(i)

        return columns, self.legacy_dataset.labels, real_features, cat_features

    def load(self) -> pd.DataFrame:
        (
            self.zs,
            self.xs,
            self.train_xs,
            self.test_xs,
            self.train_zs,
            self.test_zs,
        ) = self.legacy_dataset.get_relevant_data()

        self.label_encoder = LabelEncoder()
        labels = self.label_encoder.fit_transform(self.train_xs[:, -1])

        # ! train_zs [:-2], -2 because the last two columns are actualy the labels..
        return self.train_zs[:, :-2], labels

    def transform_to_original(self, X: np.array) -> np.array:
        translated = np.array([self.translator.instance_to_x(x) for x in X])
        return translated
        # raise Exception("Function not defined")

    """Like one-hot encoding etc."""

    def encode_features(self, X: np.array) -> np.array:
        translated = np.array([self.translator.instance_to_z(x) for x in X])
        return translated
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
        return self.train_xs[:, :-1]

    def numpy_to_df(self, X: np.array) -> pd.DataFrame:
        return pd.DataFrame(X)

    def get_optimizer_data(self) -> np.array:
        return self.train_xs[:, :-1]

    def get_classifier_data(self) -> np.array:
        return self.train_zs[:, :-2], self.label_encoder.transform(self.train_xs[:, -1])

    def get_test_data(self) -> np.array:
        # * Labels are xs and not zs because through the legacy normalization they also
        # * Changed the labels
        # ! test_zs [:-2], -2 because the last two columns are actualy the labels..
        return self.test_zs[:, :-2], self.label_encoder.transform(self.test_xs[:, -1])

    def get_processed_orig_data(self, X: np.array) -> pd.DataFrame:
        df = pd.DataFrame(X, columns=self.columns)
        return df
