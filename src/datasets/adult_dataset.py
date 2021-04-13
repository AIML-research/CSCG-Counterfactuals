import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder

from datasets.dataset import Dataset


class AdultDataset(Dataset):
    def __init__(self):
        super().__init__(name="Adult Census", description="The Adult Census dataset")

        self.cat_mappings = {
            "education": {
                "School": 0,
                "HS-grad": 1,
                "Some-college": 2,
                "Prof-school": 3,
                "Assoc": 4,
                "Bachelors": 5,
                "Masters": 6,
                "Doctorate": 7,
            },
            "marital_status": {
                "Divorced": 0,
                "Married": 1,
                "Separated": 2,
                "Single": 3,
                "Widowed": 4,
            },
            "workclass": {
                "Other/Unknown": 0,
                "Government": 1,
                "Private": 2,
                "Self-Employed": 3,
            },
            "occupation": {
                "Other/Unknown": 0,
                "Blue-Collar": 1,
                "Professional": 2,
                "Sales": 3,
                "Service": 4,
                "White-Collar": 5,
            },
            "race": {
                "White": 0,
                "Other": 1,
            },
            "gender": {
                "Male": 0,
                "Female": 1,
            },
            "native_country": {
                "?": 0,
                "Cambodia": 1,
                "Canada": 2,
                "China": 3,
                "Columbia": 4,
                "Cuba": 5,
                "Dominican-Republic": 6,
                "Ecuador": 7,
                "El-Salvador": 8,
                "England": 9,
                "France": 10,
                "Germany": 11,
                "Greece": 12,
                "Guatemala": 13,
                "Haiti": 14,
                "Holand-Netherlands": 15,
                "Honduras": 16,
                "Hong": 17,
                "Hungary": 18,
                "India": 19,
                "Iran": 20,
                "Ireland": 21,
                "Italy": 22,
                "Jamaica": 23,
                "Japan": 24,
                "Laos": 25,
                "Mexico": 26,
                "Nicaragua": 27,
                "Outlying-US(Guam-USVI-etc)": 28,
                "Peru": 29,
                "Philippines": 30,
                "Poland": 31,
                "Portugal": 32,
                "Puerto-Rico": 33,
                "Scotland": 34,
                "South": 35,
                "Taiwan": 36,
                "Thailand": 37,
                "Trinadad&Tobago": 38,
                "United-States": 39,
                "Vietnam": 40,
                "Yugoslavia": 41,
            },
        }

        self.inv_cat_mappings = {
            key: {v: k for k, v in mapping.items()}
            for key, mapping in self.cat_mappings.items()
        }

        self.__init_encoder()

    def load(self) -> pd.DataFrame:
        """Loads adult income dataset from https://archive.ics.uci.edu/ml/datasets/Adult and prepares the data for data analysis based on https://rpubs.com/H_Zhu/235617

        :param: save_intermediate: save the transformed dataset. Do not save by default.
        """
        raw_data = np.genfromtxt(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
            delimiter=", ",
            dtype=str,
        )

        #  column names from "https://archive.ics.uci.edu/ml/datasets/Adult"
        column_names = [
            "age",
            "workclass",
            "fnlwgt",
            "education",
            "educational-num",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "gender",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
            "native-country",
            "income",
        ]

        adult_data = pd.DataFrame(raw_data, columns=column_names)

        # For more details on how the below transformations are made, please refer to https://rpubs.com/H_Zhu/235617
        adult_data = adult_data.astype(
            {"age": np.int64, "educational-num": np.int64, "hours-per-week": np.int64}
        )

        adult_data = adult_data.replace(
            {
                "workclass": {
                    "Without-pay": "Other/Unknown",
                    "Never-worked": "Other/Unknown",
                }
            }
        )
        adult_data = adult_data.replace(
            {
                "workclass": {
                    "Federal-gov": "Government",
                    "State-gov": "Government",
                    "Local-gov": "Government",
                }
            }
        )
        adult_data = adult_data.replace(
            {
                "workclass": {
                    "Self-emp-not-inc": "Self-Employed",
                    "Self-emp-inc": "Self-Employed",
                }
            }
        )
        # adult_data = adult_data.replace(
        #     {
        #         "workclass": {
        #             "Never-worked": "Self-Employed",
        #             "Without-pay": "Self-Employed",
        #         }
        #     }
        # )
        adult_data = adult_data.replace({"workclass": {"?": "Other/Unknown"}})

        adult_data = adult_data.replace(
            {
                "occupation": {
                    "Adm-clerical": "White-Collar",
                    "Craft-repair": "Blue-Collar",
                    "Exec-managerial": "White-Collar",
                    "Farming-fishing": "Blue-Collar",
                    "Handlers-cleaners": "Blue-Collar",
                    "Machine-op-inspct": "Blue-Collar",
                    "Other-service": "Service",
                    "Priv-house-serv": "Service",
                    "Prof-specialty": "Professional",
                    "Protective-serv": "Service",
                    "Tech-support": "Service",
                    "Transport-moving": "Blue-Collar",
                    "Unknown": "Other/Unknown",
                    "Armed-Forces": "Other/Unknown",
                    "?": "Other/Unknown",
                }
            }
        )

        adult_data = adult_data.replace(
            {
                "marital-status": {
                    "Married-civ-spouse": "Married",
                    "Married-AF-spouse": "Married",
                    "Married-spouse-absent": "Married",
                    "Never-married": "Single",
                }
            }
        )

        adult_data = adult_data.replace(
            {
                "race": {
                    "Black": "Other",
                    "Asian-Pac-Islander": "Other",
                    "Amer-Indian-Eskimo": "Other",
                }
            }
        )

        # adult_data = adult_data[['age','workclass','education','marital-status','occupation','race','gender',
        #                 'hours-per-week','income']]

        adult_data = adult_data[
            [
                "age",
                "capital-gain",
                "hours-per-week",
                "workclass",
                "education",
                "marital-status",
                "occupation",
                "race",
                "gender",
                "capital-loss",
                "native-country",
                "income",
            ]
        ]
        # adult_data = adult_data[
        #     [
        #         "age",
        #         "hours-per-week",
        #         "workclass",
        #         "education",
        #         "marital-status",
        #         "occupation",
        #         "race",
        #         "gender",
        #         "native-country",
        #         "income",
        #     ]
        # ]

        adult_data = adult_data.replace({"income": {"<=50K": 0, ">50K": 1}})

        adult_data = adult_data.replace(
            {
                "education": {
                    "Assoc-voc": "Assoc",
                    "Assoc-acdm": "Assoc",
                    "11th": "School",
                    "10th": "School",
                    "7th-8th": "School",
                    "9th": "School",
                    "12th": "School",
                    "5th-6th": "School",
                    "1st-4th": "School",
                    "Preschool": "School",
                }
            }
        )

        adult_data = adult_data.rename(
            columns={
                "marital-status": "marital_status",
                "hours-per-week": "hours_per_week",
                "capital-gain": "capital_gain",
                "native-country": "native_country",
                "capital-loss": "capital_loss",
            }
        )

        return adult_data.drop("income", axis=1), adult_data["income"]

    def extract_info(self):
        columns = self.dataset.columns
        target = "income"
        real_feat = np.array(
            [
                0,  # age
                1,  # capital-gain
                2,  # hours-per-week
                9,  # capital-loss
            ]
        )
        cat_feat = np.array(
            [
                3,  # workclass
                4,  # education
                5,  # marital
                6,  # occupation
                7,  # race
                8,  # gender
                10,  # native-country
            ]
        )
        _both = np.concatenate([real_feat, cat_feat])
        _cond = (np.sort(_both) == np.arange(0, max(_both) + 1)).all()
        assert _cond

        # real_feat = np.array(
        #     [
        #         0,  # age
        #         1,  # hours-per-week
        #     ]
        # )
        # cat_feat = np.array(
        #     [
        #         2,  # workclass
        #         3,  # education
        #         4,  # marital
        #         5,  # occupation
        #         6,  # race
        #         7,  # gender
        #         8,  # native country
        #     ]
        # )
        return columns, target, real_feat, cat_feat

    def __init_encoder(self):
        self.encoder = OneHotEncoder(sparse=False)
        X = self.get_optimizer_data().copy()
        self.encoder.fit(X[:, self.cat_features])
        return self.encoder

    def encode_features(self, X: np.array) -> np.array:
        onehot = self.encoder.transform(X[:, self.cat_features])
        n_real = len(self.real_features)
        n_onehot = onehot.shape[1]
        _X = np.zeros((X.shape[0], n_real + n_onehot))
        _X[:, :n_real] = X[:, self.real_features]
        _X[:, n_real:] = onehot  # .astype(int)
        return _X.astype(int)

    def decode_features(self, X: np.array) -> np.array:
        _X = np.zeros((X.shape[0], self.dataset.shape[1]))
        n_real = len(self.real_features)
        orig_cat = self.encoder.inverse_transform(X[:, n_real:])
        _X[:, self.real_features] = X[:, :n_real].copy()
        _X[:, self.cat_features] = orig_cat
        return _X.astype(int)

    def preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        df = self.dataset.copy()
        return df.replace(self.cat_mappings)

    def get_optimizer_data(self) -> np.array:
        X = self.get_numpy_representation()
        X[:, self.real_features] = X[:, self.real_features].astype(float)
        X[:, self.cat_features] = X[:, self.cat_features].astype(int)
        return X.astype(int)

    def get_classifier_data(self):
        X = self.get_optimizer_data().copy()
        return self.encode_features(X), self.labels

    def get_processed_orig_data(self, X: np.array) -> pd.DataFrame:
        df = pd.DataFrame(X, columns=self.columns)
        df = df.replace(self.inv_cat_mappings)
        return df