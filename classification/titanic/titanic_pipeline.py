import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from pipelines.pipeline import Pipeline
from classification.titanic.estimators import ESTIMATORS

from pipelines import constants


class TitanicPipeline(Pipeline):
    def __init__(self, descriptor_folder: str | None = None, descriptor_file: str | None = None):
        """
    The __init__ function is the first function called when you create a new instance of a class.
    It's job is to initialize all of the attributes of the class.


    :param self: Represent the instance of the class
    :param descriptor_folder: str | None: Specify the folder where the descriptor files are located
    :param descriptor_file: str | None: Specify the file name of the descriptor
    :return: Nothing
    :doc-author: Trelent
    """
        super().__init__(descriptor_folder=descriptor_folder, descriptor_file=descriptor_file)
        self._scorer = None
        self._train_dataset = None
        self._test_dataset = None
        self._labels: pd.Series | None = None
        self._features: pd.DataFrame | None = None
        self._x_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None
        self._x_test: np.ndarray | None = None
        self._scaler: StandardScaler | None = None

    def prepare_metrics(self):
        self._scorer = make_scorer(accuracy_score,
                                   greater_is_better=False)

    def pre_process_data(self):
        self._train_dataset = self._open_dataset("train")
        self._test_dataset = self._open_dataset("test")

        self._labels: pd.DataFrame = self._train_dataset["Survived"]
        self._features: pd.DataFrame = self._train_dataset.drop(columns="Survived")

        self._features = self._impute_missing_data(self._features)
        self._test_dataset = self._impute_missing_data(self._test_dataset)

    def _open_dataset(self, reference: str) -> pd.DataFrame:
        folder = self.descriptor["datasets"][reference]["storage"]["container"]
        file_name = self.descriptor["datasets"][reference]["name"]
        resource_path = constants.ROOT_FOLDER + folder + file_name
        titanic = pd.read_csv(resource_path)
        titanic = titanic.drop(columns=["Name", "Ticket", "Cabin"])
        titanic["Sex"].replace({"male": 1, "female": 0}, inplace=True)
        titanic["Embarked"].replace({"S": 1, "C": 2, "Q": 3}, inplace=True)

        return titanic

    @staticmethod
    def _impute_missing_data(dataset: pd.DataFrame):
        age: np.ndarray = np.array(dataset["Age"])
        age = age.reshape(-1, 1)
        imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
        imp_median.fit(age)
        age: np.ndarray = imp_median.transform(age)
        count_nan_after: int = np.count_nonzero(np.isnan(age))
        assert count_nan_after == 0
        dataset["Age"] = age

        fare = np.array(dataset["Fare"])
        fare = fare.reshape(-1, 1)
        imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
        imp_median.fit(fare)
        fare = imp_median.transform(fare)
        count_nan_after: int = np.count_nonzero(np.isnan(fare))
        assert count_nan_after == 0
        dataset["Fare"] = fare

        embark = np.array(dataset["Embarked"])
        embark = embark.reshape(-1, 1)
        imp_constant = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=1)
        imp_constant.fit(embark)
        embark = imp_constant.transform(embark)
        count_nan_after: int = np.count_nonzero(np.isnan(embark))
        assert count_nan_after == 0
        dataset["Embarked"] = embark

        return dataset

    def calculate_features(self):
        ...

    def calculate_labels(self):
        ...

    def scale(self, features: np.ndarray):
        scaler = StandardScaler()
        scaler.fit(features)
        features = scaler.transform(features)
        features_mean = features.mean().round(5)
        features_std = features.std().round(5)  # todo parametrize this
        assert features_mean == 0
        return features

    def train_estimator(self, estimator_key: str) -> GridSearchCV:
        # todo: receive dataset as parameter
        estimator = ESTIMATORS[estimator_key]["estimator"]()
        hyperparameters_space = ESTIMATORS[estimator_key]["hyperparameters_space"]
        grid_search = GridSearchCV(estimator,
                                   hyperparameters_space,
                                   scoring=self._scorer)

        x_train = self.scale(self._x_train)

        if ESTIMATORS[estimator_key]["scale"]:
            x_train = self.scale(self._x_train)

        grid_search.fit(x_train,
                        self._y_train)

        grid_search_file_name = self._build_grid_search_output_path(estimator_key)
        result_dataframe = pd.DataFrame(grid_search.cv_results_)
        result_dataframe.to_csv(f"{grid_search_file_name}_grid_search_result.csv")
        grid_search.best_estimator_

        return grid_search

    def _build_grid_search_output_path(self, estimator_key: str) -> str:
        output_path: str = constants.ROOT_FOLDER + self.descriptor["outputs"]["storage"]["container"]
        return output_path

    def split_train_test(self):
        self._x_train = np.array(self._features)
        self._y_train = np.array(self._labels)
        self._x_test = np.array(self._test_dataset)


    def train_models(self):
        ...
