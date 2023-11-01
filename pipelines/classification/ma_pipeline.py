"""
Pipeline to estimate moving average

## References

https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
https://scikit-learn.org/stable/modules/model_evaluation.html#defining-your-scoring-strategy-from-metric-functions

Regressors
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge

"""
import logging
import sys
import pickle
import os

import numpy as np
import pandas as pd
from trading_lib.utils.dataset_management import open_csv

import pipelines.constants as const

from pipelines.classification.pipeline_descriptor import PipelineDescriptor

from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler


class Pipeline:
    ESTIMATORS = {
        "sgd":
            {
                "estimator": SGDRegressor,
                "hyperparameters_space":
                    {
                        "alpha": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
                        "max_iter": [1000, 1500, 2000],
                        "random_state": [42]
                    },
                "scale": True
            },
        "ridge":
            {
                "estimator": Ridge,
                "hyperparameters_space":
                    {
                        "alpha": [1.0, 1.5, 2.0, 2.5, 5, 10],
                        "solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
                        "max_iter": [1000]
                    },
                "scale": True
            },
        "svr":
            {
                "estimator": SVR,
                "hyperparameters_space":
                    {
                        "kernel": ["linear", "poly", "rbf", "sigmoid"],
                        "C": [0.1, 0.2, 0.5, 1, 1.5, 2, 2.5, 5]
                    },
                "scale": True
            }
    }

    def __init__(self,
                 ma_periods: int,
                 time_column_name,
                 lag_periods: int,
                 label_source_column: str,
                 train_set_length_ratio: float | None = None,
                 test_set_length_ratio: float | None = None,
                 label_column: str = "label",
                 cross_validate_splits: int = 10,
                 root_output_folder: str = "./output/",
                 descriptor_folder: str = ".",
                 descriptor_name: str = "descriptor.json"
                 ):
        self.ma_periods: int = ma_periods
        self.time_column_name: str = time_column_name
        self.lag_periods: int = lag_periods
        self.label_source_column: str = label_source_column
        self._root_output_folder = root_output_folder

        if train_set_length_ratio is not None:
            self.train_set_length_ratio = train_set_length_ratio
            self.test_set_length_ratio = 1 - self.train_set_length_ratio
        else:
            if test_set_length_ratio is not None:
                self.test_set_length_ratio: float = test_set_length_ratio
                self.train_set_length_ratio = 1 - self.test_set_length_ratio
            else:
                raise ValueError("Either train_set_length_ratio or test_set_length_ratio need to specify a value")

        self.label_column: str = label_column
        self._cross_validate_splits: int = cross_validate_splits
        self._scorer_rmse = make_scorer(mean_absolute_error,
                                        greater_is_better=False)
        self.features: list = []
        self.process_tracking: dict = {}
        if descriptor_folder[-1] != "/":
            descriptor_folder += "/"

        self._descriptor: dict = PipelineDescriptor(descriptor_folder,
                                                    descriptor_name).open()

        self._dataset_path = self.build_dataset_path()
        self._output_folder = self.build_output_folder_path()
        self._pipeline_record: dict = {}

    def build_dataset_path(self) -> str:
        folder: str = const.ROOT_FOLDER + self._descriptor["dataset"]["storage"]["container"]
        os.makedirs(folder, exist_ok=True)
        return folder + self._descriptor["dataset"]["name"]

    def build_output_folder_path(self) -> str:
        folder: str = const.ROOT_FOLDER + self._descriptor["outputs"]["storage"]["container"]
        if folder[-1] != "/":
            folder += "/"
        os.makedirs(folder, exist_ok=True)
        return folder

    def run(self):
        # TODO: invoke this function from command line
        # TODO: build the name of the file in a function
        # TODO :move dataset handling by dask

        file_name: str = const.ROOT_FOLDER + self._descriptor["dataset"]["folder"] + \
                         self._descriptor["dataset"]["storage"]["container"]
        price: pd.DataFrame = open_dataset(file_name,
                                           self.time_column_name,
                                           drop_cols=False)

        price = self.pre_process_data(price)
        price = self.calculate_features(price)
        price = self.calculate_labels(price)
        self.split_train_validate_test(price)
        self.scale()
        self.train_models()

    def pre_process_data(self, dataset: str) -> pd.DataFrame:
        # TODO: open dataset here and save it to disk as dask once is processed

        return self.data_quality_check(price)

    # TODO: Refactor this method name to missing_periods_check and add fix parameter as boolean

    def data_quality_check(self, price: pd.DataFrame) -> pd.DataFrame:
        missing_periods_check = MissingPeriodsChecks(price)
        missing_periods = missing_periods_check.scan()
        missing_periods.to_csv(self._output_folder + "/missing_periods.csv")
        self.process_tracking["quality-check"] = {}
        self.process_tracking["quality-check"]["gaps-found"] = len(missing_periods)

        price = missing_periods_check.fix()

        return price

    def calculate_features(self, price: pd.DataFrame):

        # TODO: save the dataset and return dataset name or payload for next process
        ma_features = MAFeatures(self.ma_periods,
                                 "Close",
                                 self.lag_periods)
        price = ma_features.compute(price)
        self.features.extend(ma_features.features)

        return price

    def calculate_labels(self, price: pd.DataFrame) -> pd.DataFrame:
        # TODO: save the dataset and return dataset name or payload for next process
        price[self.label_column] = price[self.label_source_column].shift(-1)
        price.dropna(inplace=True)

        return price

    def split_train_validate_test(self, price):

        # TODO: receives parameter as payload to open the price file

        train_records: int = int(self.train_set_length_ratio * len(price))
        train_set = price[:train_records]
        test_set = price[train_records:]
        self._pipeline_record["x-train"] = np.array(train_set[self.features])
        self._pipeline_record["y-train"] = np.array(train_set[self.label_column])

        self._pipeline_record["x-test"] = np.array(test_set[self.features])
        self._pipeline_record["y-test"] = np.array(test_set[self.label_column])

        base_name = self._build_datasets_base_name()
        test_set[self.features].to_csv(f"{base_name}_x_test.csv")
        test_set[self.label_column].to_csv(f"{base_name}_y_test.csv")

    def _build_datasets_base_name(self) -> str:
        metadata = self._descriptor["dataset"]["metadata"]
        return f"{self._output_folder}{metadata['instrument']}_{metadata['time-frame']}_{metadata['price-type']}"

    def scale(self):
        # todo: receives parameters as payload to open dataset to fit
        scaler = StandardScaler()
        scaler.fit(self._pipeline_record["x-train"])
        self._pipeline_record["x-train-scaled"] = scaler.transform(self._pipeline_record["x-train"])

        # todo: save the scaled dataset
        with open(self._build_scaler_file_name(), "wb") as file_pointer:
            pickle.dump(scaler, file_pointer)

    def _build_scaler_file_name(self) -> str:
        dataset_metadata = self._descriptor["dataset"]["metadata"]
        file_name: str = (f"{self._output_folder}scaler_{dataset_metadata['instrument']}-"
                          f"{dataset_metadata['time-frame']}-"
                          f"{dataset_metadata['price-type']}.pkl")
        return file_name

    def train_models(self):
        # todo: receive parameter as payload to open datasets
        logging.info("Train/test process started")
        train_test_record: dict = {}
        for estimator in self.ESTIMATORS.keys():
            logging.info(f"Training estimator {estimator} ...")
            estimator_record: dict = {}
            grid_search_result: GridSearchCV = self.train_estimator(estimator)
            logging.info(f"Testing estimator {estimator} ...")
            validation_record: dict = self.test_estimator(estimator)
            estimator_record["train-metric"] = grid_search_result.best_score_
            best_index = grid_search_result.best_index_
            estimator_record["best-estimator-index"] = best_index
            estimator_record["best-estimator-mean"] = -1*grid_search_result.cv_results_["mean_test_score"][best_index]
            estimator_record["best-estimator-std"] = grid_search_result.cv_results_["std_test_score"][best_index]
            estimator_record.update(validation_record)
            train_test_record[estimator]: dict = estimator_record

            logging.info(f"Estimator {estimator} processing finished")
        train_test_dataset: pd.DataFrame = pd.DataFrame.from_dict(train_test_record, orient="index")
        metrics_file_name = self._build_metrics_file_name()
        train_test_dataset.to_csv(metrics_file_name)
        logging.info("Train/Test process finished")

    def train_estimator(self, estimator_key: str) -> GridSearchCV:

        # todo: receive dataset as parameter
        estimator = self.ESTIMATORS[estimator_key]["estimator"]()
        hyperparameters_space = self.ESTIMATORS[estimator_key]["hyperparameters_space"]
        time_series_splitter = TimeSeriesSplit(gap=0,
                                               max_train_size=None,
                                               n_splits=self._cross_validate_splits,
                                               test_size=None)
        grid_search = GridSearchCV(estimator,
                                   hyperparameters_space,
                                   cv=time_series_splitter,
                                   scoring=self._scorer_rmse)

        if self.ESTIMATORS[estimator_key]["scale"]:
            grid_search.fit(self._pipeline_record["x-train-scaled"],
                            self._pipeline_record["y-train"])
        else:
            grid_search.fit(self._pipeline_record["x-train"],
                            self._pipeline_record["y-train"])

        grid_search_file_name: str = self._build_grid_search_file_name(estimator_key)
        with open(grid_search_file_name, "wb") as file_pointer:
            pickle.dump(grid_search, file_pointer)

        result_dataframe = pd.DataFrame(grid_search.cv_results_)
        result_dataframe.to_csv(f"{self._output_folder}{estimator_key}_grid_search_result.csv")

        return grid_search

    def _build_grid_search_file_name(self, estimator: str) -> str:
        return f"{self._output_folder}_grid_search.pkl"

    def test_estimator(self, estimator_key: str) -> dict:
        grid_search_file_name: str = self._build_grid_search_file_name(estimator_key)
        with open(grid_search_file_name, "br") as file_pointer:
            grid_search_result: GridSearchCV = pickle.load(file_pointer)

        estimator = grid_search_result.best_estimator_

        if self.ESTIMATORS[estimator_key]["scale"]:
            with open(self._build_scaler_file_name(), "rb") as file_pointer:
                scaler = pickle.load(file_pointer)
            x_test = scaler.transform(self._pipeline_record["x-test"])
        else:
            x_test = self._pipeline_record["x-test"]

        y_predict: np.array = estimator.predict(x_test)

        predict_file_name: str = self._build_predictions_file_name(estimator_key)
        np.savetxt(predict_file_name, y_predict, delimiter=",")
        predict_metric: float = mean_absolute_error(self._pipeline_record["y-test"], y_predict)
        residuals = self._pipeline_record["y-test"] - y_predict
        residuals_mean = residuals.mean()
        residuals_std = residuals.std()

        validation_record = {
            "best_train_score": np.abs(grid_search_result.best_score_),
            "test_metric": predict_metric,
            "hyperparameters": grid_search_result.best_params_,
            "test-residuals-mean": residuals_mean,
            "test-residuals-std": residuals_std
        }

        return validation_record

    def _build_predictions_file_name(self, estimator: str) -> str:
        return f"{self._output_folder}{estimator}_test_predictions.csv"

    def _build_metrics_file_name(self):
        return f"{self._output_folder}metrics.csv"
