import abc
from abc import ABC

import os
import numpy as np

from pipelines.classification.pipeline_descriptor import PipelineDescriptor


class Pipeline(ABC):
    def __init__(self, *args, **kwargs):
        descriptor_folder = kwargs["descriptor_folder"]
        descriptor_name = kwargs["descriptor_file"]
        self.descriptor: dict = PipelineDescriptor(descriptor_folder,
                                                    descriptor_name).open()

    def run(self):
        self.prepare_metrics()
        self.pre_process_data()
        self.calculate_features()
        self.calculate_labels()
        self.split_train_test()
        self.train_models()

    @abc.abstractmethod
    def create_output_folder(self):
        ...

    @abc.abstractmethod
    def prepare_metrics(self):
        ...

    @abc.abstractmethod
    def train_estimator(self):
        ...

    @abc.abstractmethod
    def pre_process_data(self):
        ...

    @abc.abstractmethod
    def calculate_features(self):
        ...

    @abc.abstractmethod
    def calculate_labels(self):
        ...

    @abc.abstractmethod
    def split_train_test(self):
        ...

    @abc.abstractmethod
    def scale(self, features: np.ndarray) -> np.ndarray:
        ...

    @abc.abstractmethod
    def train_models(self):
        ...

