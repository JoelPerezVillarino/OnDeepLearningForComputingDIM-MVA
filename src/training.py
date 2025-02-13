import os, gc, datetime, shutil
import json
import numpy as np
import tensorflow as tf

from src.nn import loadSequentialModel
from src.nn import Normalization, Standarization, AutomaticLrScheduler
from src.utils import timer


class Train:

    def __init__(
        self,
        model_label=None,
        dataset_name=None,
        network_params=None,
        early_stopping_config=None,
        lr_schedule_config=None,
        training_config=None,
        num_trainings=None,
    ):
        self.model_label = model_label
        self.dataset_name = dataset_name
        self.network_params = network_params
        self.early_stopping_config = early_stopping_config
        self.lr_schedule_config = lr_schedule_config
        self.training_config = training_config
        self.num_trainings = num_trainings

        self.callbacks = list()

        if (self.dataset_name is None or self.network_params is None or self.training_config is None\
            or self.num_trainings is None or self.model_label is None):
            raise ValueError('Class not properly initialized')
        
        self.pwd = os.getcwd()
        self.dataset_path = None

        # Prepare folder to store the training networks and the results
        self.nn_weights_path = os.path.join(self.pwd, "nn_weights", self.model_label)
        self.results_path = os.path.join(self.pwd, "results", self.model_label)
        os.makedirs(self.nn_weights_path, exist_ok=True)
        os.makedirs(self.results_path, exist_ok=True)

        self.nn_weights_path = os.path.join(self.nn_weights_path, self.dataset_name)
        os.makedirs(self.nn_weights_path)
        self.results_path = os.path.join(self.results_path, self.dataset_name)
        os.makedirs(self.results_path)


    @classmethod
    def from_json(cls, json_file):
        with open(json_file, "r") as file:
            data = json.load(file)
        return cls(**data)
    
    @timer
    def load_dataset(self):
        self.dataset_path = os.path.join(self.pwd,"data",self.model_label,"dataset"+"-"+self.dataset_name)
        print(f"Loading dataset '{self.dataset_path}'...")
        # Load params min and params max
        self.params_min = np.load(os.path.join(self.dataset_path, "params_min.npy")) 
        self.params_max = np.load(os.path.join(self.dataset_path, "params_max.npy")) 
        # Load monitoring times (last monitoring time is ignored)
        self.monitoring_times = np.load(os.path.join(self.dataset_path, "monitoring_times.npy"))[:-1]
        # Load funding spread values for mva computation
        self.cumfs = np.load(os.path.join(self.dataset_path, "funding_spread_discounts.npy"))
        # Load training set
        self.x_train = np.load(os.path.join(self.dataset_path, "Xtrain.npy"))
        self.y_train = np.load(os.path.join(self.dataset_path, "IMtrain.npy"))[:,:-1]
        # Load validation set
        self.x_val = np.load(os.path.join(self.dataset_path, "Xval.npy"))
        self.y_val = np.load(os.path.join(self.dataset_path, "DIMval.npy"))[:,:-1]
        self.mva = np.load(os.path.join(self.dataset_path, "MVA.npy"))
        print("Done!")
        return
    
    def load_training_callbacks(self):
        if self.early_stopping_config is not None:
            self.callbacks.append(tf.keras.callbacks.EarlyStopping(**self.early_stopping_config))
        if self.lr_schedule_config is not None:
            self.callbacks.append(AutomaticLrScheduler(**self.lr_schedule_config))
        return

    
