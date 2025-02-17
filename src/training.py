import os, gc
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
        opt_config=None,
        early_stopping_config=None,
        lr_schedule_config=None,
        num_trainings=None,
        epochs=None,
        batch_size=None,
    ):
        self.model_label = model_label
        self.dataset_name = dataset_name
        self.network_params = network_params
        self.opt_config = opt_config
        self.early_stopping_config = early_stopping_config
        self.lr_schedule_config = lr_schedule_config
        self.num_trainings = num_trainings
        self.epochs = epochs
        self.batch_size = batch_size

        if (self.dataset_name is None or self.network_params is None or self.epochs is None\
            or self.num_trainings is None or self.model_label is None or self.batch_size is None\
            or self.opt_config is None):
            raise ValueError('Class not properly initialized')
        
        self.pwd = os.getcwd()
        self.dataset_path = None

        # Data
        self.params_min = None
        self.params_max = None
        self.monitoring_times = None
        self.cumfs = None
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.mva = None

        # nn inferred from data
        self.num_inputs = None
        self.num_outputs = None

        # Prepare folder to store the training networks and the results
        self.nn_weights_path = os.path.join(self.pwd, "trained_models", self.model_label)
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

        self.num_inputs = self.params_min.size
        self.num_outputs = self.monitoring_times.size
        self.num_train_samples = self.x_train.shape[0]
        self.num_val_samples = self.x_val.shape[0]
        print("Done!")
        return
    
    def load_training_callbacks(self):
        callbacks = list()
        if self.early_stopping_config is not None:
            callbacks.append(tf.keras.callbacks.EarlyStopping(**self.early_stopping_config))
        if self.lr_schedule_config is not None:
            callbacks.append(AutomaticLrScheduler(**self.lr_schedule_config))
        return callbacks
    
    def load_model_config(self):
        # Preprocessing layer
        if self.network_params["preprocessing_layer"] == "normalization":
            preprocessing_layer = Normalization()
            preprocessing_layer.load_bounds(self.params_min, self.params_max)
            kernel_init = "glorot_uniform"
        elif self.network_params["preprocessing_layer"] == "standarization":
            preprocessing_layer = Standarization()
            preprocessing_layer.adapt(self.x_train)
            preprocessing_layer.save_params(self.dataset_path)
            kernel_init = "glorot_normal"
        else:
            preprocessing_layer = None
            kernel_init = "glorot_uniform"
        
        # optimizer
        if self.opt_config["opt"] == "adam":
            opt = tf.keras.optimizers.Adam(
                self.opt_config["lr"], 
                self.opt_config["beta_1"],
                self.opt_config["beta_2"]
            )
        elif self.opt_config["opt"] == "sgd":
            opt = tf.keras.optimizers.SGD(self.opt_config["lr"])
        elif self.opt_config["opt"] == "adam_w":
            opt = tf.keras.optimizers.AdamW(
                self.opt_config["lr"],
                self.opt_config["weight_dacay"],
                self.opt_config["beta_1"],
                self.opt_config["beta_2"]
            )
        
        config = {
            "units": self.network_params["num_units"],
            "num_layers": self.network_params["num_layers"],
            "activation": [self.network_params["activation"], "linear"],
            "num_inputs": self.num_inputs,
            "num_outputs": self.num_outputs,
            "kernel_init": kernel_init,
            "opt": opt,
            "preprocessing_layer": preprocessing_layer,
            "regularization": None
        }

        return config
    
    def run(self):
        self.load_dataset()
        config = self.load_model_config()
        # Defining the number of training samples employed
        m0 = 10 # Min dataset size: 2**10
        size_set = [2**i for i in range(m0,int(np.log2(self.num_train_samples))+1)]
        M = len(size_set) # Number of datasets
        # Allocate values
        val_mse = np.zeros((M, self.num_trainings))
        val_mae = np.zeros((M, self.num_trainings))
        mva_error = np.zeros((M, self.num_trainings, self.num_val_samples))
        mean_mva_error = np.zeros((M, self.num_train_samples))
        # Trainings
        x_train_sample = None
        y_train_sample = None
        for i, num_samples in enumerate(size_set):
            subfolder_i = os.path.join(self.nn_weights_path, f"num_samples_{num_samples}")
            os.makedirs(subfolder_i)
            for j in range(self.num_trainings):
                # Shuffle
                idx_samples = np.random.randint(self.num_train_samples,size=num_samples)
                x_train_sample = self.x_train[idx_samples]
                y_train_sample = self.y_train[idx_samples]
                # Load and train model
                model = loadSequentialModel(**config)
                callbacks = self.load_training_callbacks()
                history = model.fit(
                    x_train_sample, y_train_sample, epochs=self.epochs,
                    batch_size=self.batch_size, validation_data=(self.x_val, self.y_val),
                    callbacks=callbacks
                )
                # Save model and relevant data (training)
                fname_model = os.path.join(subfolder_i, f"model_{j}.h5")
                model.save_weights(fname_model)
                print(f"Model saved in '{fname_model}'")

                best_epoch = -1
                if self.early_stopping_config is not None:
                    if callbacks[0].stopped_epoch>0: best_epoch = callbacks[0].stopped_epoch

                val_mse[i,j] = history.history["val_mse"][best_epoch]
                val_mae[i,j] = history.history["val_mae"][best_epoch]

                # Predicitons
                y_pred = model(self.x_val).numpy()*self.cumfs
                # Assuming equispaced time grid 
                mva_pred = np.sum(y_pred[:,1:],axis=1)*(self.monitoring_times[1]-self.monitoring_times[0])
                mva_error[i,j,:] = np.abs(self.mva - mva_pred)
                mean_mva_error[i,j] = np.mean(mva_error[i,j,:])

                del model
                tf.keras.backend.clear_session()
                gc.collect()
        
        # Save results
        np.savetxt(os.path.join(self.results_path, "num_samples.txt"), np.array(size_set))
        np.savetxt(os.path.join(self.results_path, "mse.txt"), val_mse)
        np.savetxt(os.path.join(self.results_path, "mae.txt"), val_mae)
        np.save(os.path.join(self.results_path, "mva_error.npy"), mva_error)
        np.savetxt(os.path.join(self.results_path, "mean_mva_error.txt"), mean_mva_error)


        return
    


    
