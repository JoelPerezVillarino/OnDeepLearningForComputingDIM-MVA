import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
tf.keras.backend.set_floatx("float64")

from src.nn import loadSequentialModel, Normalization

key = "num_samples_1024"
model_label = "model_9.h5"

# Paths
cwd = os.getcwd()
path_data = os.path.join(cwd, "data", "cir", "dataset-1Yr5YrSwap")
path_weights = os.path.join(cwd, "trained_models", "cir", "1Yr5YrSwap", key, model_label)

# Load dataset
x_val = np.load(os.path.join(path_data, "Xval.npy"))
y_val = np.load(os.path.join(path_data, "DIMval.npy"))[:, :-1]
y_train = np.load(os.path.join(path_data, "IMtrain.npy"))[:, :-1]
mva_val = np.load(os.path.join(path_data, "MVA.npy"))
monitoring_times = np.load(os.path.join(path_data, "monitoring_times.npy"))[:-1]
num_outputs = monitoring_times.size

params_min = np.load(os.path.join(path_data, "params_min.npy"))
params_max = np.load(os.path.join(path_data, "params_max.npy"))


# Load results
path_results = os.path.join(cwd, "results", "cir", "1Yr5YrSwap")
mae = np.loadtxt(os.path.join(path_results, "mae.txt"))
mse = np.loadtxt(os.path.join(path_results, "mse.txt"))
num_samples = np.loadtxt(os.path.join(path_results, "num_samples.txt"))
mva_error = np.load(os.path.join(path_results, "mva_error.npy"))
mean_mva_error = np.load(os.path.join(path_results, "mean_mva_error.npy"))

print(mva_val.shape)
print(mva_error.shape)
print(mean_mva_error.shape)

# preprocessing_layer = Normalization()
# preprocessing_layer.load_bounds(params_min, params_max)
# model = loadSequentialModel(256,3,num_outputs,preprocessing_layer=preprocessing_layer)
# model(params_min[:,None])
# model.load_weights(path_weights)

# y_pred = model(x_val).numpy()
# with plt.ion():
#     fig = plt.figure(figsize=(12,8))
#     ax = fig.add_subplot(111)
#     ax.set_ylim([0.,3.5])

# for i in range(100):
#     plt.cla()
#     ax.set_title(f"Sample {i+1}")
#     ax.plot(monitoring_times, y_val[i], label="true", color="blue")
#     ax.plot(monitoring_times, y_pred[i], label="pred", color="red")
#     ax.legend()
#     plt.pause(3)


# fig = plt.figure(figsize=(12,8))
# ax = fig.add_subplot(111)
# ax.plot(monitoring_times, y_val[146], "o", label="DIM")
# ax.plot(monitoring_times, y_train[3004], "o", label="IM")
# ax.legend()
