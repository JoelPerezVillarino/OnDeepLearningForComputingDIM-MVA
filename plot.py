import os
import numpy as np
import matplotlib.pyplot as plt


# folder = os.path.join(os.getcwd(),"data","hull_white", "dataset-1Yr5YrSwap")
folder = os.path.join(os.getcwd(),"data","cir", "dataset-1Yr5YrSwap")
ts = np.load(os.path.join(folder, "monitoring_times.npy"))
x_train = np.load(os.path.join(folder, "Xtrain.npy"))
IM_train = np.load(os.path.join(folder, "IMtrain.npy"))
DIM_val = np.load(os.path.join(folder, "DIMval.npy"))


plt.figure(figsize=(12,8))
# plt.plot(ts,IM_train.T)
plt.plot(ts,DIM_val.T)
plt.show()