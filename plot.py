import os
import numpy as np
import matplotlib.pyplot as plt

from src.test import convergence_num_train_samples, compute_errors_per_param


convergence_num_train_samples('cir', '1Yr5YrSwap', save=False)
# compute_errors_per_param('cir', '1Yr5YrSwap', 'num_samples_4194304', save=False)
# compute_errors_per_param('cir', '1Yr5YrSwap', 'num_samples_131072', save=False) # Works fine
# compute_errors_per_param('cir', '1Yr5YrSwap', 'num_samples_65536', save=False) # Works fine

# convergence_num_train_samples('hull_white', '1Yr5YrSwap', save=False)
# compute_errors_per_param('hull_white', '1Yr5YrSwap', 'num_samples_4194304', save=False) 
# compute_errors_per_param('hull_white', '1Yr5YrSwap', 'num_samples_2097152', save=False) # Works fine



# folder = os.path.join(os.getcwd(),"data","hull_white", "dataset-1Yr5YrSwap")
# folder = os.path.join(os.getcwd(),"data","cir", "dataset-1Yr5YrSwap")
# ts = np.load(os.path.join(folder, "monitoring_times.npy"))
# x_train = np.load(os.path.join(folder, "Xtrain.npy"))
# IM_train = np.load(os.path.join(folder, "IMtrain.npy"))
# DIM_val = np.load(os.path.join(folder, "DIMval.npy"))

# plt.figure(figsize=(12,8))
# plt.plot(ts,IM_train[1500:3001].T)
# plt.plot(ts,DIM_val[:,:].T)
# plt.show()