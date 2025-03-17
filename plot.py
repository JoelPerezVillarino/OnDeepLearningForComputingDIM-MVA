import os
import numpy as np
import matplotlib.pyplot as plt

from src.test import convergence_num_train_samples, compute_errors_per_param,\
    compute_errors_per_time, compute_table_errors, portfolio_test


# convergence_num_train_samples('cir', '1Yr5YrSwap', save=True)
# compute_errors_per_param('cir', '1Yr5YrSwap', 'num_samples_4194304', save=True)
# compute_errors_per_param('cir', '1Yr5YrSwap', 'num_samples_1048576', save=True, num_models=4) # Works fine
# compute_errors_per_param('cir', '1Yr5YrSwap', 'num_samples_131072', save=False) # Works fine
# compute_errors_per_param('cir', '1Yr5YrSwap', 'num_samples_65536', save=False) # Works fine

# convergence_num_train_samples('hull_white', '1Yr5YrSwap', save=True)
# compute_errors_per_param('hull_white', '1Yr5YrSwap', 'num_samples_4194304', save=True, num_models=4) 
# compute_errors_per_param('hull_white', '1Yr5YrSwap', 'num_samples_1048576', save=False) # Works fine

# compute_table_errors("hull_white", "1Yr5YrSwap_Extreme", ["1Yr5YrSwap", "num_samples_4194304"])
# compute_table_errors("cir", "1Yr5YrSwap_Extreme", ["1Yr5YrSwap", "num_samples_1048576"])

# portfolio_test("hull_white", "portfolio_swaps",["portfolio_swaps","num_samples_4194304"], idx_scenario=23)
# portfolio_test("cir", "portfolio_swaps",["portfolio_swaps","num_samples_1048576"], idx_scenario=13)


# folder = os.path.join(os.getcwd(),"data","hull_white", "dataset-portfolio_swaps")
# xHW = np.load(os.path.join(folder, "Xval.npy"))
# print(xHW[23])
# print()
# folder = os.path.join(os.getcwd(),"data","cir", "dataset-portfolio_swaps")
# xCIR = np.load(os.path.join(folder, "Xval.npy"))
# print(xCIR[13])

# compute_errors_per_time("hull_white", "1Yr5YrSwap", "num_samples_4194304", save=True, plot=True)
# compute_errors_per_time("cir", "1Yr5YrSwap", "num_samples_1048576", save=True, plot=True)

#HW
# ts_gen_trainset = np.array(
#     [1.277965,1.959296,3.366325,6.071783,11.38792,22.313036,46.345505,91.894902,184.558059,
#     446.336076,918.214170,1847.445999,3808.832578]
# )



folder = os.path.join(os.getcwd(),"data","hull_white", "dataset-1Yr5YrSwap")
# folder = os.path.join(os.getcwd(),"data","cir", "dataset-1Yr5YrSwap")
# folder = os.path.join(os.getcwd(),"data","hull_white", "dataset-portfolio_swaps")
# folder = os.path.join(os.getcwd(),"data","cir", "dataset-portfolio_swaps")
ts = np.load(os.path.join(folder,"monitoring_times.npy"))
var = np.load(os.path.join(folder,"estimated_variance.npy"))
# xtrain = np.load(os.path.join(folder,"Xtrain.npy"))
# ytrain = np.load(os.path.join(folder,"IMtrain.npy"))
# xval = np.load(os.path.join(folder,"Xval.npy"))
# yval = np.load(os.path.join(folder,"DIMval.npy"))
# print(np.where(np.isclose(ts, 1)))

# print(var.shape)

plt.figure(figsize=(12,8))
plt.plot(ts, np.sqrt(var).T/np.sqrt(2**20))
# plt.plot(ts,ytrain[501:2000].T)
# plt.plot(ts,yval.T)
# plt.plot(ts,yval[:,:].T)
plt.show()

print(np.max(np.sqrt(var).T/np.sqrt(2**20)))