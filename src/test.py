import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def convergence_num_train_samples(model_label, dataset_name, alpha=0.05, save=True):
    folder_path = os.path.join(os.getcwd(), "results", model_label, dataset_name)
    if not os.path.isdir(folder_path):
        raise ValueError("Folder with the results not found.")
    # Load data generated in training stage
    num_samples = np.loadtxt(os.path.join(folder_path,"num_samples.txt"))
    mse = np.loadtxt(os.path.join(folder_path,"mse.txt"))
    mva = np.loadtxt(os.path.join(folder_path,"mean_mva_error.txt"))
    rmse = np.sqrt(mse)
    oNm1 = 1. / np.sqrt(num_samples) # Expected sqrt convergence order
    # Compute mean per nbr of training samples
    mean_rmse = np.mean(rmse, axis=1)
    mean_mva = np.mean(mva, axis=1)
    # Compute deviation (t-distribution)
    t_factor = stats.t(df=rmse.shape[1]).ppf((alpha,1-alpha))
    std_rmse = np.std(rmse,axis=1,ddof=1)
    std_mva = np.std(mva,axis=1,ddof=1)
    # Bounds
    bound_rmse = t_factor[-1]*std_rmse/np.sqrt(rmse.shape[1])
    bound_mva = t_factor[-1]*std_mva/np.sqrt(mva.shape[1])
    # Adjust oNm1 for plotting purposes
    oNm1 *= 1/oNm1[0]
    oNm1_rmse = oNm1*mean_rmse[0]
    oNm1_mva = oNm1*mean_mva[0]

    if save:
        folder_results = os.path.join(folder_path,"convergence_train_num_samples")
        os.makedirs(folder_results)
        data_rmse = np.concatenate(
            (num_samples[:,None],mean_rmse[:,None],bound_rmse[:,None],oNm1_rmse[:,None]),axis=1
        )     
        data_mva = np.concatenate(
            (num_samples[:,None],mean_mva[:,None],bound_mva[:,None],oNm1_mva[:,None]),axis=1
        )     
        np.savetxt(os.path.join(folder_results, "rmse.dat"),data_rmse,header="num_samples\tmean\tbound\tonm1")
        np.savetxt(os.path.join(folder_results, "mva.dat"),data_mva,header="num_samples\tmean\tbound\tonm1")
    
    # Add plot!

    return