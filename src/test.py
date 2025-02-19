import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf

from src.nn import loadSequentialModel, Normalization


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


def compute_errors_per_param(
        model_label, 
        dataset_name, 
        folder_weights,
        num_nn_layers=3, 
        num_nn_units=256,
        idx_time=35,
        eps=1e-12
    ):
    # Define paths to data and models
    data_path = os.path.join(os.getcwd(),"data", model_label, dataset_name)
    weights_path = os.path.join(os.getcwd(), "trained_models", model_label, dataset_name, folder_weights)
    folder_results = os.path.join(os.getcwd(), "results", model_label, dataset_name)
    if not os.path.isdir(folder_results):
        raise ValueError("Folder with the results not found.")
    folder_results = os.path.join(folder_results, "errors_per_param") 
    os.makedirs(folder_results, exist_ok=True)
    folder_results = os.path.join(folder_results, folder_weights)
    os.makedirs(folder_results)

    # Count the nbr of models trained (nbr of weight files)
    num_models = len([f for f in os.listdir(weights_path) if os.path.isfile(os.path.join(weights_path,f))])

    # Load dataset 
    monitoring_times = np.load(os.path.join(data_path, "monitoring_times.npy"))[:-1]
    x_val = np.load(os.path.join(data_path, "Xtrain.npy"))
    DIM = np.load(os.path.join(data_path, "DIMval.npy"))[:,:-1]
    MVA = np.load(os.path.join(data_path, "MVA.npy"))
    cumfs = np.load(os.path.join(data_path, "fundind_spread_discounts.npy"))
    
    # Data for nn
    params_min = np.load(os.path.join(data_path,"params_min.npy"))
    params_max = np.load(os.path.join(data_path,"params_max.npy"))
    num_outputs = monitoring_times.size

    preprocessing_layer = Normalization()
    preprocessing_layer.load_bounds(params_min, params_max)

    # Load nn architecture
    model = loadSequentialModel(num_nn_units,num_nn_layers,num_outputs,preprocessing_layer=preprocessing_layer)
    model(params_min[:,None]) # Needed for initialize the network (before load the saved weights)

    # Allocate vars
    epp_matrix = np.zeros((DIM.shape[0], params_min.size+2)) # Matrix errors per param (last entries DIM and MVA)
    DIMs = np.zeros((DIM.shape[0], num_models))
    MVAs = np.zeros((DIM.shape[0], num_models))
    dt = monitoring_times[1] - monitoring_times[0] # Equispaced time grid

    # Allocate x_val values in the matrix
    epp_matrix[:,:params_min.size] = x_val

    # Evaluate models and compute relative differences
    for j in range(num_models):
        path_nn_weights = os.path.join(weights_path, f"model_{j}.h5")
        if os.path.exists(path_nn_weights):
            model.load_weights(path_nn_weights)
        else:
            raise FileNotFoundError(f"File not found: {path_nn_weights}")
        
        # Compute DIM error in monitoring_time[idx_time]
        y_pred = model(x_val).numpy()
        DIMs[:,j] = (DIM[:,idx_time] - y_pred[:,idx_time]) / (DIM[:,idx_time]+eps)
        # Compute MVA error 
        y_pred*=cumfs
        mva_pred = np.sum(y_pred[:,1:],axis=1)*dt
        MVAs[:,j] = (MVA - mva_pred) / (MVA+eps)
    
    epp_matrix[:, params_min.size] = np.mean(DIMs, axis=1)
    epp_matrix[:, params_min.size+1] = np.mean(MVAs, axis=1)

    # Save results
    np.savetxt(os.path.join(folder_results, f"epp_matrix.txt"), epp_matrix)
    np.save(os.path.join(folder_results, f"MVA_errors_per_nn.npy"), MVAs)
    np.save(os.path.join(folder_results, f"DIM_errors_per_nn_time_idx_{idx_time}.npy"), DIMs)

    return