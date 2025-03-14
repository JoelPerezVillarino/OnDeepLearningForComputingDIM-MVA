import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf

tf.keras.backend.set_floatx("float64")

from src.nn import loadSequentialModel, Normalization


def convergence_num_train_samples(model_label, dataset_name, alpha=0.05, save=True, plot=True):
    folder_path = os.path.join(os.getcwd(), "results", model_label, dataset_name)
    if not os.path.isdir(folder_path):
        raise ValueError("Folder with the results not found.")
    # Load data generated in training stage
    num_samples = np.loadtxt(os.path.join(folder_path,"num_samples.txt"))
    training_time = np.loadtxt(os.path.join(folder_path, "train_time.txt"))
    mse = np.loadtxt(os.path.join(folder_path,"mse.txt"))
    mva = np.load(os.path.join(folder_path,"mean_mva_error.npy"))
    # mva = np.load(os.path.join(folder_path,"mva_error.npy"))
    rmse = np.sqrt(mse)
    oNm1 = 1. / np.sqrt(num_samples) # Expected sqrt convergence order
    # Compute mean per nbr of training samples
    mean_rmse = np.mean(rmse, axis=1)
    mean_mva = np.mean(mva, axis=1)
    # mean_mva = np.mean(mva, axis=(1,2), keepdims=False)
    # Compute deviation (t-distribution)
    t_factor = stats.t(df=rmse.shape[1]).ppf((alpha,1-alpha))
    std_rmse = np.std(rmse,axis=1,ddof=1)
    std_mva = np.std(mva,axis=1,ddof=1)
    # std_mva = np.std(mva,axis=(1,2),ddof=1)
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
        np.savetxt(os.path.join(folder_results, "rmse.dat"),data_rmse)
        np.savetxt(os.path.join(folder_results, "mva.dat"),data_mva)

        # ts_gen_trainset = np.array(
        #     [1.277965,1.959296,3.366325,6.071783,11.38792,22.313036,46.345505,91.894902,184.558059,
        #     446.336076,918.214170,1847.445999,3808.832578]
        # )
        # training_time += ts_gen_trainset
        # data_time = np.concatenate(
        #     [num_samples[:,None], training_time[:,None]], axis=1
        # )
        # np.savetxt(os.path.join(folder_results, "train_time.dat"), data_time, header="num_samples\ttime")
    
    if plot:
        fig = plt.figure(figsize=(12,8))
        ax1 = fig.add_subplot(211)
        ax1.set_title(f"RMSE")
        ax1.set_xlabel("Nbr of samples")
        ax1.set_xscale("log", base=2)
        ax1.set_yscale("log", base=10)
        ax1.errorbar(num_samples, mean_rmse, yerr=bound_rmse, fmt='o', color='blue', capsize=3, label='Mean')
        ax1.plot(num_samples, oNm1_rmse, 'k--', label='O(1/sqrt(K))')

        ax12 = ax1.twinx() # Create a second y-axis sharing the same x-axis
        ax12.plot(num_samples, training_time, color="red")
        ax12.set_ylabel("Training time [s]")
        ax12.set_yscale("log", base=10)
        ax12.tick_params(axis="y", labelcolor="red")

        ax2 = fig.add_subplot(212)
        ax2.set_title(f"MVA")
        ax2.set_xlabel("Nbr of samples")
        ax2.set_xscale("log", base=2)
        ax2.set_yscale("log", base=10)
        ax2.errorbar(num_samples, mean_mva, yerr=bound_mva, fmt='o', color='blue', capsize=3, label='Mean')
        ax2.plot(num_samples, oNm1_mva, 'k--', label='O(1/sqrt(K))')
        # ax2.plot(num_samples, mean_mva, label="mean")
        plt.show()

    return


def compute_errors_per_param(
        model_label, 
        dataset_name, 
        folder_weights,
        num_nn_layers=3, 
        num_nn_units=256,
        idx_time=21,
        eps=1e-12,
        save=False,
        plot=True,
        num_models=None
    ):
    # Define paths to data and models
    data_path = os.path.join(os.getcwd(),"data", model_label, 'dataset-'+dataset_name)
    weights_path = os.path.join(os.getcwd(), "trained_models", model_label, dataset_name, folder_weights)
    if save:
        folder_results = os.path.join(os.getcwd(), "results", model_label, dataset_name)
        if not os.path.isdir(folder_results):
            raise ValueError("Folder with the results not found.")
        folder_results = os.path.join(folder_results, "errors_per_param") 
        os.makedirs(folder_results, exist_ok=True)
        folder_results = os.path.join(folder_results, folder_weights)
        os.makedirs(folder_results, exist_ok=True)

    # Count the nbr of models trained (nbr of weight files)
    if num_models is None:
        num_models = len([f for f in os.listdir(weights_path) if os.path.isfile(os.path.join(weights_path,f))])
    else:
        num_models = num_models

    # Load dataset 
    monitoring_times = np.load(os.path.join(data_path, "monitoring_times.npy"))[:-1]
    x_val = np.load(os.path.join(data_path, "Xval.npy"))
    DIM = np.load(os.path.join(data_path, "DIMval.npy"))[:,:-1]
    MVA = np.load(os.path.join(data_path, "MVA.npy"))
    cumfs = np.load(os.path.join(data_path, "funding_spread_discounts.npy"))
    
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
    # epp_matrix = np.zeros((DIM.shape[0], params_min.size+2)) # Matrix errors per param (last entries DIM and MVA)
    epp_matrix = np.zeros((DIM.shape[0]*num_models, params_min.size+2)) # Matrix errors per param (last entries DIM and MVA)
    DIMs = np.zeros((DIM.shape[0], num_models))
    MVAs = np.zeros((DIM.shape[0], num_models))
    dt = monitoring_times[1] - monitoring_times[0] # Equispaced time grid

    # Allocate x_val values in the matrix
    epp_matrix[:,:params_min.size] = np.repeat(x_val, num_models, axis=0)
    # epp_matrix[:,:params_min.size] = x_val

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
    
    # epp_matrix[:, params_min.size] = DIMs[:,5]
    # epp_matrix[:, params_min.size] = MVAs[:,5]
    # epp_matrix[:, params_min.size] = np.mean(DIMs, axis=1)
    # epp_matrix[:, params_min.size+1] = np.mean(MVAs, axis=1)
    epp_matrix[:, params_min.size] = DIMs.flatten()
    epp_matrix[:, params_min.size+1] = MVAs.flatten()


    # Save results
    if save:
        np.savetxt(os.path.join(folder_results, f"epp_matrix_time_idx_{idx_time}.txt"), epp_matrix)
        np.save(os.path.join(folder_results, f"MVA_errors_per_nn_time_idx_{idx_time}.npy"), MVAs)
        np.save(os.path.join(folder_results, f"DIM_errors_per_nn.npy"), DIMs)
    
    if plot:
        num_params = params_min.size
        fig, axs = plt.subplots(figsize=(16,10), nrows=num_params,ncols=1)
        for i,ax in enumerate(axs):
            ax.scatter(epp_matrix[:, num_params], epp_matrix[:,i])
        plt.show()
        

    return

def compute_errors_per_time(
    model_label, 
    dataset_name, 
    folder_weights,
    num_nn_layers=3, 
    num_nn_units=256,
    alpha=0.05,
    eps=1e-12,
    save=False,
    plot=True
):
    # Compute mean errors per monitoring time
    data_path = os.path.join(os.getcwd(),"data", model_label, 'dataset-'+dataset_name)
    weights_path = os.path.join(os.getcwd(), "trained_models", model_label, dataset_name, folder_weights)
    if save:
        folder_results = os.path.join(os.getcwd(), "results", model_label, dataset_name)
        if not os.path.isdir(folder_results):
            raise ValueError("Folder with the results not found.")
        folder_results = os.path.join(folder_results, "errors_per_time") 
        os.makedirs(folder_results, exist_ok=True)
        folder_results = os.path.join(folder_results, folder_weights)
        os.makedirs(folder_results)

    # Count the nbr of models trained (nbr of weight files)
    num_models = len([f for f in os.listdir(weights_path) if os.path.isfile(os.path.join(weights_path,f))])

    # Load dataset 
    monitoring_times = np.load(os.path.join(data_path, "monitoring_times.npy"))[:-1]
    x_val = np.load(os.path.join(data_path, "Xval.npy"))
    DIM = np.load(os.path.join(data_path, "DIMval.npy"))[:,:-1]
    
    # Data for nn
    params_min = np.load(os.path.join(data_path,"params_min.npy"))
    params_max = np.load(os.path.join(data_path,"params_max.npy"))
    num_outputs = monitoring_times.size

    preprocessing_layer = Normalization()
    preprocessing_layer.load_bounds(params_min, params_max)

    # Load nn architecture
    model = loadSequentialModel(num_nn_units,num_nn_layers,num_outputs,preprocessing_layer=preprocessing_layer)
    model(params_min[:,None]) # Needed for initialize the network (before load the saved weights)

    # Allocate arrays
    mean_difs = np.zeros((num_models, num_outputs,))
    std_difs = np.zeros((num_models, num_outputs,))

    # Evaluate models
    for j in range(num_models):
        path_nn_weights = os.path.join(weights_path, f"model_{j}.h5")
        if os.path.exists(path_nn_weights):
            model.load_weights(path_nn_weights)
        else:
            raise FileNotFoundError(f"File not found: {path_nn_weights}")

        y_pred = model(x_val).numpy()
        # temp = np.abs(DIM - y_pred) / (DIM+eps)
        temp = np.abs((DIM - y_pred))
        mean_difs[j] = np.mean(temp, axis=0)
        std_difs[j] = np.std(temp, axis=0)


    t_factor = stats.t(df=num_models).ppf((alpha,1-alpha))
    means = np.mean(mean_difs, axis=0)
    stds = np.std(std_difs, axis=0, ddof=1)
    bounds = t_factor[-1]*stds/np.sqrt(num_models)

    if save:
        np.save(os.path.join(folder_results, "mean_erros.npy"), means)
        np.save(os.path.join(folder_results, "t_bounds.npy"), bounds)
        table = np.concatenate([monitoring_times[:,None], means[:,None], bounds[:,None]], axis=1)
        np.savetxt(os.path.join(folder_results, "plot_table.txt"), table)
    
    if plot:
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111)
        ax.set_title("Error per monitoring time")
        ax.errorbar(monitoring_times, means, yerr=bounds, fmt='o', capsize=5)
        # ax.set_yscale("log")
        ax.set_xlabel("Time")
        ax.set_ylabel("Absolute error")
        plt.show()

    return


def compute_table_errors(
    model_label, 
    dataset_name, 
    folder_weights,
    num_nn_layers=3, 
    num_nn_units=256,
    idx1=21,
    idx2=42,
    eps=1e-12,
):
    data_path = os.path.join(os.getcwd(),"data", model_label, 'dataset-'+dataset_name)
    data_path_2 = os.path.join(os.getcwd(), "data", model_label, "dataset-1Yr5YrSwap")
    weights_path = os.path.join(os.getcwd(), "trained_models", model_label, folder_weights[0], folder_weights[1])
    folder_results = os.path.join(os.getcwd(), "results", model_label, folder_weights[0])
    if not os.path.isdir(folder_results):
        raise ValueError("Folder with the results not found.")
    folder_results = os.path.join(folder_results, "table_errors_extreme") 
    os.makedirs(folder_results, exist_ok=True)

    # Count the nbr of models trained (nbr of weight files)
    num_models = len([f for f in os.listdir(weights_path) if os.path.isfile(os.path.join(weights_path,f))])
    
    # Load dataset 
    monitoring_times = np.load(os.path.join(data_path, "monitoring_times.npy"))[:-1]
    x_val = np.load(os.path.join(data_path, "Xval.npy"))
    DIM = np.load(os.path.join(data_path, "DIMval.npy"))[:,:-1]
    MVA = np.load(os.path.join(data_path, "MVA.npy"))
    cumfs = np.load(os.path.join(data_path, "funding_spread_discounts.npy"))
    dt = monitoring_times[1] - monitoring_times[0]

    # Data for nn
    params_min = np.load(os.path.join(data_path_2,"params_min.npy"))
    params_max = np.load(os.path.join(data_path_2,"params_max.npy"))
    num_outputs = monitoring_times.size

    preprocessing_layer = Normalization()
    preprocessing_layer.load_bounds(params_min, params_max)

    # Load nn architecture
    model = loadSequentialModel(num_nn_units,num_nn_layers,num_outputs,preprocessing_layer=preprocessing_layer)
    model(params_min[:,None]) # Needed for initialize the network (before load the saved weights)

    # Allocate data
    table = np.zeros((DIM.shape[0], 5))
    if model_label == "hull_white":
        table[:,0] = x_val[:,3]
        table[:,1] = x_val[:,4]
    else:
        table[:,0] = x_val[:,3]
        table[:,1] = x_val[:,5]
    
    eDIM1 = np.zeros((DIM.shape[0], num_models))
    eDIM2 = np.zeros((DIM.shape[0], num_models))
    eMVAs = np.zeros((DIM.shape[0], num_models))

    # Evaluate models
    for j in range(num_models):
        path_nn_weights = os.path.join(weights_path, f"model_{j}.h5")
        if os.path.exists(path_nn_weights):
            model.load_weights(path_nn_weights)
        else:
            raise FileNotFoundError(f"File not found: {path_nn_weights}")

        print(x_val)
        y_pred = model(x_val).numpy()
        # print(y_pred)
        eDIM1[:, j] = np.abs(DIM[:,idx1] - y_pred[:,idx1]) / (DIM[:,idx1]+eps)
        eDIM2[:, j] = np.abs(DIM[:,idx2] - y_pred[:,idx2]) / (DIM[:,idx2]+eps)

        # MVA
        y_pred*=cumfs
        mva_pred = np.sum(y_pred[:,1:],axis=1)*dt
        eMVAs[:,j] = (MVA - mva_pred) / (MVA+eps)
    
    table[:, 2] = np.mean(eDIM1,axis=1)
    table[:, 3] = np.mean(eDIM2,axis=1)
    table[:, 4] = np.mean(eMVAs, axis=1)

    np.savetxt(os.path.join(folder_results, "error_table.txt"), table)
    return
