import os
import json
from copy import deepcopy
# from itertools import repeat
from multiprocessing import Pool

import numpy as np
from pyDOE import lhs
from scipy.interpolate import interp1d

# from src import HullWhite, NelsonSiegel, IRS, InitialMargin
from src import *
from src.utils import timer


class DataGen:
    def __init__(
        self,
        model_label = None,
        dataset_name = None,
        num_MC_paths = None,
        num_samples_train = None,
        num_samples_val = None,
        num_processes = None,
        irs_params = None,
        params_min = None,
        params_max = None,
        time_step = None
    ):
        # Prefixed values simulation
        self.t0 = 0. # Initial simul time
        self.tenors = np.array(
            [0., 15./360., 1./12., 0.25, 0.5, 1., 2., 3., 5., 10., 15., 20., 30.]
        ) # Tenors term ir structure
        self.im_engine = InitialMargin(self.tenors)
        self.BP = 1.e-4 # Definition of basis point
        self.LAMB = 1.37 # lambda NelsonSiegel

        # Fixed funding spread parameters (between counterparties B and C)
        # Following 'MVA: Initial Margin Valuation Adjustment by Replication and Regression'
        # by Andrew Green and Chris Kenyon, 2014
        self.lB = 164*self.BP # Default intensity B
        self.lC = 0 # Default intensity C
        self.rB = 0.4 # Recovery rate B
        self.sI = 0. # Spread on IM

        # --------------------------- Input values ----------------------------------
        # Model values
        self.dataset_name = dataset_name
        self.model_label = model_label
        self.params_min = params_min 
        self.params_max = params_max

        # Product data
        self.irs_params = irs_params

        # Computational parameters
        self.time_step = time_step
        self.num_MC_paths = num_MC_paths # Only for gen_val_set
        self.num_samples_train = num_samples_train
        self.num_samples_val = num_samples_val
        self.num_processes = num_processes # Only for gen_val_set

        if (self.irs_params is None or self.params_min is None or self.params_max is None or
            self.time_step is None or self.num_MC_paths is None or self.num_samples_train is None
            or self.num_samples_val is None or self.model_label is None or self.num_processes is None):
            raise ValueError("Class not properly initialized!")

        self.num_params = len(self.params_min)
        if (len(self.params_max) != self.num_params):
            raise ValueError("Number of param_min and param_max do not match!")

        self.pmin = np.array(
            [v for v in self.params_min.values()]
        )
        self.pmax = np.array(
            [v for v in self.params_max.values()]
        )

        if all(isinstance(value, dict) for value in self.irs_params.values()):
            self.num_swaps = len(self.irs_params)
            self.portfolio = [IRS(**self.irs_params[key]) for key in self.irs_params]
            self.tend = max(swap["maturity"] for swap in self.irs_params.values())
        else:
            self.num_swaps = 1
            self.portfolio = [IRS(**self.irs_params)]
            self.tend = self.irs_params["maturity"]
        # Handle time grid
        self.num_monitoring_times = int((self.tend-self.t0)/self.time_step)+1
        self.monitoring_times = np.linspace(self.t0, self.tend, self.num_monitoring_times)

        # Manage folder for saving data
        current_dir = os.getcwd()
        self.data_dir = os.path.join(current_dir, "data")
        os.makedirs(self.data_dir, exist_ok=True)
        self.data_dir = os.path.join(self.data_dir, self.model_label, "dataset"+"-"+self.dataset_name)
        os.makedirs(self.data_dir, exist_ok=True)

        # Save params min and params max
        np.save(os.path.join(self.data_dir, "params_min.npy"),self.pmin)
        np.save(os.path.join(self.data_dir, "params_max.npy"),self.pmax)
        # Save monitoring times
        np.save(os.path.join(self.data_dir, "monitoring_times.npy"), self.monitoring_times)
        # Generate and save funding spread values at monitoring times
        self.cumfs = self.generate_funding_spread_values()
        np.save(os.path.join(self.data_dir, "funding_spread_discounts.npy"), self.cumfs)

    @classmethod
    def from_json(cls, json_file):
        with open(json_file, "r") as file:
            data = json.load(file)
        return cls(**data)
    
    def build_discount_curve(self, ys:np.ndarray)->callable:
        return interp1d(self.tenors, np.exp(-self.tenors*ys),copy=False)
    
    def eval_portfolio(self, portfolio:list, d_curve:callable, t:float):
        if len(portfolio)>1:
            return np.sum([swap(d_curve,t) for swap in portfolio],axis=0)
        return portfolio[0](d_curve,t)
    
    def compute_portfolio_sensitivities(self, portfolio:list, t:float, ys:np.ndarray, V:np.ndarray, S:np.ndarray):
        for k in range(1, self.tenors.size):
            ys[:,k]+=self.BP
            P = self.build_discount_curve(ys)
            S[:,k-1] = self.eval_portfolio(portfolio,P,t)-V
            ys[:,k]-=self.BP
        return None
    
    def check_swap_maturities(self, portfolio:list, t:float):
        idx = 0
        while idx<len(portfolio):
            if np.isclose(portfolio[idx].getMaturity(),t):
                portfolio.pop(idx)
            else:
                idx+=1
        return None
    
    def clear_swap_status(self, portfolio:list):
        for swap in portfolio: swap.clearStatus()

    def generate_lhs_samples(self, num_samples):
        return self.pmin + lhs(self.pmin.size, samples=num_samples)*(self.pmax-self.pmin)

    def generate_funding_spread_values(self):
        times = self.monitoring_times[:-1]
        dts = times[1:]-times[:-1]
        cte = (1.-self.rB)*self.lB
        funding_spread_fun = lambda tau: np.exp(-(self.lB+self.lC)*tau)
        cumfs = cte*np.cumprod(funding_spread_fun(dts))
        cumfs = np.concatenate([cte*np.ones((1,1)),cumfs[:,None]]).squeeze()
        return cumfs

    def generate_DIM_path(self, x:np.ndarray ,portfolio:list ,rng:np.random._generator.Generator):
        # Compute DIM path for a given a market state x
        ones = np.ones((self.num_MC_paths,))
        if self.model_label == "hull_white":
            C = NelsonSiegel(b0=x[0]*ones,b1=x[1]*ones,b2=x[2]*ones,lamb=self.LAMB*ones)
            ir_model = HullWhite(a=x[3]*ones, vol=x[4]*ones, Y=C)
        elif self.model_label == "cir":
            C = NelsonSiegel(b0=x[0]*ones,b1=x[1]*ones,b2=x[2]*ones,lamb=self.LAMB*ones)
            ir_model = CIR(kappa=x[3]*ones,theta=x[4]*ones,vol=x[5]*ones,x0=x[6]*ones,Y=C)
        # Initialize vars
        r = np.zeros((self.num_MC_paths,)) 
        V = np.zeros((self.num_MC_paths,))
        discount = np.ones((self.num_MC_paths,))
        S = np.zeros((self.num_MC_paths,self.tenors.size-1))
        DIM = np.zeros((self.num_monitoring_times,))

        # Time t0
        ys = ir_model.computeYieldPoints(self.monitoring_times[0],ir_model.x0,self.tenors)
        P = self.build_discount_curve(ys)
        if self.num_swaps == 1: # One swap case, we work with no ATM swap
            portfolio[0].setRateATM(P,x[-1]*ones)
        else: # ATM swap
            for swap in portfolio: swap.setRateATM(P)

        V = self.eval_portfolio(portfolio,P,self.monitoring_times[0])
        self.compute_portfolio_sensitivities(portfolio,self.monitoring_times[0],ys,V,S)
        DIM[0] = np.mean(self.im_engine.compute_initial_margin(S))
        # Forward times
        r = ir_model.x0
        for n in range(1,self.num_monitoring_times-1):
            r = ir_model.shortRateSimulStep(self.monitoring_times[n-1],self.monitoring_times[n],r,rng)
            discount *= np.exp(-ir_model.fromXtoR(self.monitoring_times[n],r)\
                            *(self.monitoring_times[n]-self.monitoring_times[n-1]))
            ys = ir_model.computeYieldPoints(self.monitoring_times[n],r,self.tenors)
            P = self.build_discount_curve(ys)
            for swap in portfolio: swap.checkStatus(P, self.monitoring_times[n])
            V = self.eval_portfolio(portfolio,P,self.monitoring_times[n])
            self.compute_portfolio_sensitivities(portfolio,self.monitoring_times[n],ys,V,S)
            DIM[n] = np.mean(self.im_engine.compute_initial_margin(S)*discount)
            self.check_swap_maturities(portfolio,self.monitoring_times[n])
        self.clear_swap_status(portfolio)
        return DIM
    
    def generate_variance_path(
        self, 
        x:np.ndarray, 
        y:np.ndarray,
        portfolio:list, 
        rng:np.random._generator.Generator
    ):
        # Estimate variance of one sample (per monitoring time)
        ones = np.ones((self.num_MC_paths,))
        if self.model_label == "hull_white":
            C = NelsonSiegel(b0=x[0]*ones,b1=x[1]*ones,b2=x[2]*ones,lamb=self.LAMB*ones)
            ir_model = HullWhite(a=x[3]*ones, vol=x[4]*ones, Y=C)
        elif self.model_label == "cir":
            C = NelsonSiegel(b0=x[0]*ones,b1=x[1]*ones,b2=x[2]*ones,lamb=self.LAMB*ones)
            ir_model = CIR(kappa=x[3]*ones,theta=x[4]*ones,vol=x[5]*ones,x0=x[6]*ones,Y=C)
        # Initialize vars
        r = np.zeros((self.num_MC_paths,)) 
        V = np.zeros((self.num_MC_paths,))
        discount = np.ones((self.num_MC_paths,))
        S = np.zeros((self.num_MC_paths,self.tenors.size-1))
        VAR = np.zeros((self.num_monitoring_times,))

        # Time t0
        ys = ir_model.computeYieldPoints(self.monitoring_times[0],ir_model.x0,self.tenors)
        P = self.build_discount_curve(ys)
        if self.num_swaps == 1: # One swap case, we work with no ATM swap
            portfolio[0].setRateATM(P,x[-1]*ones)
        else: # ATM swap
            for swap in portfolio: swap.setRateATM(P)

        V = self.eval_portfolio(portfolio,P,self.monitoring_times[0])
        self.compute_portfolio_sensitivities(portfolio,self.monitoring_times[0],ys,V,S)
        IM = self.im_engine.compute_initial_margin(S)
        VAR[0] = np.sum(np.square(IM - y[0])) / (self.num_MC_paths-1)
        # Forward times
        r = ir_model.x0
        for n in range(1,self.num_monitoring_times-1):
            r = ir_model.shortRateSimulStep(self.monitoring_times[n-1],self.monitoring_times[n],r,rng)
            discount *= np.exp(-ir_model.fromXtoR(self.monitoring_times[n],r)\
                            *(self.monitoring_times[n]-self.monitoring_times[n-1]))
            ys = ir_model.computeYieldPoints(self.monitoring_times[n],r,self.tenors)
            P = self.build_discount_curve(ys)
            for swap in portfolio: swap.checkStatus(P, self.monitoring_times[n])
            V = self.eval_portfolio(portfolio,P,self.monitoring_times[n])
            self.compute_portfolio_sensitivities(portfolio,self.monitoring_times[n],ys,V,S)
            IM = self.im_engine.compute_initial_margin(S)*discount
            VAR[n] = np.sum(np.square(IM - y[n])) / (self.num_MC_paths-1)
            self.check_swap_maturities(portfolio,self.monitoring_times[n])
        self.clear_swap_status(portfolio)

        return VAR

    @timer
    def gen_train_set(self):
        print(f"Starting generation of the training set...")
        X = self.generate_lhs_samples(self.num_samples_train)
        # Load term struct and ir model
        if self.model_label == "hull_white":
            C = NelsonSiegel(b0=X[:,0], b1=X[:,1], b2=X[:,2], lamb=self.LAMB*np.ones((self.num_samples_train,)))
            ir_model = HullWhite(a=X[:,3], vol=X[:,4], Y=C)
        elif self.model_label == "cir":
            C = NelsonSiegel(b0=X[:,0], b1=X[:,1], b2=X[:,2], lamb=self.LAMB*np.ones((self.num_samples_train,)))
            ir_model = CIR(kappa=X[:,3],theta=X[:,4],vol=X[:,5],x0=X[:,6],Y=C)

        # Do a deepcopy of the portfolio to avoid problems (work with copy of portfolio)
        portfolio = deepcopy(self.portfolio)
        # Initialize vars
        r = np.zeros((self.num_samples_train,)) # Array short rate
        V = np.zeros((self.num_samples_train,)) # Array portfolio price
        discount = np.ones((self.num_samples_train,))
        S = np.zeros((self.num_samples_train, self.tenors.size-1)) # Array of portfolio sensitivities
        IM = np.zeros((self.num_samples_train, self.num_monitoring_times)) # Array IM values
        # Initialize rng
        rng = np.random.default_rng()

        # Time t0
        ys = ir_model.computeYieldPoints(self.monitoring_times[0], ir_model.x0, self.tenors)
        P = self.build_discount_curve(ys)
        if self.num_swaps==1: # One swap case: we work with different fixed rates (based on fair swap rate)
            # self.portfolio[0].setRateATM(P,X[:,-1]) 
            portfolio[0].setRateATM(P,X[:,-1]) 
        else: # Portfolio swap ATM
            # for swap in self.portfolio: swap.setRateATM(P)
            for swap in portfolio: swap.setRateATM(P)

        # V = self.eval_portfolio(self.portfolio,P, self.monitoring_times[0])
        V = self.eval_portfolio(portfolio,P,self.monitoring_times[0])
        # self.compute_portfolio_sensitivities(self.portfolio,self.monitoring_times[0],ys,V,S)
        self.compute_portfolio_sensitivities(portfolio,self.monitoring_times[0],ys,V,S)
        IM[:,0] = self.im_engine.compute_initial_margin(S)
        # Forward times
        r = ir_model.x0
        for n in range(1,self.num_monitoring_times-1):
            r = ir_model.shortRateSimulStep(
                self.monitoring_times[n-1],
                self.monitoring_times[n],
                r,
                rng
            )
            discount *= np.exp(-ir_model.fromXtoR(self.monitoring_times[n],r)*\
                            (self.monitoring_times[n]-self.monitoring_times[n-1]))
            ys = ir_model.computeYieldPoints(self.monitoring_times[n],r,self.tenors)
            P = self.build_discount_curve(ys)
            # for swap in self.portfolio: swap.checkStatus(P,self.monitoring_times[n]) # Manage last coupon if needed
            for swap in portfolio: swap.checkStatus(P,self.monitoring_times[n]) # Manage last coupon if needed
            # V = self.eval_portfolio(self.portfolio,P,self.monitoring_times[n])
            V = self.eval_portfolio(portfolio,P,self.monitoring_times[n])
            # self.compute_portfolio_sensitivities(self.portfolio,self.monitoring_times[n],ys,V,S)
            self.compute_portfolio_sensitivities(portfolio,self.monitoring_times[n],ys,V,S)
            IM[:,n] = self.im_engine.compute_initial_margin(S)*discount
            # Check if some swap matures and delete if it is the case
            # self.check_swap_maturities(self.portfolio,self.monitoring_times[n])
            self.check_swap_maturities(portfolio,self.monitoring_times[n])
        
        # Save generated data
        np.save(os.path.join(self.data_dir, "Xtrain.npy"), X)
        np.save(os.path.join(self.data_dir, "IMtrain.npy"), IM)
        # self.clear_swap_status(self.portfolio)
        self.clear_swap_status(self.portfolio) # Is not needed since i did a copy of portfolio
        print(f"Done!")
        return
    
    @timer
    def gen_val_set(self):
        print(f"Starting generation of validation set...")
        X = self.generate_lhs_samples(self.num_samples_val)
        rng = np.random.default_rng()
        child_rngs = rng.spawn(self.num_samples_val) # Create child rng for parallel computation
        # duplicate portfolios to avoid issues with multiprocessing
        duplicated_portfolios = [deepcopy(self.portfolio) for _ in range(self.num_samples_val)]

        with Pool(processes=self.num_processes) as p: # Multiprocessing DIM computation
            DIM = p.starmap(
                self.generate_DIM_path,
                zip(X, duplicated_portfolios, child_rngs)
            )
            # DIM = p.starmap(DataGen.worker_function, zip(repeat(self),X, duplicated_portfolios, child_rngs))
        DIM = np.array(DIM)
        # MVA computation with funding spread
        fdDIM = DIM[:,1:] * self.cumfs 
        MVA = np.sum(fdDIM,axis=1)*self.time_step
        # Save generated data
        np.save(os.path.join(self.data_dir, "Xval.npy"),X)
        np.save(os.path.join(self.data_dir, "DIMval.npy"),DIM)
        np.save(os.path.join(self.data_dir, "MVA.npy"),MVA)
        self.clear_swap_status(self.portfolio)
        print(f"Done!")
        return
    
    def gen_val_set_adhoc(self, a:list=[0.01,0.025,0.05], sigma:list=[0.005,0.0075,0.015]):
        print("Generating adhoc DIM paths...")
        a = np.array(a)
        sigma = np.array(sigma)
        self.num_samples_val = a.size*sigma.size
        X = self.generate_lhs_samples(self.num_samples_val)
        # Change the values of a sigma for our a sigma combinations!
        A, S = np.meshgrid(a, sigma, indexing="ij")
        temp = np.concatenate([np.ravel(A)[:,None], np.ravel(S)[:,None]], axis=1)
        if self.model_label == "hull_white":
            X[:,3:5] = temp
        else:
            X[:,4:6] = temp

        rng = np.random.default_rng()
        if self.num_samples_val<self.num_processes:
            child_rngs = rng.spawn(self.num_samples_val) 
        else:
            child_rngs = rng.spawn(self.num_processes) 

        duplicated_portfolios = [deepcopy(self.portfolio) for _ in range(self.num_samples_val)]
        with Pool(processes=self.num_processes) as p: # Multiprocessing DIM computation
            DIM = p.starmap(
                self.generate_DIM_path,
                zip(X, duplicated_portfolios, child_rngs)
            )
        # DIM = list()
        # for j in range(self.num_samples_val):
        #     DIM.append(self.generate_DIM_path(X[j], duplicated_portfolios[j], rng))
        DIM = np.array(DIM)
        fdDIM = DIM[:,1:] * self.cumfs 
        MVA = np.sum(fdDIM,axis=1)*self.time_step
        np.save(os.path.join(self.data_dir, "Xval.npy"),X)
        np.save(os.path.join(self.data_dir, "DIMval.npy"),DIM)
        np.save(os.path.join(self.data_dir, "MVA.npy"),MVA)
        self.clear_swap_status(self.portfolio)
        print(f"Done!")
        return
    
    @timer
    def val_set_variance(self):
        # Estimate variance of the dataset taking as a reference the val-set values 
        print(f"Computing estimated variance....")
        path_xval = os.path.join(self.data_dir,"Xval.npy")
        path_yval = os.path.join(self.data_dir,"DIMval.npy")
        if not os.path.isfile(path_xval) or not os.path.isfile(path_yval):
            print("xVal and DIMval data not found")
            return
        # Load data
        X = np.load(path_xval)
        Y = np.load(path_yval)
        rng = np.random.default_rng()
        child_rngs = rng.spawn(self.num_samples_val) # Create child rng for parallel computation
        duplicated_portfolios = [deepcopy(self.portfolio) for _ in range(self.num_samples_val)]
        # Multiprocessing for variance computation
        with Pool(processes=self.num_processes) as p:
            VAR = p.starmap(
                self.generate_variance_path,
                zip(X, Y, duplicated_portfolios, child_rngs)
            )
        VAR = np.array(VAR)
        np.save(os.path.join(self.data_dir, "estimated_variance.npy"),VAR)
        print(f"Done!")
        return


