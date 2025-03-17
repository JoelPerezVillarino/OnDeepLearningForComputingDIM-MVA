# OnDeepLearningForComputingDIM-MVA
This repository contains all the code needed for reproducing the numerical experiments presented in the paper `On Deep Learning For Computing the Dynamic Initial Margin and Margin Value Adjustment'.

The work addresses the challenge of training neural networks for Dynamic Initial Margin (DIM) computation in counterparty credit risk, a task traditionally burdened by the high costs associated with generating training datasets through nested Monte Carlo (MC) simulations.
By condensing the initial market state variables into an input vector, determined through an interest rate model and a parsimonious parameterization of the current interest rate term structure, we construct a training dataset where the labels are future realizations, generated with a single MC path, of the Initial Margin (IM) variable. Since DIM is defined as the conditional expectation of IM, the latter can be understood as noisy and unbiased samples of DIM, allowing the application of deep learning regression techniques to its computation. 
To this end, a multi-output neural network structure is employed to handle DIM as a time-dependent function, facilitating training across a mesh of monitoring times. This methodology offers significant advantages: it reduces the dataset generation cost to a single MC execution and parameterizes the neural network by initial market state variables, obviating the need for repeated training. Experimental results demonstrate the approach's convergence properties and robustness across different interest rate models (Hull-White and \joel{Cox–Ingersoll–Ross}) and portfolio complexities, validating its general applicability and efficiency in more realistic scenarios.