import numpy as np


class InitialMargin:

    def __init__(self, tenors:np.ndarray):
        # ISDA parameters
        self.isda_tenors = np.array(
            [15./360., 1./12., 0.25, 0.5, 1., 2., 3., 5., 10., 15., 20., 30.]
        )
        self.risk_weights = np.array(
            [114., 115., 102., 71., 61., 52., 50., 51., 51., 51., 54., 62.]
        )
        self.threshold = 210.e+6
        self.corr_matrix = np.array([
            [0., 0.63, 0.59, 0.47, 0.31, 0.22, 0.18, 0.14, 0.09, 0.06, 0.04, 0.05],
            [0.63, 0., 0.79, 0.67, 0.52, 0.42, 0.37, 0.30, 0.23, 0.18, 0.15, 0.13],
            [0.59, 0.79, 0., 0.84, 0.68, 0.56, 0.50, 0.42, 0.32, 0.26, 0.24, 0.21],
            [0.47, 0.67, 0.84, 0., 0.86, 0.76, 0.69, 0.60, 0.48, 0.42, 0.38, 0.33],
            [0.31, 0.52, 0.68, 0.86, 0., 0.94, 0.89, 0.80, 0.67, 0.60, 0.57, 0.53],
            [0.22, 0.42, 0.56, 0.76, 0.94, 0., 0.98, 0.91, 0.79, 0.73, 0.70, 0.66],
            [0.18, 0.37, 0.50, 0.69, 0.89, 0.98, 0., 0.96, 0.87, 0.81, 0.78, 0.74],
            [0.14, 0.30, 0.42, 0.60, 0.80, 0.91, 0.96, 0., 0.95, 0.91, 0.88, 0.84],
            [0.09, 0.23, 0.32, 0.48, 0.67, 0.79, 0.87, 0.95, 0., 0.98, 0.97, 0.94],
            [0.06, 0.18, 0.26, 0.42, 0.60, 0.73, 0.81, 0.91, 0.98, 0., 0.99, 0.97],
            [0.04, 0.15, 0.24, 0.38, 0.57, 0.70, 0.78, 0.88, 0.97, 0.99, 0., 0.99],
            [0.05, 0.13, 0.21, 0.33, 0.53, 0.66, 0.74, 0.84, 0.94, 0.97, 0.99, 0.]
        ])
        # Matrix change of tenor basis
        self.M = self.allocate_into_isda_tenors(tenors)

    def allocate_into_isda_tenors(self, tenors):
        M = np.zeros((tenors.size-1, self.isda_tenors.size))
        for i in range(1,tenors.size):
            for j in range(self.isda_tenors.size-1):
                if np.isclose(tenors[i], self.isda_tenors[j]):
                    M[i-1, j] = 1.
                    break
                if self.isda_tenors[j]<tenors[i]<self.isda_tenors[j+1]:
                    t = (tenors[i]-self.isda_tenors[j])/(self.isda_tenors[j+1]-self.isda_tenors[j])
                    M[i-1, j] = t
                    M[i-1, j+1] = 1. - t
                    break
            if np.greater_equal(tenors[i], self.isda_tenors[-1]):
                M[i-1, -1] = 1.
                
        return M

    def compute_delta_margin(self, S:np.ndarray):
        S = np.einsum("bi, ij->bj", S, self.M) # Cambio de base
        deltaFactor = np.maximum(1., np.sqrt(np.abs(np.sum(S, axis=1))/self.threshold))
        S *= self.risk_weights
        S *= deltaFactor[:, None]
        return np.sqrt(np.einsum("bi,bi->b", S, S) + np.einsum("bi,ij,bj->b", S, self.corr_matrix, S))

    def compute_initial_margin(self, S:np.ndarray):
        return self.compute_delta_margin(S)



