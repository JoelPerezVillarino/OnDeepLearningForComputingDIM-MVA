import numpy as np
from scipy.interpolate import interp1d



def buildP(tenors:np.ndarray, ys:np.ndarray)->callable:
    return interp1d(tenors, np.exp(-tenors*ys), copy=False)

def computeSensitivities(
    t:float,
    tenors:np.ndarray,
    ys:np.ndarray,
    V:np.ndarray,
    pricer:callable,
    S:np.ndarray,
    BP:float=1.e-4
)->None:
    for k in range(1, tenors.size):
        ys[:, k] += BP
        P = buildP(tenors, ys)
        S[:, k-1] = pricer(P, t) - V
        ys[:, k] -= BP
    return None
