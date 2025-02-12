import numpy as np


class TermStruct:

    def __init__(self) -> None:
        pass

    def yields(self, t):
        raise NotImplemented()
    
    def discounts(self, t):
        return np.exp(-self.yields(t)*t)
    
    def forward(self, t, eps=1.e-5):
        return -(np.log(self.discounts(t+eps))-np.log(self.discounts(t)))/eps


class ConstantTermStruct(TermStruct):

    def __init__(self, y0:np.ndarray):
        self.y0 = y0
        super().__init__()
    
    def yields(self, t):
        return self.y0



class NelsonSiegel(TermStruct):
    """Nelson Siegel parametrization of the yield curve
    Params:
        b0: long-term level of interest rates.
        b1: slope.
        b2: curvature of the yield curve.
        lamb: decay factor. Small vals result in slow decay (better fit at long),
            while large values produce fast decay (better fit at short maturities).
    """
    def __init__(self, b0:np.ndarray, b1:np.ndarray, b2:np.ndarray, lamb:np.ndarray) -> None:
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2
        self.lamb = lamb
        super().__init__()
    
    def yields(self, t):
        temp = -self.lamb * np.expm1(-t/self.lamb) / np.maximum(t, 1.e-14)
        return self.b0+self.b1*temp+self.b2*(temp-np.exp(-t/self.lamb))