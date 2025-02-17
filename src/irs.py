import numpy as np


class IRS:

    def __init__(
        self, 
        start_time:float,
        maturity:float,
        fixed_tenor:float,
        float_tenor:float,
        K:np.ndarray=None,
        is_payer:bool=True,
        notional:float=100.,
    ) -> None:
        self.startTime = start_time
        self.T = maturity
        self.fixedTenor = fixed_tenor 
        self.floatTenor = float_tenor
        self.K = None if K is None else K
        self.isPayer = True if is_payer == 1 else False
        self.alpha = 1. if self.isPayer else -1.
        self.N = notional

        self.lastCoupon = None

        length = self.T - self.startTime
        numFixedPayments = int(length / self.fixedTenor)+1
        numFloatDates = int(length / self.floatTenor)+1
        self.fixedLegTimes = np.linspace(self.startTime, self.T, numFixedPayments)[1:]
        self.floatLegTimes = np.linspace(self.startTime, self.T, numFloatDates)
    
    def __call__(self, P:callable, t:float)->np.ndarray:
        # Payer leg
        idx = np.where(np.greater_equal(self.fixedLegTimes, t))[0]
        payerTimes = self.fixedLegTimes[idx]
        temp1 = np.sum(np.array([P(tau - t) for tau in payerTimes]), axis=0)
        temp1 *= self.fixedTenor * self.K
        # Receiver leg
        if self.lastCoupon is None:
            idx = np.where(np.greater_equal(self.floatLegTimes, t))[0]
            receiverTimes = self.floatLegTimes[idx]
            temp2 = P(receiverTimes[0]-t) - P(receiverTimes[-1]-t)
        else:
            # No se si el floatTenor es necesario....
            temp2 = self.lastCoupon * P(self.floatLegTimes[-1]-t) * self.floatTenor
        return self.N * self.alpha * (temp2 - temp1)

    def getRateATM(self, P:callable, t:float=0.)->np.ndarray:
        A = np.sum(np.array([P(tau-t) for tau in self.fixedLegTimes]), axis=0)
        A *= self.fixedTenor
        temp = P(self.floatLegTimes[0]-t)-P(self.floatLegTimes[-1]-t)
        return temp / A

    def setRateATM(self, P:callable, spread=0.):
        self.K = self.getRateATM(P) + spread

    def checkStatus(self, P:callable, t:float)->None:
        if np.isclose(t, self.floatLegTimes[-2]):
            self.lastCoupon = (1. / P(self.floatLegTimes[-1]-t) -1.) / self.floatTenor 
    
    def clearStatus(self):
        self.lastCoupon = None
        self.K = None
    
    def getMaturity(self):
        return self.T

