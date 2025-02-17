import numpy as np
from .parametric_curves import TermStruct


# Create a base clase for computing yield points and zcb
# Write the general form P(t,T)=A(t,T)*exp(-B*(t,T)r_t)


class AffineInterestRateModel:

    def __init__(self):
        pass

    def shortRateSimulStep(self, s:float, t:float, r0:np.ndarray, rng:np.random._generator.Generator):
        raise NotImplementedError("Subclass implementation!")

    def A(self, t:float, T:float):
        raise NotImplementedError("Subclass implementation!")

    def B(self, t:float, T:float):
        raise NotImplementedError("Subclass implementation!")

    def zcb(self, t:float, T:float, r:np.ndarray):
        if np.isclose(t, T): return np.ones_like(r)
        return self.A(t,T)*np.exp(-self.B(t,T)*r)
    
    def computeYieldPoints(self, t:float, r:np.ndarray, tenors:np.ndarray):
        ys = np.array([-np.log(self.zcb(t, t+tenor, r))/tenor for tenor in tenors[1:]]).T
        return np.concatenate([np.zeros((ys.shape[0], 1)), ys], axis=1)    


class VasicekModel(AffineInterestRateModel):

    def __init__(self, theta:np.ndarray, kappa:np.ndarray, vol:np.ndarray) -> None:
        self.theta = theta
        self.kappa = kappa
        self.vol = vol

        super().__init__()

    def shortRateSimulStep(self, s:float, t:float, r0:np.ndarray, rng:np.random._generator.Generator)->np.ndarray:
        return r0 * np.exp(-self.kappa*(t-s))-self.theta*np.expm1(-self.kappa*(t-s)) +\
            np.sqrt(-0.5*self.vol**2/self.kappa*np.expm1(-2.*self.kappa*(t-s)))*rng.standard_normal(size=r0.size)
    
    def B(self, t:float, T:float):
        return -np.expm1(-self.kappa*(T-t)) / self.kappa
    
    def A(self, t:float, T:float):
        temp = (self.theta-0.5*np.power(self.vol/self.kappa, 2))*(self.B(t,T) - T + t)-\
            0.25*np.power(self.vol*self.B(t,T),2)/self.kappa
        return np.exp(temp)


class HullWhite(AffineInterestRateModel):

    def __init__(self, a:np.ndarray, vol:np.ndarray, Y:TermStruct) -> None:
        self.a = a
        self.vol = vol
        self.Y = Y
        self.x0 = np.zeros_like(self.vol) + self.alpha(0.) # Here x0 = r0

        super().__init__()
    
    def alpha(self, t:float):
        f = self.Y.forward(t)
        return f+ 0.5*np.power(self.vol/self.a*np.expm1(-self.a*t), 2)
    
    def shortRateSimulStep(self, s:float, t:float, r0:np.ndarray, rng:np.random._generator.Generator)->np.ndarray:
        cmean = (r0-self.alpha(s))*np.exp(-self.a*(t-s)) + self.alpha(t) 
        cvar = -0.5*self.vol*self.vol*np.expm1(-2.*self.a*(t-s))/self.a
        return cmean + np.sqrt(cvar)*rng.standard_normal(size=r0.size)
    
    def B(self, t:float, T:float):
        return -np.expm1(-self.a*(T-t))/self.a
    
    def A(self, t:float, T:float):
        temp = self.B(t,T)*self.Y.forward(0,t)+0.25*np.power(self.vol*self.B(t,T),2)*np.expm1(-2.*self.a*t)/self.a
        return self.Y.discounts(T)/self.Y.discounts(t)*np.exp(temp)


class CIR(AffineInterestRateModel):
    # Refers to CIR++ model explained in Brigo, Mercurio.
    # Remember, for the valuation of zcb in CIR++, i only need to provide the x_t st
    # r_t = x_t + phi_t
    def __init__(self, kappa:np.ndarray, theta:np.ndarray, vol:np.ndarray, x0:np.ndarray, Y:TermStruct):
        self.theta = theta
        self.kappa = kappa
        self.vol = vol
        self.x0 = x0
        self.Y = Y

        self.h = np.sqrt(np.power(self.kappa,2)+2.*np.power(self.vol,2))
        self.d = 4.*self.theta*self.kappa/(self.vol*self.vol)
        # idx where we will apply simul strategy 1 (Glasserman)
        self.idx1 = np.where(self.d>1)[0]
        # idx where we will apply simul strategy 2 (Glasserman) 
        self.idx2 = np.where(self.d<=1)[0]

        super().__init__()


    def fellerCondition(self):
        print(f"Number of cases where the Feller condition does not hold: {np.sum(2*self.theta*self.kappa<=self.vol**2)} of {self.theta.size}") 
    
    def shortRateSimulStep(self, s:float, t:float, r0:np.ndarray, rng:np.random._generator.Generator):
        # I have to add the complete version CIR++, until now this is only CIR simulation
        # Other option could be to sample directly from the non-central chisquare..
        out = np.zeros_like(r0)
        c = -self.vol*self.vol*np.expm1(-self.kappa*(t-s))/(4.*self.kappa)
        l = r0*np.exp(-self.kappa*(t-s))/c
        # Case 1
        chi1 = rng.chisquare(df=self.d[self.idx1]-1)
        w = rng.standard_normal(size=len(self.idx1))
        out[self.idx1] = c[self.idx1]*(np.power(w+np.sqrt(l[self.idx1]),2) + chi1)
        # Case 2
        pois = rng.poisson(lam=0.5*l[self.idx2])
        out[self.idx2] = c[self.idx2] * rng.chisquare(df=self.d[self.idx2]+2*pois)
        return out
    
    def __aux_den(self, t:float, T:float):
        return 2*self.h+(self.kappa+self.h)*np.expm1(self.h*(T-t))

    def __A(self, t:float, T:float):
        num = 2.*self.h*np.exp(0.5*(self.kappa+self.h)*(T-t))
        return np.power(num/self.__aux_den(t,T),2.*self.kappa*self.theta/self.vol**2)
    
    def B(self, t:float, T:float):
        return 2.*np.expm1(self.h*(T-t))/self.__aux_den(t,T)
    
    def fcirpp(self, t:float):
        num1 = 2.*self.kappa*self.theta*np.expm1(self.h*t)
        den = self.__aux_den(0,t)
        return num1/den + self.x0*4.*np.power(self.h,2)*np.exp(self.h*t)/np.power(den,2)
    
    def alpha(self, t:float):
        return self.Y.forward(t) - self.fcirpp(t)
    
    def A(self, t:float, T:float):
        return self.Y.discounts(T)/self.Y.discounts(t)*self.__A(0,t)/self.__A(0,T)*np.exp(self.x0*(self.B(0,T)-self.B(0,t)))*\
            self.__A(t,T)
    





        
