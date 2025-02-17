import sys
sys.path.append("./")
sys.path.append("../")

import numpy as np
from scipy.stats import ncx2
import matplotlib.pyplot as plt

from src.parametric_curves import ConstantTermStruct
from src.ir_models import CIR

# Model values
# Cir
# x0 = 0.04
x0 = 0.01
theta = 0.035
# kappa = 0.2
kappa = 1
vol = 0.05
# Cte term struct
y0 = 0.01


# Simulation parameters
M = 100# Nber mc paths
N = 10 # Nber simulation steps
tstart = 0 # start time
tend = 5 # end time
times = np.linspace(tstart, tend, N+1)
dt = times[1] - times [0]
tenors = np.array([0., 0.5, 1., 2., 5., 10., 15., 20., 30.])

term_struct = ConstantTermStruct(y0*np.ones(M))
model = CIR(kappa*np.ones(M), theta*np.ones(M), vol*np.ones(M), x0*np.ones(M), term_struct)
model.fellerCondition()

# Simulation
rng = np.random.default_rng()
X = np.zeros((N+1,M))
yp = np.zeros((N+1,M, tenors.size))
X[0] = x0

for j in range(1,N+1):
    X[j] = model.shortRateSimulStep(times[j-1],times[j],X[j-1],rng)
    yp[j] = model.computeYieldPoints(times[j], X[j],tenors)
print(np.sum(X<=0, axis=1))

with plt.ion():
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

ax1.set_xlim([tstart,tend+0.3])
ax1.set_ylim([0., 0.07])

for n in range(1,N+2):
    ax1.plot(times[:n], X[:n])
    ax2.plot(tenors[1:], yp[n-1,:,1:].T)
    plt.pause(1)



fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
hist = ax.hist(X[1], density=True, bins='auto')
# ax.plot(xref, ncx2.pdf(xref, degrees_of_freedom,non_central_loca_param))
plt.show()

