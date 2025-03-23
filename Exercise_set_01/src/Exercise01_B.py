#%%
"""
Computational Finance - Exercise Set 1
Exercise 1.B, Numerical Validation
@author: Francesco Giraudo
"""
import numpy as np
import matplotlib.pyplot as plt

def ComputeIntegrals(NoOfPaths,NoOfSteps,T):
    """
    This function generates $NoOfPaths random paths of the two stochastic 
    process  and compute the integral numerically given $NoOfSteps number of
    steps from 0 to T.
    """
    Z = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    W = np.zeros([NoOfPaths, NoOfSteps+1])
    I1 = np.zeros([NoOfPaths, NoOfSteps+1])
    I2 = np.zeros([NoOfPaths, NoOfSteps+1])
    time = np.zeros([NoOfSteps+1])
    
    dt = T / float(NoOfSteps)

    # For each step i compute the the numerical integral value from 0 to t(i)
    # where t(i)=i*dt
    for i in range(0,NoOfSteps):

        # Making sure that samples from a normal have mean 0 and variance 1
        if NoOfPaths > 1:
            Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
        
        # Compute value of the sample BM W(t)
        W[:,i+1]  = W[:,i]  + np.power(dt, 0.5)*Z[:,i]
        # Compute the value of the 2 sample stochastic process
        I1[:,i+1] = I1[:,i] + W[:,i]*dt
        I2[:,i+1] = I2[:,i] + (T-time[i])*(W[:,i+1]-W[:,i])
        
        time[i+1] = time[i] + dt
    
    paths = {"time":time,"W":W,"I1":I1,"I2":I2}
    return paths

# 1. Simulate one path and compute the integrals and plot a visualization of the
#    process.

# Parameters of the simulation
NoOfPaths = 1
NoOfSteps = 100
T = 5

# Random seed for reproducibility
np.random.seed(20080915)
W_t = ComputeIntegrals(NoOfPaths,NoOfSteps,T)

timeGrid = W_t["time"]
Ws = W_t["W"]
intWsds = W_t["I1"]
int_t_s_dWs = W_t["I2"]

plt.figure(1)
plt.plot(timeGrid, np.transpose(Ws))
plt.plot(timeGrid, np.transpose(intWsds),'r')
plt.plot(timeGrid, np.transpose(int_t_s_dWs),'k')
plt.grid()
plt.xlabel("time")
plt.ylabel("value")
plt.title("Integrated Brownian Motion paths")

# %%
NoOfPaths = 10000
NoOfSteps = 1000
T = 5

np.random.seed(16370201)
W_t = ComputeIntegrals(NoOfPaths,NoOfSteps,T)

timeGrid = W_t["time"]
Ws = W_t["W"]
intWsds = W_t["I1"]
int_t_s_dWs = W_t["I2"]

# Computing the mean absolute error
diff = intWsds[:,-1]-int_t_s_dWs[:,-1]
abs_diff = np.abs(diff)

print(f"Mean absolute error: {np.mean(abs_diff):1e}")
print(f"Max abolute error: {np.max(abs_diff):1e}\n")
print(f"{"-"*20}\n")
print(f"Mean error: {np.mean(diff):.1e}")
print(f"Error std. deviation:  {np.std(abs_diff):1e}")

ci = np.mean(diff) + np.array([-1.96, 1.96]) * (np.std(diff, ddof=1) / np.sqrt(len(diff)))
print(f"95% Confidence Interval: {ci}")


#%%
NoOfPaths = 10000
NoOfSteps = [(2**i) for i in range(1,16)]
T = 5

mean_abs_error = np.zeros(len(NoOfSteps))

np.random.seed(16370201)
for i in range(len(NoOfSteps)):
    i
    W_t = ComputeIntegrals(NoOfPaths, NoOfSteps[i], T)
    abs_diff = np.abs(W_t["I1"][:,-1] - W_t["I2"][:,-1])
    mean_abs_error[i] = np.mean(abs_diff)

plt.figure()
plt.plot(NoOfSteps, mean_abs_error)
plt.grid()
plt.yscale("log")
plt.xlabel("Number of steps")
plt.ylabel("Value")
plt.title("Absolute error convergence to 0")
# %%
