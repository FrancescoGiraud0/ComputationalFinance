"""
Python version 3.12.6
Computational Finance - Exercise Set 1
Exercise 1.B, Numerical Validation
@author: Francesco Giraudo - VU Amsterdam
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
np.random.seed(1929)

W_t = ComputeIntegrals(NoOfPaths,NoOfSteps,T)

timeGrid = W_t["time"]
Ws = W_t["W"]
intWsds = W_t["I1"]
int_t_s_dWs = W_t["I2"]

plt.figure(figsize=(8, 6))
plt.plot(timeGrid, np.transpose(Ws), label='BM')
plt.plot(timeGrid, np.transpose(intWsds), 'r', label=r'$\int_0^5{W(s)\,ds}$')
plt.plot(timeGrid, np.transpose(int_t_s_dWs), 'k', label=r'$\int_0^5{(5-s)\,dW(s)}$')
plt.grid()
plt.xlabel("time")
plt.ylabel("value")
plt.title("Integrated Brownian Motion paths")
plt.legend(loc='lower right')
plt.show()

# 2. Simulating the previous step for 1000 random paths and computing the mean
#    absolute error convergence and 95% confidence intervals for different 
#    number of steps

NoOfPaths = 1000
NoOfSteps = [(2**i) for i in range(1,16)]
T = 5
mean_abs_error = np.zeros(len(NoOfSteps))
std_error = np.zeros(len(NoOfSteps))

# Random seed for reproducibility
np.random.seed(2008)

# For every step size in NoOfSteps, simulate and compute mean absolute error
for i in range(len(NoOfSteps)):
    W_t = ComputeIntegrals(NoOfPaths, NoOfSteps[i], T)
    diff = W_t["I1"][:,-1] - W_t["I2"][:,-1]
    abs_diff = np.abs(diff)
    mean_abs_error[i] = np.mean(abs_diff)
    # Calculate standard error for confidence interval
    std_error[i] = np.std(abs_diff, ddof=1) / np.sqrt(NoOfPaths)

# Create confidence intervals (95%)
upper_bound = mean_abs_error + 1.96 * std_error
lower_bound = mean_abs_error - 1.96 * std_error

plt.figure(figsize=(8, 6))
# Plot mean absolute error with a label for the legend
plt.plot(NoOfSteps, mean_abs_error, 'b-', linewidth=1, 
         label='Mean Absolute Error')
# Add confidence interval as a shaded area
plt.fill_between(NoOfSteps, lower_bound, upper_bound, color='r', 
                 alpha=0.5, label='95% Confidence Interval')

# Logarithmic scales for both axes for better visualization
plt.xscale("log")
plt.yscale("log")
# Add grid, labels, and title
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.xlabel("Number of Steps (log scale)")
plt.ylabel("Mean Absolute Error (log scale)")
plt.title("Convergence of Mean Absolute Error with Confidence Intervals")
# Add legend
plt.legend(loc='best')
# Tight layout for better spacing
plt.tight_layout()
plt.show()