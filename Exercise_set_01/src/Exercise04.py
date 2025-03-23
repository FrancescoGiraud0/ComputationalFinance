"""
Python version 3.12.6
Computational Finance - Exercise Set 1
Exercise 4, Numerical Validation for T=10
@author: Francesco Giraudo - VU Amsterdam
"""
import numpy as np
import matplotlib.pyplot as plt

def compute_paths_less_precision(NoOfPaths, NoOfSteps, T):
    """
    Simulate Wiener process paths and compute X(t) = W(t) - (t/T) W(T-t)
    """
    dt = T / NoOfSteps
    Z = np.random.normal(0.0, 1.0, [NoOfPaths, NoOfSteps])
    W = np.zeros([NoOfPaths, NoOfSteps + 1])
    X = np.zeros([NoOfPaths, NoOfSteps + 1])
    time = np.linspace(0, T, NoOfSteps + 1)
    
    # Simulating BM path
    for i in range(NoOfSteps):
        if NoOfPaths > 1:
            Z[:, i] = (Z[:, i] - np.mean(Z[:, i])) / np.std(Z[:, i])
        
        W[:, i + 1] = W[:, i] + np.sqrt(dt) * Z[:, i]
    
    for i in range(NoOfSteps + 1):
        t = time[i]
        T_minus_t_idx = int((T - t) / T * NoOfSteps)
        X[:, i] = W[:, i] - (t / T) * W[:, T_minus_t_idx]
    
    return time, X

def compute_paths(NoOfPaths, NoOfSteps, T):
    """
    Simulate Wiener process paths and compute X(t) = W(t) - (t/T) W(T-t)
    with higher precision by using more steps for the Brownian motion simulation.
    """
    # Use a finer grid for the Brownian motion simulation
    NoOfSteps_BM = NoOfSteps * 10  # Increase the number of steps for BM
    dt = T / NoOfSteps_BM  # Smaller time step for higher precision
    Z = np.random.normal(0.0, 1.0, [NoOfPaths, NoOfSteps_BM])
    W = np.zeros([NoOfPaths, NoOfSteps_BM + 1])
    X = np.zeros([NoOfPaths, NoOfSteps + 1])
    time = np.linspace(0, T, NoOfSteps + 1)  # Time grid for X(t)
    time_BM = np.linspace(0, T, NoOfSteps_BM + 1)  # Finer time grid for W(t)

    # Simulating BM path with finer steps
    for i in range(NoOfSteps_BM):
        if NoOfPaths > 1:
            Z[:, i] = (Z[:, i] - np.mean(Z[:, i])) / np.std(Z[:, i])
        
        W[:, i + 1] = W[:, i] + np.sqrt(dt) * Z[:, i]
    
    # Compute X(t) = W(t) - (t/T) W(T-t)
    for i in range(NoOfSteps + 1):
        t = time[i]
        # Find the index corresponding to T - t in the finer BM time grid
        T_minus_t_idx = int((T - t) / T * NoOfSteps_BM)
        X[:, i] = W[:, i * 10] - (t / T) * W[:, T_minus_t_idx]
    
    return time, X

# Parameters
T = 10
NoOfPaths = 10000
NoOfSteps = 1000

# Random seed for reproducibility
np.random.seed(2008)

# Simulate paths
time, X = compute_paths(NoOfPaths, NoOfSteps, T)

# Compute simulated variance
simulated_var = np.var(X, ddof=1, axis=0)

# Analytical variance
analytical_var = np.array([t - (t**2)/T - (t**3)/(T**2) if t <= T/2 
                          else -t + 3*(t**2)/T - (t**3)/(T**2) 
                          for t in time])

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(time, analytical_var, label='Analytical Variance', linestyle='-', color='b')
plt.plot(time, simulated_var, label='Simulated Variance', linestyle='--', color='r')
plt.xlabel("Time t")
plt.ylabel("Variance")
plt.title("Analytical vs Simulated Variance of X(t)")
plt.legend()
plt.grid()
plt.show()

# Absolute difference
absolute_diff = np.abs(analytical_var - simulated_var)

plt.figure(figsize=(8, 6))
plt.plot(time, absolute_diff, label='Absolute Difference',
         linestyle='-', color='b', linewidth=1)
plt.xlabel("Time t")
plt.ylabel("Variance / Absolute Difference")
plt.title("Analytical vs Simulated Variance of X(t) (Absolute Difference)")
plt.legend()
plt.grid()
plt.show()
