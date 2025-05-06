"""
Python version 3.12.6
Computational Finance - Exercise Set 2
Exercise 1.A, Analytical Price of Call Option with time-dependent volatility
@author: Francesco Giraudo - VU Amsterdam
"""
import scipy.stats as st
import enum
import numpy as np
import matplotlib.pyplot as plt

class OptionType(enum.Enum):
    CALL = 1.0
    PUT = -1.0

def Vc(CP,S_0,K,sigma,tau,r):
    if K is list:
        K = np.array(K).reshape([len(K),1])
    d1  = (np.log(S_0 / K) + (r + 0.5 * np.power(sigma,2.0)) * tau) / (sigma * np.sqrt(tau))
    d2  = d1 - sigma * np.sqrt(tau)
    if CP == OptionType.CALL:
        value = st.norm.cdf(d1) * S_0 - st.norm.cdf(d2) * K * np.exp(-r * tau)
    elif CP == OptionType.PUT:
        value = st.norm.cdf(-d2) * K * np.exp(-r * tau) - st.norm.cdf(-d1)*S_0
    return value

def sigma_star(T):
    return np.sqrt(0.36 + (0.16/T)*(np.exp(-1.5*T)-1) - (0.04/(3*T))*(np.exp(-0.5*T)-1))

S0 = 1
r = 0.05
M0 = 1

# Plot graph of sigma_star for different values of T
T = np.linspace(1e-4, 3, int(3e4))
# Apply the sigma_star function to each element of the grid T
sigma_star_T = np.vectorize(sigma_star)(T)
#plot the graph
plt.plot(T, sigma_star_T)
plt.title(r'Volatility($\sigma_*$) at different maturities (T)')
plt.xlabel('Maturity (T)')
plt.ylabel(r'Volatility ($\sigma_*$)')
plt.grid()
plt.show()

# Compute the value the price of call option at different maturities for different strikes
T = np.linspace(0.01, 3, 100)
K = np.linspace(0.01, 3, 100)
T, K = np.meshgrid(T, K)
# Apply the Vc (BS price function) to each element of the grid T and K
V = np.vectorize(lambda tau,k: Vc(OptionType.CALL, S0, k, sigma_star(tau), tau, r))(T,K)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(K, T, V, cmap='viridis')
ax.view_init(elev=15, azim=-55)
ax.set_xlabel('Strike Price (K)', labelpad=5)
ax.set_ylabel('Maturity (T)', labelpad=5)
ax.set_zlabel(r'Call Option Price ($V_C$)', labelpad=3)
ax.zaxis.label.set_rotation(90)
ax.zaxis.label.set_horizontalalignment('right')
plt.title('Call option price values at different maturities (T) and strikes (K)')
plt.show()


s = np.vectorize(sigma_star)(T)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(K, T, s, cmap='viridis')
ax.view_init(elev=15, azim=-55)
ax.set_zlim(0.3, 0.6)
ax.set_xlabel('Strike Price (K)', labelpad=5)
ax.set_ylabel('Maturity (T)', labelpad=5)
ax.set_zlabel(r'Volatility ($\sigma_*$)', labelpad=3)
ax.zaxis.label.set_rotation(90)
ax.zaxis.label.set_horizontalalignment('left')
ax.set_title('Volatility value at different maturities (T) and strikes (K)')
plt.show()