import numpy as np
import enum
import scipy.stats as st
import scipy.optimize as optimize
import random
import matplotlib.pyplot as plt

class OptionType(enum.Enum):
    CALL = 1.0
    PUT = -1.0

# Monte-Carlo simulation for the call option value
def simulate_asset_paths(S0, r, T, num_paths, num_steps):
    sigma = lambda t: 0.6 - 0.2 * np.exp(-1.5 * t)

    dt = T / num_steps
    times = np.linspace(0, T, num_steps + 1)
    
    # Generate random paths
    Z = np.random.normal(0.0, 1.0, [num_paths, num_steps])
    for i in range(num_steps):
        Z[:, i] = (Z[:, i] - np.mean(Z[:, i])) / np.std(Z[:, i])  # Standardize

    dW = np.sqrt(dt) * Z  # Brownian increments

    S = np.full(num_paths, S0)  # Store only last value (at maturity)
    current_sigma = np.vectorize(sigma)(times)  # Time-dependent volatility

    # Simulate stock price paths
    for i in range(num_steps):
        S = S * np.exp((r - 0.5 * current_sigma[i]**2) * dt + current_sigma[i]  * dW[:, i])
    
    return S  # Return the entire matrix of paths

def BS_Call_Option_Price(CP,S_0,K,sigma,tau,r):
    if K is list:
        K = np.array(K).reshape([len(K),1])
    d1  = (np.log(S_0 / K) + (r + 0.5 * np.power(sigma,2.0)) * tau) / (sigma * np.sqrt(tau))
    d2  = d1 - sigma * np.sqrt(tau)
    if CP == OptionType.CALL:
        value = st.norm.cdf(d1) * S_0 - st.norm.cdf(d2) * K * np.exp(-r * tau)
    elif CP == OptionType.PUT:
        value = st.norm.cdf(-d2) * K * np.exp(-r * tau) - st.norm.cdf(-d1)*S_0
    return value

def ImpliedVolatility1(CP,marketPrice,S0,K,T,r):
    # To determine initial volatility we define a grid for sigma
    # and interpolate on the inverse function

    sigmaGrid = np.linspace(0.0, 2.0, int(1e7))
    optPriceGrid = BS_Call_Option_Price(CP,S0,K,sigmaGrid,T,r)
    sigmaInitial = np.interp(marketPrice,optPriceGrid,sigmaGrid)
    
    # Use already determined input for the local-search (final tuning)
    func = lambda sigma: np.power(BS_Call_Option_Price(CP,S0,K,sigma,T,r)-marketPrice, 1)

    #d2  = lambda sigma : (np.log(S0 / K) + (r - 0.5 * np.power(sigma,2.0)) * T) / (sigma * np.sqrt(T))
    #deriv = lambda sigma: -K * np.exp(-r * T) * st.norm.pdf(d2(sigma)) * np.sqrt(T)

    if sigmaInitial < 1e-7:
        sigmaInitial = 0.5
    if sigmaInitial > 1.0-1e-7:
        sigmaInitial = 0.5

    try:
        #impliedVol = optimize.newton(func, sigmaInitial, fprime=deriv, tol=1e-10)
        impliedVol = optimize.newton(func, sigmaInitial, tol=1e-10, maxiter=200)
    except RuntimeError:
        impliedVol = sigmaInitial
    except RuntimeWarning:
        impliedVol = sigmaInitial
    
    return impliedVol

#%%
# Parameters
S0 = 1
r = 0.05
num_steps = 100
num_paths = int(1e7)
strikes = np.linspace(0.1, 3.0, 5).round(1)
maturities = np.linspace(0.1, 3, 5).round(1)

# Store results
IV_surface = np.zeros((len(strikes), len(maturities)))

random.seed(42)  # For reproducibility

# Compute implied vol for each strike and maturity
for j, T in enumerate(maturities):
    print(f"{j+1} - Maturity T={T}")
    # Extract prices at time T
    ST = simulate_asset_paths(S0, r, T, num_paths, num_steps)

    for i, K in enumerate(strikes):
        # Compute the payoff for the call option
        print(f"Strike K={K}")
        H = np.maximum(ST - K, 0)
        Vc= np.exp(-r * T) * np.mean(H)
        
        # Compute implied volatility
        iv = ImpliedVolatility1(OptionType.CALL, Vc, S0, K, T, r)
        IV_surface[i, j] = iv

#%%
# Meshgrid for plotting
K_mesh, T_mesh = np.meshgrid(strikes, maturities)

# Plot implied volatility surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(K_mesh, T_mesh, IV_surface, cmap='viridis')
ax.view_init(elev=20, azim=-130)
ax.set_title('Implied Volatility Surface (GBM with Time-Dependent Volatility)')
ax.set_xlabel('Maturity (T)')
ax.set_ylabel('Strike Price (K)')
ax.set_zlabel('Implied Volatility')
plt.tight_layout()
plt.show()

# %%
iv_mod = np.clip(IV_surface, 0.3, 0.6)

# Plot implied volatility surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(K_mesh, T_mesh, iv_mod, cmap='viridis')
ax.view_init(elev=20, azim=130)
ax.set_title('Implied Volatility Surface (GBM with Time-Dependent Volatility)')
ax.set_xlabel('Maturity (T)')
ax.set_ylabel('Strike Price (K)')
ax.set_zlabel('Implied Volatility')
plt.tight_layout()
plt.show()
