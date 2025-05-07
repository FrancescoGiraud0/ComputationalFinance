import numpy as np
import enum
import scipy.stats as st
import scipy.optimize as optimize
import matplotlib.pyplot as plt

class OptionType(enum.Enum):
    CALL = 1.0
    PUT = -1.0

# Monte-Carlo simulation for the call option value
def simulate_asset_paths(S0, r, T, num_paths, num_steps):
    sigma = lambda t: 0.6 - 0.2 * np.exp(-1.5 * t)

    dt = T / num_steps
    times = np.linspace(0, T, num_steps + 1)
    
    dW = np.random.normal(0.0, 1.0, [num_paths, num_steps])
    for i in range(num_steps):
        dW[:, i] = (dW[:, i] - np.mean(dW[:, i])) / np.std(dW[:, i])  # Standardize

    dW = np.sqrt(dt) * dW  # Brownian increments

    S = np.full((num_paths, num_steps + 1), S0)  # Allocate space for all paths
    current_sigma = np.vectorize(sigma)(times)   # Time-dependent volatility

    # Simulate stock price paths
    for i in range(num_steps):
        current_sigma = sigma(times[i])
        S[:, i + 1] = S[:, i] * np.exp((r - 0.5 * current_sigma**2) * dt + current_sigma * dW[:, i])
    
    return S  # Return the entire matrix of paths

def BS_Call_Option_Price(CP, S_0, K, sigma, tau, r):
    if K is list:
        K = np.array(K).reshape([len(K), 1])
    d1 = (np.log(S_0 / K) + (r + 0.5 * np.power(sigma, 2.0)) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    if CP == OptionType.CALL:
        value = st.norm.cdf(d1) * S_0 - st.norm.cdf(d2) * K * np.exp(-r * tau)
    elif CP == OptionType.PUT:
        value = st.norm.cdf(-d2) * K * np.exp(-r * tau) - st.norm.cdf(-d1) * S_0
    return value

def ImpliedVolatility(CP, Vmkt, S0, K, T, r):
    sigmaGrid = np.linspace(0.0, 1.0, int(2e6))
    optPriceGrid = BS_Call_Option_Price(CP, S0, K, sigmaGrid, T, r)
    sigmaInitial = np.interp(Vmkt, optPriceGrid, sigmaGrid)
    
    func = lambda sigma: np.power(BS_Call_Option_Price(CP, S0, K, sigma, T, r) - Vmkt, 1)

    if sigmaInitial <= 1e-6 or sigmaInitial >= 1.0 - 1e-6:
        sigmaInitial = 0.5

    try:
        impliedVol = optimize.newton(func, sigmaInitial, tol=1e-10, maxiter=1000)
    except (RuntimeError, RuntimeWarning):
        impliedVol = sigmaInitial
    
    return impliedVol

# Parameters
S0 = 1.0
r = 0.05
num_steps = 500
num_paths = int(1e6)
strikes = np.linspace(0.1, 5.0, 20).round(1)
maturities = np.linspace(0.1, 5.0, 10).round(1)

# Precompute step indices for each maturity
max_maturity = maturities[-1]
step_indices = (maturities / max_maturity * num_steps).astype(int)

# Simulate all paths for the maximum maturity
all_paths = simulate_asset_paths(S0, r, max_maturity, num_paths, num_steps)

t = np.linspace(0, max_maturity, num_steps + 1)

plt.figure(figsize=(10, 6))
for i in range(1000):
    plt.plot(t, all_paths[i, :], lw=0.5, color='blue', alpha=0.5)
plt.title('Simulated Asset Paths', fontsize=14)
plt.xlabel('Time (t)')
plt.ylabel('Asset Price S(t)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# Store results
IV_surface = np.zeros((len(maturities), len(strikes)))

# Compute implied vol for each strike and maturity
for j, step_index in enumerate(step_indices):
    T = maturities[j]
    print(f"{j+1} - Maturity T={T}")
    ST = all_paths[:, step_index]  # Extract prices at the exact step
    for i, K in enumerate(strikes):
        print(f"Stike K={K}")
        H = np.maximum(ST - K, 0)
        Vc = np.exp(-r * T) * np.mean(H)
        iv = ImpliedVolatility(OptionType.CALL, Vc, S0, K, T, r)
        IV_surface[j, i] = iv

# Meshgrid for plotting
K_mesh, T_mesh = np.meshgrid(strikes, maturities)

# Plot implied volatility surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(K_mesh, T_mesh, IV_surface, cmap='viridis')
ax.view_init(elev=15, azim=-55)
ax.set_title('Implied Volatility Surface (GBM with Time-Dependent Volatility)')
ax.set_xlabel('Strike Price (K)')
ax.set_ylabel('Maturity (T)')
ax.set_zlabel('Implied Volatility')
plt.tight_layout()
plt.show()

iv_mod = np.clip(IV_surface, 0.3, 0.6)

#%%
# Plot clipped implied volatility surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(K_mesh, T_mesh, iv_mod, cmap='viridis')
ax.view_init(elev=15, azim=55)
ax.set_title('Clipped Implied Volatility Surface (GBM with Time-Dependent Volatility)')
ax.set_xlabel('Strike Price (K)')
ax.set_ylabel('Maturity (T)')
ax.set_zlabel('Implied Volatility')
plt.tight_layout()
plt.show()
# %%
