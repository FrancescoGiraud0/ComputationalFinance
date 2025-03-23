#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Black-Scholes analytical formula for European call option
def bs_call_price(S0, K, r, sigma, T):
    d1 = (np.log(S0/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# Monte Carlo simulation that computes BOTH standardized and non-standardized prices
def mc_call_price_comparison(S0, K, r, sigma, T, num_paths, num_steps):
    dt = T / num_steps
    
    # Generate random paths - use same random numbers for both methods
    Z = np.random.normal(0.0, 1.0, [num_paths, num_steps])
    
    # Create a standardized copy of Z
    Z_std = Z.copy()
    for i in range(num_steps):
        # Standardize to ensure mean=0 and std=1
        Z_std[:,i] = (Z_std[:,i] - np.mean(Z_std[:,i])) / np.std(Z_std[:,i])
    
    # Initialize stock price arrays for both methods
    S_std = np.zeros([num_paths, num_steps + 1])
    S_nonstd = np.zeros([num_paths, num_steps + 1])
    S_std[:, 0] = S0
    S_nonstd[:, 0] = S0
    
    # Simulate stock price paths
    for i in range(num_steps):
        # Standardized version
        S_std[:, i+1]    = S_std[:, i]    * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z_std[:, i])
        
        # Non-standardized version
        S_nonstd[:, i+1] = S_nonstd[:, i] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, i])
    
    # Calculate option payoffs at maturity
    payoffs_std = np.maximum(S_std[:, -1] - K, 0)
    payoffs_nonstd = np.maximum(S_nonstd[:, -1] - K, 0)
    
    # Calculate present values
    option_price_std = np.exp(-r * T) * np.mean(payoffs_std)
    std_error_std = np.exp(-r * T) * np.std(payoffs_std) / np.sqrt(num_paths)
    
    option_price_nonstd = np.exp(-r * T) * np.mean(payoffs_nonstd)
    std_error_nonstd = np.exp(-r * T) * np.std(payoffs_nonstd) / np.sqrt(num_paths)
    
    return option_price_std, std_error_std, option_price_nonstd, std_error_nonstd

# Parameters
S0 = 100.0      # Initial stock price
K  = 95.0       # Strike price
r = 0.03        # Risk-free rate
sigma = 0.10    # Volatility
T = 1           # Time to maturity (in years)

# Calculate analytical price
bs_price = bs_call_price(S0, K, r, sigma, T)
print(f"Black-Scholes analytical price: {bs_price:.6f}")

# Test convergence with increasing number of paths
path_counts = [int(i) for i in np.logspace(-1, 4, num=40) if int(i)>1]
num_steps = 365

# Set a fixed seed to ensure reproducibility
np.random.seed(42)

# Arrays to store results
std_prices = []
std_errors = []
nonstd_prices = []
nonstd_errors = []
abs_errors_std = []
abs_errors_nonstd = []

for paths in path_counts:
    # Run simulation with both methods at once
    np.random.seed(42)  # Reset seed for reproducibility
    price_std, error_std, price_nonstd, error_nonstd = mc_call_price_comparison(
        S0, K, r, sigma, T, paths, num_steps
    )
    
    # Store results
    std_prices.append(price_std)
    std_errors.append(error_std)
    abs_errors_std.append(abs(price_std - bs_price))
    
    nonstd_prices.append(price_nonstd)
    nonstd_errors.append(error_nonstd)
    abs_errors_nonstd.append(abs(price_nonstd - bs_price))

# Plotting convergence
plt.figure(figsize=(8, 6))
plt.semilogx(path_counts, std_prices, 'o-', label='With Standardization')
plt.semilogx(path_counts, nonstd_prices, 's-', label='Without Standardization')
plt.axhline(y=bs_price, color='r', linestyle='-', label='Black-Scholes Price')
plt.xlabel('Number of Paths')
plt.ylabel('Option Price')
plt.title('Convergence of Option Prices')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.loglog(path_counts, abs_errors_std, 'o-', label='With Standardization')
plt.loglog(path_counts, abs_errors_nonstd, 's-', label='Without Standardization')
plt.xlabel('Number of Paths')
plt.ylabel('Absolute Error')
plt.title('Absolute Error vs. Black-Scholes Price')
plt.grid(True)
plt.legend()
plt.show()

#%%
# Additional analysis: Test stability with different random seeds
num_seeds = 1000
fixed_paths = 100

std_results = []
nonstd_results = []

np.random.seed(2008)

for seed in np.random.randint(0, 2**32, size=num_seeds):
    np.random.seed(seed)
    price_std, _, price_nonstd, _ = mc_call_price_comparison(S0, K, r, sigma, T, fixed_paths, num_steps)
    std_results.append(price_std)
    nonstd_results.append(price_nonstd)

print("\nSummary Statistics:")
print("With standardization:    Mean = {:.6f}, Std = {:.6f}".format(np.mean(std_results), np.std(std_results)))
print("Without standardization: Mean = {:.6f}, Std = {:.6f}".format(np.mean(nonstd_results), np.std(nonstd_results)))
print("Black-Scholes price:     {:.6f}".format(bs_price))
print("Mean absolute error (Standardized):     {:.6f}".format(np.mean(np.abs(np.array(std_results) - bs_price))))
print("Mean absolute error (Non-Standardized): {:.6f}".format(np.mean(np.abs(np.array(nonstd_results) - bs_price))))


#%%
# Compute common min and max
min_value = min(min(std_results), min(nonstd_results))
max_value = max(max(std_results), max(nonstd_results))

# Define common bins
num_bins = 100  # Number of bins (you can adjust this)
bins = np.linspace(min_value, max_value, num_bins + 1)  # Ensure same bin edges

# Create subplots (2 rows, 1 column)
fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# Histogram for Standardized Results (Top)
axes[0].hist(std_results, bins=bins, alpha=0.5, color='blue', label='With Standardization', edgecolor='black')
axes[0].axvline(x=bs_price, color='r', linestyle='-', label='Black-Scholes Price')
axes[0].set_ylabel('Frequency')
axes[0].set_title('With Standardization')
axes[0].legend()
axes[0].grid(True)

# Histogram for Non-Standardized Results (Bottom)
axes[1].hist(nonstd_results, bins=bins, alpha=0.5, color='green', label='Without Standardization', edgecolor='black')
axes[1].axvline(x=bs_price, color='r', linestyle='-', label='Black-Scholes Price')
axes[1].set_xlabel('Option Price')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Without Standardization')
axes[1].legend()
axes[1].grid(True)

# Adjust layout and save
plt.tight_layout()
plt.savefig('option_pricing_stability.png')
plt.show()
