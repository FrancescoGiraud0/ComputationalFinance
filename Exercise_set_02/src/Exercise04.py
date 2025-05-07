"""
Python version 3.12.6
Computational Finance - Exercise Set 2
Exercise 4, Monte Carlo Simulation of two correlated assets
author: Francesco Giraudo - VU Amsterdam
"""
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def simulate_two_assets(S0_1, S0_2, sigma1, sigma2, r, rho,
                        T=1.0, n_paths=100000, n_steps=1, seed=None):
    dt = T / n_steps
    S1 = np.full((n_paths, n_steps + 1), S0_1)
    S2 = np.full((n_paths, n_steps + 1), S0_2)

    if seed is not None:
        np.random.seed(seed)
    dW1 = np.random.normal(0, 1, (n_paths, n_steps))
    dW2 = np.random.normal(0, 1, (n_paths, n_steps))

    dW1 = (dW1 - np.mean(dW1, axis=0)) / np.std(dW1, axis=0) * np.sqrt(dt)
    dW2 = (dW2 - np.mean(dW2, axis=0)) / np.std(dW2, axis=0) * np.sqrt(dt)

    for step in range(n_steps):
        S1[:, step+1] = S1[:, step] + r * S1[:, step] * dt + \
                        sigma1 * S1[:, step] * dW1[:, step]
        S2[:, step+1] = S2[:, step] + r * S2[:, step] * dt + \
                        rho * sigma2 * S2[:, step] * dW1[:, step] + \
                        np.sqrt(1 - rho**2) * sigma2 * S2[:, step] * dW2[:, step]

    return S1[:, -1], S2[:, -1]

def mean_var_max_x1_x2(mu1, mu2, sigma1, sigma2, rho):
    theta = np.sqrt(sigma1**2 + sigma2**2 - 2 * rho * sigma1 * sigma2)
    delta = (mu1 - mu2) / theta

    Phi_delta = norm.cdf(delta)
    Phi_neg_delta = norm.cdf(-delta)
    phi_delta = norm.pdf(delta)

    # Mean of max(X1, X2)
    mean = (
        mu1 * Phi_delta
        + mu2 * Phi_neg_delta
        + theta * phi_delta
    )

    # Second moment of max(X1, X2)
    E_X2 = (
        (sigma1**2 + mu1**2) * Phi_delta
        + (sigma2**2 + mu2**2) * Phi_neg_delta
        + (mu1 + mu2) * theta * phi_delta
    )

    var = E_X2 - mean**2

    return mean, var

def mean_var_max_lognormal(mu1, mu2, sigma1, sigma2, rho):
    mu_Z, var_Z = mean_var_max_x1_x2(mu1, mu2, sigma1, sigma2, rho)
    
    mean_ln = np.exp(mu_Z + 0.5 * var_Z)
    var_ln = (np.exp(var_Z) - 1) * np.exp(2 * mu_Z + var_Z)

    return mean_ln, var_ln


def pdf_max_normal(x, mu1, mu2, sigma1, sigma2, rho):
    x = np.asarray(x)
    z1 = (-x + mu1) / sigma1
    z2 = (-x + mu2) / sigma2
    arg1 = (rho * (-x + mu1) / sigma1 - (-x + mu2) / sigma2) / np.sqrt(1 - rho**2)
    arg2 = (rho * (-x + mu2) / sigma2 - (-x + mu1) / sigma1) / np.sqrt(1 - rho**2)

    f1 = (1 / sigma1) * norm.pdf(z1) * norm.cdf(arg1)
    f2 = (1 / sigma2) * norm.pdf(z2) * norm.cdf(arg2)
    return f1 + f2


def pdf_max_lognormal(z, mu1, mu2, sigma1, sigma2, rho):
    z = np.asarray(z)
    z[z <= 0] = 1e-12
    x = np.log(z)
    return pdf_max_normal(x, mu1, mu2, sigma1, sigma2, rho) / z

if __name__ == "__main__":
    # Parameters
    S0_1 = 1.0
    S0_2 = 1.0
    T = 1.0
    r = 0.01
    sigma1 = 0.4
    sigma2 = 0.15
    rho_values = [-0.9, 0.9]
    n_steps = 100
    n_paths = int(1e5)
    conf_level = 0.95

    for rho in rho_values:
        # Simulate two correlated assets
        S1_T, S2_T = simulate_two_assets(S0_1, S0_2, sigma1, sigma2, r, rho,
                                        T, n_paths, n_steps, seed=42)
        # Calculate the payoff and confidence interval
        payoff = np.maximum(S1_T, S2_T)
        sim_mean = np.mean(payoff)
        sim_var = np.var(payoff, ddof=1)
        std_error = np.std(payoff, ddof=1) / np.sqrt(n_paths)
        z = norm.ppf(0.5 + conf_level / 2)
        conf_interval = (sim_mean - z * std_error, sim_mean + z * std_error)

        # Theoretical PDF
        mu1 = np.log(S0_1) + (r - 0.5 * sigma1**2) * T
        mu2 = np.log(S0_2) + (r - 0.5 * sigma2**2) * T
        sigma1_T = sigma1 * np.sqrt(T)
        sigma2_T = sigma2 * np.sqrt(T)
        # Theoretical mean and variance
        theoretical_mean, theoretical_var = mean_var_max_lognormal(mu1, mu2, sigma1_T, sigma2_T, rho)


        z_vals = np.linspace(0.1, 3.0, 500)
        pdf_vals = pdf_max_lognormal(z_vals, mu1, mu2, sigma1_T, sigma2_T, rho)

        plt.figure(figsize=(6, 4))
        plt.xlim(0.5, 3)
        plt.ylim(0,3.5)
        plt.hist(payoff, bins=200, density=True, alpha=0.5, label='Simulated Payoff')
        plt.plot(z_vals, pdf_vals, label='Theoretical PDF', linewidth=0.5, color='red')
        plt.title(f'Distribution of max(S1, S2), rho = {rho}')
        plt.xlabel('Payoff')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Print results
        print(f"\nCorrelation: rho = {rho}")
        print(f"Sample Mean: {sim_mean:.4f}")
        print(f"Sample Variance: {sim_var:.4f}")
        print(f"{conf_level * 100:.1f}% Confidence Interval for Mean: ({conf_interval[0]:.4f}, {conf_interval[1]:.4f})")
        print(f"Sample Std. Error: {std_error:.4f}")
        # Print the theoretical mean and variance
        print(f"Theoretical Mean: {theoretical_mean:.4f}")
        print(f"Theoretical Variance: {theoretical_var:.4f}")
