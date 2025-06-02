"""
Python version 3.12.8
Computational Finance - Final Project
Task 1
@author: Francesco Giraudo - VU Amsterdam
"""

import numpy as np
import matplotlib.pyplot as plt

def price_convertible_bond(method='binomial', plot=False, **kwargs):
    if method == 'binomial':
        return price_convertible_bond_binomial_finegrid(plot=plot, **kwargs)
    else:
        raise ValueError(f"Unknown method {method}")

def price_convertible_bond_binomial_finegrid(S0, N, ell, K_coupon, r, sigma, T, N_steps, convertible_times, plot=False):
    """
    Fine binomial tree with many time steps.
    Conversion only allowed at specific convertible_times.
    """
    dt = T / N_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    discount = np.exp(-r * dt)
    coupon = N * K_coupon * (0.5)  # semiannual coupon (not every dt!)

    # Stock price tree
    stock_tree = np.zeros((N_steps + 1, N_steps + 1))
    for i in range(N_steps + 1):
        for j in range(i + 1):
            stock_tree[j, i] = S0 * (u ** j) * (d ** (i - j))

    # Bond value tree
    bond_tree = np.zeros_like(stock_tree)

    # Terminal condition
    for j in range(N_steps + 1):
        S_T = stock_tree[j, N_steps]
        bond_tree[j, N_steps] = max(N, ell * S_T)

    # Backward induction
    for i in range(N_steps - 1, -1, -1):
        for j in range(i + 1):
            continuation = (p * bond_tree[j + 1, i + 1] + (1 - p) * bond_tree[j, i + 1]) * discount
            # Only add coupon at coupon dates
            if i in convertible_times:
                continuation += coupon
                conversion = ell * stock_tree[j, i]
                bond_tree[j, i] = max(continuation, conversion)
            else:
                bond_tree[j, i] = continuation

    if plot:
        plot_tree(stock_tree, T, S0, N, ell, K_coupon, r, sigma, N_steps, tree_type='Stock Price')
        plot_tree(bond_tree, T, S0, N, ell, K_coupon, r, sigma, N_steps, tree_type='Bond Value')

    bond_value = bond_tree[0, 0]
    return bond_value


def plot_tree(tree, T, S0, N, ell, K_coupon, r, sigma, N_steps, tree_type='Stock Price'):
    dt = T / N_steps
    times = np.linspace(0, T, N_steps + 1)

    plt.figure(figsize=(14, 7))

    for i in range(N_steps + 1):
        for j in range(i + 1):
            plt.plot(times[i], tree[j, i], 'bo', markersize=3)
            if i < N_steps:
                plt.plot([times[i], times[i + 1]], [tree[j, i], tree[j, i + 1]], 'k-', lw=0.5)
                plt.plot([times[i], times[i + 1]], [tree[j, i], tree[j + 1, i + 1]], 'k-', lw=0.5)

    plt.title(f"{tree_type} Tree (Binomial Model)", fontsize=16)
    plt.xlabel("Time (years)", fontsize=14)
    plt.ylabel(tree_type, fontsize=14)
    plt.grid(True)

    param_text = (
        f"$S_0$ = {S0}\n"
        f"$N$ = {N}\n"
        f"$\\ell$ (conversion ratio) = {ell}\n"
        f"$K$ (coupon rate) = {K_coupon}\n"
        f"$r$ (risk-free rate) = {r}\n"
        f"$\\sigma$ (volatility) = {sigma}\n"
        f"$T$ (maturity) = {T}\n"
        f"Steps = {N_steps}"
    )

    plt.gca().text(1.02, 0.5, param_text, fontsize=12,
                   bbox=dict(boxstyle="round", facecolor="white", edgecolor="black"),
                   transform=plt.gca().transAxes, verticalalignment='center')

    plt.tight_layout()
    plt.savefig('01_'+tree_type.replace(" ", "")+'.png')
    plt.show()


if __name__ == "__main__":
    S0 = 200.0
    N = 1000.0
    ell = 3
    K_coupon = 0.016 # 1.6% yearly interest rate 
    r = 0.03
    sigma = 0.3
    T = 5.0
    N_steps = 12 * 5  # monthly steps for 5 years

    # Define coupon dates every 6 months: 6/12 = every 6th month
    step_every_6_months = int(N_steps / (T * 2))  # 2 coupons/year
    convertible_times = set(range(step_every_6_months, N_steps + 1, step_every_6_months))
    convertible_times.add(N_steps)  # Always at maturity

    # Binomial Tree Method with fine steps
    bond_price_binomial = price_convertible_bond(
        method='binomial',
        S0=S0, N=N, ell=ell, K_coupon=K_coupon,
        r=r, sigma=sigma, T=T, N_steps=N_steps,
        convertible_times=convertible_times,
        plot=True
    )
    print(f"Convertible Bond Price (Binomial Fine Grid): {bond_price_binomial:.4f}")

