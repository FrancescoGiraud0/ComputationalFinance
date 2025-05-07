"""
Python version 3.12.6
Computational Finance - Exercise Set 2
Exercise 3, Monte Carlo Simulation of European and Barrier Options
author: Francesco Giraudo - VU Amsterdam
"""
import numpy as np
import matplotlib.pyplot as plt

def sigma_t(t):
    return 0.6 - 0.2 * np.exp(-1.5 * t)

def r_t(t):
    return 0.05 if t < 2 else 0.08

def simulate_paths(S0, T, N, M, scheme='euler', seed=None):
    if seed is not None:
        np.random.seed(seed)
    dt = T / M
    times = np.linspace(0, T, M + 1)
    S = np.full((N, M + 1), S0)
    dW = np.random.normal(0.0, 1.0, (N, M))

    # Standardize the Wiener increments
    dW = (dW - np.mean(dW, axis=0)) / np.std(dW, axis=0) * np.sqrt(dt)

    for i in range(M):
        t = times[i]
        sig = sigma_t(t)
        r = r_t(t)
        if scheme == 'euler':
            S[:, i + 1] = S[:, i] + r * S[:, i] * dt + sig * S[:, i] * dW[:, i]
        elif scheme == 'milstein':
            S[:, i + 1] = S[:, i] + r * S[:, i] * dt + sig * S[:, i] * dW[:, i] + \
                          0.5 * sig * S[:, i] * (dW[:, i]**2 - dt) * sig
    return S, times

def price_european_call(S_paths, K, T, r_avg):
    ST = S_paths[:, -1]
    payoff = np.maximum(ST - K, 0)
    return np.exp(-r_avg * T) * np.mean(payoff)

def price_barrier_options(S_paths, K, T, r_avg, B):
    ST = S_paths[:, -1].copy()
    crossed = np.any(S_paths>B, axis=1)
    call_payoff = np.maximum(ST - K, 0)
    up_and_out = np.where(~crossed, call_payoff.copy(), 0)
    up_and_in  = np.where(crossed,  call_payoff.copy(), 0)

    return np.exp(-r_avg * T) * np.mean(up_and_out), np.exp(-r_avg * T) * np.mean(up_and_in)

def run_simulation():
    S0 = 1.0
    K = 1.6
    T = 4.0
    B = 1.5
    seed = 42
    path_counts = [1000, 10000, 100000]
    step_counts = [50, 100, 200, 400]
    
    # Create a list to store all results
    results = []

    for scheme in ['euler', 'milstein']:
        for N in path_counts:
            for M in step_counts:
                S_paths, times = simulate_paths(S0, T, N, M, scheme, seed)
                r_vals = np.array([r_t(t) for t in times])
                r_avg = np.mean(r_vals)

                european = price_european_call(S_paths, K, T, r_avg)
                up_out, up_in = price_barrier_options(S_paths, K, T, r_avg, B)

                # Store results
                results.append({
                    'scheme': scheme,
                    'N': N,
                    'M': M,
                    'european': european,
                    'up_out': up_out,
                    'up_in': up_in
                })

    # Print all results in a formatted table
    print("\nSimulation Results:")
    print("-" * 80)
    print(f"{'Scheme':<8} {'Paths':<7} {'Steps':<6} {'European':>12} {'Up-Out':>12} {'Up-In':>12}")
    print("-" * 80)
    
    for scheme in ['euler', 'milstein']:
        print(f"\n{scheme.upper()} SCHEME:")
        print("-" * 80)
        for result in [r for r in results if r['scheme'] == scheme]:
            print(f"{result['scheme']:<8} {result['N']:<7} {result['M']:<6} "
                  f"{result['european']:>12.5f} {result['up_out']:>12.5f} "
                  f"{result['up_in']:>12.5f}")
        print("-" * 80)

#%%        
if __name__ == "__main__":
    run_simulation()
