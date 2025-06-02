import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

def price_convertible_bond_regression_mc(S0, N, ell, K_coupon, r, sigma, T, 
                                       n_paths=100000, n_steps=60, 
                                       coupon_frequency=2, regression_degree=3,
                                       plot_paths=False, plot_exercise=False):
    """
    Price convertible bond using Regression Monte Carlo (Longstaff-Schwartz method)
    
    Parameters:
    -----------
    S0 : float - Initial stock price
    N : float - Face value of bond
    ell : float - Conversion ratio (shares per bond)
    K_coupon : float - Annual coupon rate
    r : float - Risk-free rate
    sigma : float - Stock volatility
    T : float - Time to maturity
    n_paths : int - Number of simulation paths
    n_steps : int - Number of time steps
    coupon_frequency : int - Coupons per year (2 for semiannual)
    regression_degree : int - Polynomial degree for regression
    plot_paths : bool - Whether to plot sample paths
    plot_exercise : bool - Whether to plot exercise boundary
    
    Returns:
    --------
    bond_price : float - Price of convertible bond
    """
    
    dt = T / n_steps
    discount_factor = np.exp(-r * dt)
    
    # Generate stock price paths using geometric Brownian motion
    np.random.seed(42)  # For reproducibility
    Z = np.random.standard_normal((n_paths, n_steps))
    
    # Stock price matrix: S[path, time_step]
    S = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = S0
    
    for t in range(1, n_steps + 1):
        S[:, t] = S[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t-1])
    
    # Determine coupon payment dates
    coupon_steps = []
    steps_per_coupon = int(n_steps / (T * coupon_frequency))
    for i in range(1, int(T * coupon_frequency) + 1):
        step = min(i * steps_per_coupon, n_steps)
        if step not in coupon_steps:
            coupon_steps.append(step)
    
    # Coupon amount (semiannual)
    coupon_payment = N * K_coupon / coupon_frequency
    
    # Initialize cash flow matrix
    cash_flows = np.zeros((n_paths, n_steps + 1))
    
    # Terminal condition at maturity
    conversion_value_T = ell * S[:, n_steps]
    cash_flows[:, n_steps] = np.maximum(N, conversion_value_T)
    
    # Track exercise decisions for analysis
    exercise_boundary = []
    exercise_times = []
    
    # Backward induction with regression
    for t in range(n_steps - 1, -1, -1):
        # Discount future cash flows
        cash_flows[:, t] = cash_flows[:, t + 1] * discount_factor
        
        # Add coupon if this is a coupon date
        if t in coupon_steps:
            cash_flows[:, t] += coupon_payment
        
        # Check conversion option at coupon dates and maturity
        if t in coupon_steps or t == n_steps:
            conversion_value = ell * S[:, t]
            
            # Identify in-the-money paths for regression
            itm_mask = conversion_value > cash_flows[:, t]
            
            if np.sum(itm_mask) > 0:  # Only regress if there are ITM paths
                # Prepare regression features (polynomial basis)
                X_itm = S[itm_mask, t].reshape(-1, 1)
                y_itm = cash_flows[itm_mask, t]
                
                # Create polynomial features
                poly_features = PolynomialFeatures(degree=regression_degree, include_bias=True)
                X_poly = poly_features.fit_transform(X_itm)
                
                # Fit regression to estimate continuation value
                reg = LinearRegression()
                reg.fit(X_poly, y_itm)
                
                # Predict continuation values for all ITM paths
                continuation_value = np.zeros(len(y_itm))
                if X_poly.shape[1] > 1:  # Avoid issues with constant features
                    continuation_value = reg.predict(X_poly)
                else:
                    continuation_value = np.full(len(y_itm), np.mean(y_itm))
                
                # Exercise decision: convert if conversion value > continuation value
                exercise_mask = conversion_value[itm_mask] > continuation_value
                
                # Update cash flows for exercised paths
                full_exercise_mask = np.zeros(n_paths, dtype=bool)
                full_exercise_mask[itm_mask] = exercise_mask
                cash_flows[full_exercise_mask, t] = conversion_value[full_exercise_mask]
                
                # Store exercise boundary information
                if np.sum(exercise_mask) > 0:
                    exercise_boundary.append(np.mean(S[full_exercise_mask, t]))
                    exercise_times.append(t * dt)
    
    # Calculate bond price as average of discounted cash flows at t=0
    bond_price = np.mean(cash_flows[:, 0])
    
    # Optional plotting
    if plot_paths:
        plot_sample_paths(S, T, n_steps, S0, N, ell, K_coupon, r, sigma)
    
    if plot_exercise and exercise_times:
        plot_exercise_boundary(exercise_times, exercise_boundary, T, S0)
    
    # Calculate some statistics
    final_conversion_rate = np.mean(ell * S[:, n_steps] > N)
    avg_stock_final = np.mean(S[:, n_steps])
    
    print(f"\n=== Regression Monte Carlo Results ===")
    print(f"Convertible Bond Price: {bond_price:.4f}")
    print(f"Face Value: {N:.2f}")
    print(f"Conversion Premium: {((bond_price/N - 1) * 100):.2f}%")
    print(f"Final Conversion Rate: {final_conversion_rate:.2%}")
    print(f"Average Final Stock Price: {avg_stock_final:.2f}")
    print(f"Number of Exercise Opportunities: {len(exercise_times)}")
    
    return bond_price

def plot_sample_paths(S, T, n_steps, S0, N, ell, K_coupon, r, sigma, n_sample=50):
    """Plot sample stock price paths"""
    times = np.linspace(0, T, n_steps + 1)
    
    plt.figure(figsize=(12, 8))
    
    # Plot sample paths
    for i in range(min(n_sample, S.shape[0])):
        plt.plot(times, S[i, :], alpha=0.6, linewidth=0.8)
    
    # Add conversion threshold line
    conversion_threshold = N / ell
    plt.axhline(y=conversion_threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Conversion Threshold = {conversion_threshold:.2f}')
    
    plt.title('Sample Stock Price Paths (Regression Monte Carlo)', fontsize=16)
    plt.xlabel('Time (years)', fontsize=14)
    plt.ylabel('Stock Price', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add parameter text box
    param_text = (
        f"Initial Stock Price: {S0:.2f}\n"
        f"Face Value: {N:.2f}\n"
        f"Conversion Ratio: {ell}\n"
        f"Coupon Rate: {K_coupon:.1%}\n"
        f"Risk-free Rate: {r:.1%}\n"
        f"Volatility: {sigma:.1%}\n"
        f"Maturity: {T} years"
    )
    
    plt.text(0.02, 0.98, param_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def plot_exercise_boundary(exercise_times, exercise_boundary, T, S0):
    """Plot the exercise boundary over time"""
    if not exercise_times:
        print("No exercise points to plot.")
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(exercise_times, exercise_boundary, 'ro-', linewidth=2, markersize=8)
    plt.axhline(y=S0, color='blue', linestyle='--', alpha=0.7, label=f'Initial Stock Price = {S0:.2f}')
    
    plt.title('Optimal Exercise Boundary', fontsize=16)
    plt.xlabel('Time (years)', fontsize=14)
    plt.ylabel('Stock Price at Exercise', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def sensitivity_analysis(base_params):
    """Perform sensitivity analysis on key parameters and plot."""
    
    print("\n=== Sensitivity Analysis ===")
    
    # Base case
    base_price = price_convertible_bond_regression_mc(**base_params)
    
    # Store results
    volatility_list = []
    bond_price_vol_list = []
    
    stock_price_list = []
    bond_price_stock_list = []
    
    interest_rate_list = []
    bond_price_rate_list = []
    
    # Sensitivity to volatility
    print(f"\nVolatility Sensitivity:")
    print(f"Base σ = {base_params['sigma']:.1%}: {base_price:.4f}")
    
    for sigma_mult in [0.8, 1.0, 1.2, 1.5]:
        params = base_params.copy()
        params['sigma'] *= sigma_mult
        price = price_convertible_bond_regression_mc(**params, plot_paths=False)
        volatility_list.append(params['sigma'])
        bond_price_vol_list.append(price)
        print(f"σ = {params['sigma']:.1%}: {price:.4f} (Δ = {price - base_price:+.4f})")
    
    # Sensitivity to stock price
    print(f"\nStock Price Sensitivity:")
    print(f"Base S0 = {base_params['S0']:.2f}: {base_price:.4f}")
    
    for S0_mult in [0.8, 1.0, 1.2, 1.5]:
        params = base_params.copy()
        params['S0'] *= S0_mult
        price = price_convertible_bond_regression_mc(**params, plot_paths=False)
        stock_price_list.append(params['S0'])
        bond_price_stock_list.append(price)
        print(f"S0 = {params['S0']:.2f}: {price:.4f} (Δ = {price - base_price:+.4f})")
    
    # Sensitivity to interest rate
    print(f"\nInterest Rate Sensitivity:")
    print(f"Base r = {base_params['r']:.1%}: {base_price:.4f}")
    
    for r_add in [-0.01, 0.0, 0.01, 0.02]:
        params = base_params.copy()
        params['r'] += r_add
        price = price_convertible_bond_regression_mc(**params, plot_paths=False)
        interest_rate_list.append(params['r'])
        bond_price_rate_list.append(price)
        print(f"r = {params['r']:.1%}: {price:.4f} (Δ = {price - base_price:+.4f})")
    
    # Now plot all three graphs in one figure
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Volatility plot
    axs[0, 0].plot(volatility_list, bond_price_vol_list, 'o-', color='blue')
    axs[0, 0].set_title('Bond Price vs. Volatility')
    axs[0, 0].set_xlabel('Volatility')
    axs[0, 0].set_ylabel('Bond Price')
    axs[0, 0].grid(True)

    # Stock price plot
    axs[0, 1].plot(stock_price_list, bond_price_stock_list, 'o-', color='green')
    axs[0, 1].set_title('Bond Price vs. Stock Price')
    axs[0, 1].set_xlabel('Stock Price')
    axs[0, 1].set_ylabel('Bond Price')
    axs[0, 1].grid(True)

    # Interest rate plot
    axs[1, 0].plot(interest_rate_list, bond_price_rate_list, 'o-', color='red')
    axs[1, 0].set_title('Bond Price vs. Interest Rate')
    axs[1, 0].set_xlabel('Interest Rate')
    axs[1, 0].set_ylabel('Bond Price')
    axs[1, 0].grid(True)

    # Remove the empty subplot
    fig.delaxes(axs[1, 1])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    base_parameters = {
        'S0': 200.0,          # Initial stock price
        'N': 1000.0,          # Face value
        'ell': 3,             # Conversion ratio
        'K_coupon': 0.016,    # Annual coupon rate (1.6%)
        'r': 0.03,            # Risk-free rate
        'sigma': 0.3,         # Volatility
        'T': 5.0,             # Maturity
        'n_paths': 10000,    # Number of simulation paths
        'n_steps': 252*5,        # Daily steps for 5 years
        'coupon_frequency': 2, # Semiannual coupons
        'regression_degree': 3, # Polynomial degree for regression
        'plot_paths': True,
        'plot_exercise': True
    }
    
    # Price the convertible bond
    print("Pricing Convertible Bond using Regression Monte Carlo...")
    bond_price = price_convertible_bond_regression_mc(**base_parameters)
    
    # Perform sensitivity analysis
    sensitivity_analysis({k: v for k, v in base_parameters.items() 
                         if k not in ['plot_paths', 'plot_exercise']})
    
    # Compare with straight bond value (without conversion option)
    straight_bond_value = base_parameters['N']  # Simplified for par value
    option_value = bond_price - straight_bond_value
    print(f"\n=== Option Value Analysis ===")
    print(f"Convertible Bond Price: {bond_price:.4f}")
    print(f"Straight Bond Value: {straight_bond_value:.4f}")
    print(f"Embedded Option Value: {option_value:.4f}")
    print(f"Option Value as % of Face: {(option_value/base_parameters['N'])*100:.2f}%")