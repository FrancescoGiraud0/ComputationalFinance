# Computational Finance

Homework and final project for *Computational Finance* (WISL529), taught by
Prof. Lech Grzelak at Utrecht University, spring 2025.

The course covers the numerical pricing of financial derivatives: stochastic
calculus, Monte Carlo simulation and its discretisation schemes, implied
volatility, Fourier methods (COS) and simulation-based pricing of options with
early exercise. Each assignment asks for a report in LaTeX and plain Python
scripts, so every folder contains the `.tex` sources, the report as PDF, the
figures and the code that produced them.

## Contents

| Folder | Topics |
|---|---|
| `Exercise_set_01/` | Brownian motion, Itô calculus, Monte Carlo for Black-Scholes |
| `Exercise_set_02/` | Time-dependent volatility, implied volatility, COS method, Euler and Milstein, barrier options, two-asset options |
| `13_ConvertibleBonds/` | Final project: convertible bond pricing with regression Monte Carlo |

### Exercise set 1

1. The identity $\int_0^t W(s)\,ds = \int_0^t (t-s)\,dW(s)$: proof and numerical
   check at $t = 5$, with the convergence of the error.
2. Dynamics of $Y_1 = \mu S^2$ for a geometric Brownian motion (it is again a
   GBM), and Itô's formula for $e^{W(t)}$.
3. Effect of standardising the normal samples on Monte Carlo pricing of a call
   in the Black-Scholes model: with standardisation the estimates converge
   faster and vary less across seeds.
4. Variance of $X(t) = W(t) - \frac{t}{T}W(T-t)$ for $T = 10$, derived analytically
   and confirmed by simulation.

Report: `GiraudoF_HomeworkSet1_CF.pdf`. Code: `src/`.

### Exercise set 2

1. GBM with $\sigma(t) = 0.6 - 0.2e^{-1.5t}$: analytical call prices and the
   implied-volatility smile and skew across maturities.
2. The model $dS = \gamma\sqrt{S}\,dW$: characteristic function and recovery of
   the density with the COS method.
3. Euler and Milstein schemes for a European call with time-dependent
   $\sigma(t)$ and $r(t)$, for $N \in \{10^3, 10^4, 10^5\}$ paths and
   $m \in \{50, 100, 200, 400\}$ steps, plus up-and-out and up-and-in barrier
   options. With $K = 1.6 > B = 1.5$ the up-and-out call is worth zero, since a
   payoff requires crossing the barrier, and the up-and-in call coincides with
   the plain call.
4. Option on the maximum of two correlated assets with $\rho = \pm 0.9$, with
   confidence intervals: negative correlation spreads the outcomes and raises
   the expected maximum.

Report: `ComputationalFinance_Set03_GiraudoF.pdf`. Code: `src/`. The paper in
`references/` (Nadarajah, 2008) is the reference used for the density of the
model in exercise 2.

### Final project: convertible bonds

A convertible bond pays coupons and the notional at maturity, and lets the
holder convert it into $\ell$ shares at given dates. The project:

1. writes the backward induction for the value at each payment date, with the
   choice between holding and converting;
2. prices the bond with a Longstaff-Schwartz style regression Monte Carlo
   (polynomial regression on the stock price);
3. extends the code to conversion at arbitrary dates;
4. studies how the number of conversion dates affects the price.

Base case: $S_0 = 200$, notional 1000, $\ell = 3$, coupon 1.6%, $r = 3\%$,
$\sigma = 30\%$, $T = 5$ years, 10,000 paths with daily steps and a cubic
regression. The estimated price at issuance is 1064.92. Raising the
volatility from 24% to 45% moves it from about 1012 to 1222, and more
conversion dates give a higher price: conversion at any time is worth about 5%
more than conversion only at coupon dates.

Report sources: `main.tex` with the section files and `references.bib`. The
section files are included with paths from the root of the repository, so
compile `13_ConvertibleBonds/main.tex` from there (this is the Overleaf layout).
Code: `src/`, one script per task (`01_pricing_bonds.py`,
`02_MC_LinearRegression.py`, `03_04_MC_more_exercise_times.py`).
`13_ConvertibleBonds.pdf` is the assignment text.

## Running the code

The scripts are plain Python and write their figures with matplotlib. They use
`numpy`, `scipy`, `matplotlib` and, for the convertible bond, `scikit-learn`:

```
pip install numpy scipy matplotlib scikit-learn
python Exercise_set_01/src/Exercise03.py
```

## References

* F. A. Longstaff, E. S. Schwartz (2001). Valuing American options by
  simulation: a simple least-squares approach. *Review of Financial Studies*
  14(1), 113-147.
* J. C. Hull (2022). *Options, Futures, and Other Derivatives*. Pearson.
* B. Tuckman, A. Serrat (2011). *Fixed Income Securities*. Wiley.
