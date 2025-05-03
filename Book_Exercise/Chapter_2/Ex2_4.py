import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np

# Download data
s1 = yf.download("AAPL", start="2023-01-01", end="2024-01-01")["Close"]
s2 = yf.download("XOM", start="2023-01-01", end="2024-01-01")["Close"]

# Calculate returns
r1 = ((s1-s1.shift(1))/s1.shift(1)).dropna()
r1 = r1["AAPL"]
r2 = ((s2-s2.shift(1))/s2.shift(1)).dropna()
r2 = r2["XOM"]

# Scatter plot
plt.scatter(r1, r2)
plt.xlabel("AAPL Returns")
plt.ylabel("XOM Returns")
plt.title("Scatter Plot of Daily Returns")
plt.grid(True)
plt.show()

# Print correlation
print("Correlation:", r1.corr(r2))


# Download data
s1 = yf.download("SHEL", start="2023-01-01", end="2024-01-01")["Close"]
s2 = yf.download("XOM", start="2023-01-01", end="2024-01-01")["Close"]

# Calculate returns
r1 = ((s1-s1.shift(1))/s1.shift(1)).dropna()
r1 = r1["SHEL"]
r2 = ((s2-s2.shift(1))/s2.shift(1)).dropna()
r2 = r2["XOM"]

# Scatter plot
plt.scatter(r1, r2)
plt.xlabel("SHEL Returns")
plt.ylabel("XOM Returns")
plt.title("Scatter Plot of Daily Returns")
plt.grid(True)
plt.show()

# Print correlation
print("Correlation:", r1.corr(r2))

# Download data
s1 = yf.download("SHEL", start="2023-01-01", end="2024-01-01")["Close"]
s2 = yf.download("BP", start="2023-01-01", end="2024-01-01")["Close"]

# Calculate returns
r1 = ((s1-s1.shift(1))/s1.shift(1)).dropna()
r1 = r1["SHEL"]
r2 = ((s2-s2.shift(1))/s2.shift(1)).dropna()
r2 = r2["BP"]

# Scatter plot
plt.scatter(r1, r2)
plt.xlabel("SHEL Returns")
plt.ylabel("BP Returns")
plt.title("Scatter Plot of Daily Returns")
plt.grid(True)
plt.show()

# Print correlation
print("Correlation:", r1.corr(r2))

