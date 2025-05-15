import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def local_weight(xq, X, y, tau):
    w = np.exp(-((X - xq)**2) / (2 * tau**2))
    X_b = np.c_[np.ones(len(X)), X]
    W = np.diag(w)
    theta = np.linalg.pinv(X_b.T @ W @ X_b) @ (X_b.T @ W @ y)
    return np.array([1, xq]) @ theta

df = pd.read_csv("house_price.csv")
X, y = df["size"].values, df["price"].values
tau, xq = 100, 1600
predicted_tip = local_weight(xq, X, y, tau)
print(f"predicted price: {predicted_tip:.2f}")

Xr = np.linspace(X.min(), X.max(), 100)
yp = np.array([local_weight(x, X, y, tau) for x in Xr])

plt.scatter(X, y, color='red', alpha=0.5, label="actual data")
plt.plot(Xr, yp, color='blue', label="local weighted regression")
plt.scatter([xq], [predicted_tip], color='green', marker='o', label="prediction")
plt.xlabel("size")
plt.ylabel("price")
plt.legend()
plt.show()
