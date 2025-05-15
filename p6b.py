import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

data = pd.read_csv("auto-mpg.csv")
data.replace({'horsepower': {'?': np.nan}}, inplace=True)
data.dropna(subset=['horsepower'], inplace=True)
data['horsepower'] = data['horsepower'].astype(float)

X = data[['horsepower']]
y = data['mpg']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, y_train)
y_pred = model.predict(X_test_poly)

plt.scatter(X_test, y_test, color='green', alpha=0.6, label='Actual MPG')
sorted_idx = X_test['horsepower'].argsort()
plt.plot(X_test.iloc[sorted_idx], y_pred[sorted_idx], color='orange', label='Polynomial Fit')
plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.title("Polynomial Regression: Horsepower vs MPG")
plt.legend()
plt.grid(True)
plt.show()
