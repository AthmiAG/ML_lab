import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset (update 'your_dataset.csv' with your actual file path)
file_path = "IRIS.csv" # Replace with your dataset file path
df = pd.read_csv(file_path)

# Extract two columns (update 'Column1' and 'Column2' with actual column names)
column1 = "petal_length" # Replace with the first column name
column2 = "petal_width" # Replace with the second column name
selected_data = df[[column1, column2]]

# Scatter plot of the selected columns
plt.figure(figsize=(8, 6))
plt.scatter(selected_data[column1], selected_data[column2], color="blue", alpha=0.7)
plt.title(f"Scatter Plot of {column1} vs {column2}")
plt.xlabel(column1)
plt.ylabel(column2)
plt.grid(True)
plt.show()

# Calculate Pearson correlation coefficient
correlation_coefficient = np.corrcoef(selected_data[column1], selected_data[column2])[0, 1]
print(f"Pearson Correlation Coefficient ({column1} vs {column2}):{correlation_coefficient}")

# Compute covariance matrix
cov_matrix = np.cov(selected_data[column1], selected_data[column2])


print("\nCovariance Matrix:")
print(cov_matrix)

# Compute correlation matrix
correlation_matrix = selected_data.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
plt.title("Correlation Matrix Heatmap")
plt.show()