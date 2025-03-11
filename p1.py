import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create the dataset based on the provided data
data = pd.DataFrame({
'Age': [23, 23, 27, 27, 39, 41, 47, 49, 50, 52, 54, 54, 56, 57, 58, 58, 60, 61],
'%Fat': [7.8, 9.5, 17.8, 25.9, 26.5, 27.2, 27.4, 28.8, 30.2, 31.2, 31.4, 32.9, 33.4, 34.1, 34.6, 35.7, 41.2, 42.5]
})

# Save the dataset to a CSV file
data.to_csv('Age_Fat.csv', index=False)

# Load the dataset
print("Dataset Loaded from Age_Fat.csv")
data = pd.read_csv('Age_Fat.csv')

# Preview the dataset
print("Dataset Preview:")
print(data.head())

# Numerical column: %Fat
numerical_column = '%Fat'


data_num = data[numerical_column]

# Compute statistics
mean_val = data_num.mean()
median_val = data_num.median()
mode_val = data_num.mode()[0]
std_dev = data_num.std()
variance = data_num.var()
range_val = data_num.max() - data_num.min()

# Print statistics
print("\nStatistics for Numerical Column:")
print(f"Mean: {mean_val}")
print(f"Median: {median_val}")
print(f"Mode: {mode_val}")
print(f"Standard Deviation: {std_dev}")
print(f"Variance: {variance}")
print(f"Range: {range_val}")

# Generate a histogram
plt.figure(figsize=(8, 5))
plt.hist(data_num, bins=10, color='skyblue', edgecolor='black')
plt.title(f"Histogram of {numerical_column}")
plt.xlabel(numerical_column)
plt.ylabel("Frequency")
plt.show()

# Generate a boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(x=data_num, color='lightgreen')
plt.title(f"Boxplot of {numerical_column}")
plt.show()

# Identify outliers using IQR



q1 = data_num.quantile(0.25)
q3 = data_num.quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers = data_num[(data_num < lower_bound) | (data_num > upper_bound)]

print("\nOutliers:")
print(outliers)

# Add a categorical column (Age Group) for demonstration purposes
def age_group(age):
  if age < 30:
    return 'Young' 
  elif 30 <= age <= 50:
    return 'Middle-aged' 
  else:
    return 'Older' 
data['Age Group'] = data['Age'].apply(age_group)

# Categorical column: Age Group
categorical_column = 'Age Group' 
data_cat = data[categorical_column]

# Compute frequency of each category
category_counts = data_cat.value_counts()

print("\nCategory Frequencies:")
print(category_counts)

# Bar chart for the categorical column
plt.figure(figsize=(8, 5))
category_counts.plot(kind='bar', color='coral', edgecolor='black')


plt.title(f"Bar Chart of {categorical_column}")
plt.xlabel(categorical_column)
plt.ylabel("Frequency")
plt.show()

# Pie chart for the categorical column
plt.figure(figsize=(8, 5))
category_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
plt.title(f"Pie Chart of {categorical_column}")
plt.ylabel("") # Remove y-axis label
plt.show()