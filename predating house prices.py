# Importing the necessary libraries
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Fix for micro_pip in some environments
if 'micropip' in sys.modules:
    del sys.modules['micropip']

# Connect or Load the California Housing dataset
file_path = 'california_housing.csv'

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
else:
    from sklearn.datasets import fetch_california_housing
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names) # type: ignore
    df['Price'] = housing.target # type: ignore
    df.to_csv(file_path, index=False)

# Splitting features and target
X = df.drop('Price', axis=1)
y = df['Price']

print("Preview of the dataset:")
print(df.head())

# Check if there are missing values
print("\nMissing values check:")
print(df.isnull().sum())

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Linear Regression
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
y_pred_lin = lin_model.predict(X_test)

# Train Decision Tree Regressor
tree_model = DecisionTreeRegressor(max_depth=5)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)

# Evaluation function
def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{name} -> MAE: {mae:.3f}, RMSE: {rmse:.3f}")

# Compare both models
evaluate("Linear Regression", y_test, y_pred_lin)
evaluate("Decision Tree", y_test, y_pred_tree)

# Plot predictions
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_lin, color='blue', alpha=0.5)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Linear Regression")

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_tree, color='green', alpha=0.5)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Decision Tree")

plt.tight_layout()
plt.show()

# Residual plots
def plot_residuals(name, y_true, y_pred, color):
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, residuals, color=color, alpha=0.6)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title(f"{name} Residuals")
    plt.show()

plot_residuals("Linear Regression", y_test, y_pred_lin, 'blue')
plot_residuals("Decision Tree", y_test, y_pred_tree, 'green')
