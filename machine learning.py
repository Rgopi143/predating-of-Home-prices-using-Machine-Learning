# Step 1: Import the necessary libraries

# System and basic data manipulation
import sys
import os
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt

# Machine Learning tools from scikit-learn
from sklearn.model_selection import train_test_split  # To split the dataset
from sklearn.linear_model import LinearRegression  # Linear Regression model
from sklearn.tree import DecisionTreeRegressor  # Decision Tree model
from sklearn.preprocessing import StandardScaler  # For feature scaling
from sklearn.metrics import mean_absolute_error, mean_squared_error  # Model evaluation metrics

# Ensure 'micropip' error is avoided (for specific environments)
if 'micropip' in sys.modules:
    del sys.modules['micropip']

# Step 2: Load the California Housing Dataset

# Check if a local copy of the dataset exists to save time and prevent refetching
file_path = 'california_housing.csv'

if os.path.exists(file_path):
    # Load the dataset from the local CSV file
    df = pd.read_csv(file_path)
else:
    # Fetch the dataset from scikit-learn if not available locally
    from sklearn.datasets import fetch_california_housing
    housing = fetch_california_housing()

    # Create a DataFrame for better handling
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['Price'] = housing.target

    # Save a local copy for future use
    df.to_csv(file_path, index=False)

# Separate features (X) and target (y)
X = df.drop('Price', axis=1)
Y = df['Price']

# Display a preview of the dataset
print("First 5 rows of the dataset:")
print(df.head())

# Step 3: Check for missing values

# Verify if there are any missing values in the dataset
print("\nChecking for missing values:")
print(df.isnull().sum())  # Should print 0 for all columns if no missing values

# Step 4: Scale the features

# Standardize the features to have a mean of 0 and standard deviation of 1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Split the dataset

# Use 80% of the data for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# Step 6: Train two models (Linear Regression and Decision Tree)

# 1. Linear Regression Model
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
y_pred_lin = lin_model.predict(X_test)

# 2. Decision Tree Model
# We limit the depth of the tree to prevent overfitting
tree_model = DecisionTreeRegressor(max_depth=5)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)

# Step 7: Evaluate the models

def evaluate_model(model_name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{model_name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}")

# Evaluate both models
evaluate_model("Linear Regression", y_test, y_pred_lin)
evaluate_model("Decision Tree", y_test, y_pred_tree)

# Step 8: Visualize the predictions

plt.figure(figsize=(12, 6))

# Linear Regression: Actual vs Predicted
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_lin, color='blue')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Linear Regression: Actual vs Predicted")

# Decision Tree: Actual vs Predicted
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_tree, color='green')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Decision Tree: Actual vs Predicted")

plt.tight_layout()
plt.show()

# Step 9: Analyze residuals (errors between actual and predicted values)

def plot_residuals(model_name, y_true, y_pred, color):
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, color=color)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("Predicted Prices")
    plt.ylabel("Residuals")
    plt.title(f"{model_name} Residual Plot")
    plt.show()

# Residual plots for both models
plot_residuals("Linear Regression", y_test, y_pred_lin, 'blue')
plot_residuals("Decision Tree", y_test, y_pred_tree, 'green')

# Summary:
# - We loaded and explored the California housing dataset.
# - We trained two models (Linear Regression and Decision Tree).
# - We evaluated both models using MAE and RMSE metrics.
# - We visualized predictions and residuals to understand model performance.
