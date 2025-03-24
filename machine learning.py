# Step 1: Import necessary libraries
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split  # To split the data into training and testing sets
from sklearn.linear_model import LinearRegression  # A simple linear regression model
from sklearn.tree import DecisionTreeRegressor  # A decision tree model for regression
from sklearn.preprocessing import StandardScaler  # To scale the features (normalize them)
from sklearn.metrics import mean_absolute_error, mean_squared_error  # To evaluate our model's performance

# Ensure 'micropip' error is avoided
if 'micropip' in sys.modules:
    del sys.modules['micropip']

# Step 2: Load the California Housing Dataset
# Use a local dataset if available to avoid URL errors
file_path = 'california_housing.csv'

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
else:
    from sklearn.datasets import fetch_california_housing
    housing = fetch_california_housing()
    X = housing.data       # Features (input variables)
    y = housing.target     # Target (house prices)

    # Create a DataFrame for easier manipulation
    df = pd.DataFrame(X, columns=housing.feature_names)
    df['Price'] = y

    # Save dataset locally to avoid re-fetching
    df.to_csv(file_path, index=False)

X = df.drop('Price', axis=1)
y = df['Price']

print(df.head())  # Display the first five rows of the dataset

# Step 4: Check for any missing values (there are none in this dataset)
print(df.isnull().sum())  # This checks for missing values in each column. In our case, it should print 0 for all columns.

# Step 5: Scale the features (Normalize them)
# We use StandardScaler to scale our features so they have a mean of 0 and a standard deviation of 1.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Apply the scaling to our feature data

# Step 6: Split the data into training and testing sets
# We’ll use 80% of the data for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 7: Choose a model to train (Linear Regression and Decision Tree)
# Let's try two models to see which one performs better: Linear Regression and Decision Tree

# Linear Regression Model
lin_model = LinearRegression()  # Create a linear regression model
lin_model.fit(X_train, y_train)  # Train the model using the training data
y_pred_lin = lin_model.predict(X_test)  # Use the model to predict house prices on the test set

# Decision Tree Model
tree_model = DecisionTreeRegressor(max_depth=5)  # Create a decision tree model with a maximum depth of 5
tree_model.fit(X_train, y_train)  # Train the decision tree on the training data
y_pred_tree = tree_model.predict(X_test)  # Use the decision tree to predict house prices

# Step 8: Evaluate the models (How well did they do?)
# We'll use two common metrics: Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE)
# These metrics help us understand how close our predictions are to the actual house prices.

# Linear Regression Evaluation
mae_lin = mean_absolute_error(y_test, y_pred_lin)  # Calculate MAE for Linear Regression
rmse_lin = np.sqrt(mean_squared_error(y_test, y_pred_lin))  # Manually calculate RMSE for Linear Regression

# Decision Tree Evaluation
mae_tree = mean_absolute_error(y_test, y_pred_tree)  # Calculate MAE for Decision Tree
rmse_tree = np.sqrt(mean_squared_error(y_test, y_pred_tree))  # Manually calculate RMSE for Decision Tree

# Print the evaluation results
print(f"Linear Regression - MAE: {mae_lin}, RMSE: {rmse_lin}")
print(f"Decision Tree - MAE: {mae_tree}, RMSE: {rmse_tree}")

# Step 9: Visualize the results
# We’ll create scatter plots to show how well the predicted prices match the actual prices.

# Create a plot with two subplots: one for Linear Regression, one for Decision Tree
plt.figure(figsize=(12, 6))

# Predicted vs Actual for Linear Regression
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_lin, color='blue')  # Scatter plot for Linear Regression
plt.xlabel("Actual Prices")  # Label the x-axis
plt.ylabel("Predicted Prices")  # Label the y-axis
plt.title("Linear Regression: Actual vs Predicted")  # Title for the plot

# Predicted vs Actual for Decision Tree
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_tree, color='green')  # Scatter plot for Decision Tree
plt.xlabel("Actual Prices")  # Label the x-axis
plt.ylabel("Predicted Prices")  # Label the y-axis
plt.title("Decision Tree: Actual vs Predicted")  # Title for the plot

plt.tight_layout()  # Adjust the layout so everything fits nicely
plt.show()  # Show the plot

# Step 10: Residual Plot
# Residuals are the differences between the actual and predicted values.
# If the model is good, these should be randomly scattered around zero.

# Residual plot for Linear Regression
residuals_lin = y_test - y_pred_lin  # Calculate residuals for Linear Regression
plt.figure(figsize=(8, 6))
plt.scatter(y_pred_lin, residuals_lin, color='blue')  # Scatter plot of predicted vs residuals
plt.axhline(y=0, color='r', linestyle='--')  # Add a horizontal line at y=0
plt.xlabel("Predicted Prices")  # Label the x-axis
plt.ylabel("Residuals")  # Label the y-axis
plt.title("Linear Regression Residual Plot")  # Title for the plot
plt.show()

# Residual plot for Decision Tree
residuals_tree = y_test - y_pred_tree  # Calculate residuals for Decision Tree
plt.figure(figsize=(8, 6))
plt.scatter(y_pred_tree, residuals_tree, color='green')  # Scatter plot of predicted vs residuals
plt.axhline(y=0, color='r', linestyle='--')  # Add a horizontal line at y=0
plt.xlabel("Predicted Prices")  # Label the x-axis
plt.ylabel("Residuals")  # Label the y-axis
plt.title("Decision Tree Residual Plot")  # Title for the plot
plt.show()
