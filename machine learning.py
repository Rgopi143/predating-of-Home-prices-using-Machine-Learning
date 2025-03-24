# Step 1: Import the necessary libraries

# System-related modules for handling files and system-level tasks
import sys
import os
import numpy as np  # For numerical operations
import pandas as pd  # For handling datasets

# Visualization library to plot graphs and understand model performance
import matplotlib.pyplot as plt

# Machine Learning tools from scikit-learn
from sklearn.model_selection import train_test_split  # To split the dataset into training and testing sets
from sklearn.linear_model import LinearRegression  # Linear Regression model for prediction
from sklearn.tree import DecisionTreeRegressor  # Decision Tree model for prediction
from sklearn.preprocessing import StandardScaler  # To normalize feature values for better model performance
from sklearn.metrics import mean_absolute_error, mean_squared_error  # To measure model accuracy

# Avoiding potential issues with 'micropip' in certain environments
if 'micropip' in sys.modules:
    del sys.modules['micropip']

# Step 2: Load the California Housing Dataset

# Check if the dataset already exists locally to save time on downloading
file_path = 'california_housing.csv'

if os.path.exists(file_path):
    # If the dataset is found locally, load it to save time
    df = pd.read_csv(file_path)
else:
    # Otherwise, download the dataset from scikit-learn
    from sklearn.datasets import fetch_california_housing
    housing = fetch_california_housing()

    # Store the dataset in a DataFrame for easier manipulation
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['Price'] = housing.target

    # Save a local copy of the dataset for future use
    df.to_csv(file_path, index=False)

# Separate features (X) and target (Y) for model training
X = df.drop('Price', axis=1)  # Features (input variables)
Y = df['Price']  # Target variable (house prices)

# Display a quick preview of the dataset
print("First 5 rows of the dataset:")
print(df.head())

# Step 3: Check for missing values

# Ensure the dataset has no missing values, as they can cause errors
print("\nChecking for missing values:")
print(df.isnull().sum())  # Outputs 0 if there are no missing values

# Step 4: Scale the features

# Feature scaling helps models converge faster and improves accuracy
scaler = StandardScaler()  # Standardizes data to have mean=0 and std=1
X_scaled = scaler.fit_transform(X)  # Apply scaling to the feature set

# Step 5: Split the dataset

# Divide the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# Step 6: Train two models (Linear Regression and Decision Tree)

# 1. Linear Regression Model
lin_model = LinearRegression()  # Create a linear regression instance
lin_model.fit(X_train, y_train)  # Train the model
y_pred_lin = lin_model.predict(X_test)  # Make predictions on the test set

# 2. Decision Tree Model
# Limit tree depth to prevent overfitting to the training data
tree_model = DecisionTreeRegressor(max_depth=5)  # Create a Decision Tree instance
tree_model.fit(X_train, y_train)  # Train the model
y_pred_tree = tree_model.predict(X_test)  # Make predictions on the test set

# Step 7: Evaluate the models

def evaluate_model(model_name, y_true, y_pred):
    """Calculate and display model performance metrics."""
    mae = mean_absolute_error(y_true, y_pred)  # Mean Absolute Error
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # Root Mean Squared Error
    print(f"{model_name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}")

# Compare model performance
evaluate_model("Linear Regression", y_test, y_pred_lin)
evaluate_model("Decision Tree", y_test, y_pred_tree)

# Step 8: Visualize the predictions

plt.figure(figsize=(12, 6))  # Set figure size

# Linear Regression: Compare actual vs. predicted prices
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_lin, color='blue')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Linear Regression: Actual vs Predicted")

# Decision Tree: Compare actual vs. predicted prices
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_tree, color='green')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Decision Tree: Actual vs Predicted")

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()

# Step 9: Analyze residuals (errors between actual and predicted values)

def plot_residuals(model_name, y_true, y_pred, color):
    """Visualize the residuals to understand model error distribution."""
    residuals = y_true - y_pred  # Calculate the differences
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, color=color)  # Scatter plot of residuals
    plt.axhline(y=0, color='r', linestyle='--')  # Reference line at zero
    plt.xlabel("Predicted Prices")
    plt.ylabel("Residuals")
    plt.title(f"{model_name} Residual Plot")
    plt.show()

# Generate residual plots to check error patterns
plot_residuals("Linear Regression", y_test, y_pred_lin, 'blue')
plot_residuals("Decision Tree", y_test, y_pred_tree, 'green')

# Summary:
# - We loaded and explored the California housing dataset.
# - We trained two models (Linear Regression and Decision Tree).
# - We evaluated both models using MAE and RMSE metrics.
# - We visualized predictions and residuals to understand model performance.

# Instructions to upload the code to GitHub
# 1. Initialize a Git repository (run these commands in terminal)
#    git init
# 2. Add your files to the repository
#    git add .
# 3. Commit the changes
#    git commit -m "Add house price prediction script"
# 4. Connect to a GitHub repository (replace URL with your repo URL)
#    git remote add origin https://github.com/yourusername/your-repo.git
# 5. Push the code to GitHub
#    git branch -M main
#    git push -u origin main

# Ensure your repository includes the following:
# - This Python script
# - Dataset (if permissible)
# - README.md explaining the project
# - requirements.txt listing necessary libraries (e.g., pandas, numpy, scikit-learn, matplotlib)
