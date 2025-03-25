# predating-of-Home-prices-using-Machine-Learning# 🏠 House Price Prediction using Machine Learning

## 📌 Project Overview
This project focuses on predicting **house prices** using a **Machine Learning model**. The goal is to develop a predictive model using features such as square footage, number of bedrooms, and other relevant factors. This task is part of my Machine Learning internship at **IBI**.

## 📊 Dataset
We used the **Boston Housing Dataset** (or a similar dataset) that includes various features influencing house prices. For demonstration, the **Diabetes Dataset** was used as a placeholder during model development.

### Features:
- CRIM: Per capita crime rate
- ZN: Proportion of residential land zoned for large lots
- RM: Average number of rooms per dwelling
- LSTAT: % lower status population
- ... and more

## 📈 Project Workflow
1. **Dataset Selection**
   - Loaded dataset using `scikit-learn`.
   - Explored dataset to understand feature distributions.

2. **Data Preprocessing**
   - Handled missing values (if any) using imputation.
   - Normalized and scaled features where necessary.
   - Split data into **80% training** and **20% testing**.

3. **Model Selection & Training**
   - Implemented **Linear Regression** and **Decision Tree** models.
   - Tuned **Decision Tree** hyperparameters using `GridSearchCV`.

4. **Model Evaluation**
   - Evaluated performance using:
     - **Mean Absolute Error (MAE)**
     - **Root Mean Squared Error (RMSE)**
   - Compared model predictions to actual values using visualization.

5. **Hyperparameter Tuning**
   - Tuned `max_depth`, `min_samples_split`, and `min_samples_leaf` for the Decision Tree model.

## 📊 Results
- **Best Model:** Decision Tree with optimal hyperparameters
- **Performance Metrics:**
  - Mean Absolute Error (MAE): `XX.XX`
  - Root Mean Squared Error (RMSE): `XX.XX`

## 📊 Visualizations
1. **Actual vs. Predicted Prices**: Scatter plot showing model predictions vs actual values.
2. **Residual Plot**: Visualizes model errors to evaluate fit quality.

## 🛠️ How to Run the Project
1. Clone the repository:

    ```bash
    git clone <your-repo-link>
    cd house-price-prediction
    ```

2. Set up the environment:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the main script:

    ```bash
    python main.py
    ```

## 📚 Dependencies
- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## 🔍 Future Work
- Try advanced models (Random Forest, XGBoost).
- Deploy the model as a web API.
- Enhance feature engineering with location-based information.

## 📢 Connect
Feel free to check out my LinkedIn post for more insights and updates!

[🔗 Linkedin]([R. Gopinathreddy Reddyvari](https://www.linkedin.com/in/r-gopinathreddy-reddyvari-8a0a1a324?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BN1xjvhlfSDyebtxa1m3KHA%3D%3D)](#) | [📂 [GitHub Repository](https://lnkd.in/eMMGD2nc)](#)

---

✅ **Completed as part of the IBI Machine Learning Internship**

