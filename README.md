
# Cafe Sales Analysis and Prediction

## Project Overview

This project involves a comprehensive analysis of cafe sales data, including data cleaning, exploratory data analysis (EDA), and machine learning model development for both regression (predicting 'Total Spent') and classification (predicting 'high_spend' vs. 'low_spend'). The goal is to understand sales patterns, identify key factors influencing spending, and build robust predictive models.
The dataset was taken from kaggle  https://www.kaggle.com/datasets/ahmedmohamed2003/cafe-sales-dirty-data-for-cleaning-training
"""

## Dataset

The dataset `dirty_cafe_sales.csv` contains various transaction details from a cafe, including:
- `Transaction ID`: Unique identifier for each transaction.
- `Item`: The item purchased.
- `Quantity`: Number of items purchased.
- `Price Per Unit`: Price of a single unit of the item.
- `Total Spent`: Total amount spent on the transaction (target variable for regression).
- `Payment Method`: Method used for payment.
- `Location`: Where the purchase was made (In-store/Takeaway).
- `Transaction Date`: Date of the transaction.

## Project Structure

- `app.py`: Streamlit application for interactive prediction.
- `cafe_sales_analysis.ipynb`:Colab notebook containing the full data analysis, EDA, and model development process.
- `tuned_random_forest_regressor.pkl`: Trained and tuned Random Forest Regressor model saved using `joblib`.
- `dataset.csv`: The cleaned dataset used for model training and application (derived from `dirty_cafe_sales.csv`).
- `README.md`: This file, providing an overview of the project.
- `Requirements.txt`:Here all of the libraries are stored.
## Key Steps


### 1. Data Cleaning and Preprocessing

- **Handling Missing Values**: Missing values in 'Item', 'Quantity', 'Price Per Unit', 'Total Spent', 'Payment Method', 'Location', and 'Transaction Date' were imputed using mode or appropriate methods.
- **Handling Inconsistent Entries**: 'ERROR' and 'UNKNOWN' values in categorical columns were replaced with the mode or a consistent category.
- **Data Type Conversion**: Numerical columns like 'Quantity', 'Price Per Unit', and 'Total Spent' were converted to appropriate numeric types.
- **Outlier Treatment**: Outliers in 'Total Spent' were identified and managed using the IQR method.
- **Feature Engineering**: 'Year', 'Month', and 'Day' were extracted from 'Transaction Date'. 'Transaction ID', original 'Item', and 'Transaction Date' columns were dropped.
- **One-Hot Encoding**: Categorical features ('Payment Method', 'Location', 'item') were one-hot encoded for machine learning models.

### 2. Exploratory Data Analysis (EDA)

- **Total Spent Distribution**: Revealed a right-skewed distribution.
- **Average Sales Analysis**:
    - Minimal difference in average 'Total Spent' across 'Payment Method' and 'Location'.
    - Significant differences in average 'Total Spent' by 'item', with 'Salad', 'Smoothie', and 'Sandwich' being high-value items.
- **Monthly Sales Trend**: Identified fluctuations, with February and November being peak sales months, and July and January being lower.

### 3. Model Development - Regression (Predicting 'Total Spent')

- **Target Variable**: `Total Spent` (continuous).
- **Models Explored**: Linear Regression (Ridge) and Random Forest Regressor.
- **Feature Selection**: Highly correlated features (`Quantity`, `Price Per Unit`) were initially used for Linear Regression. All features were used for Random Forest.
- **Evaluation Metrics**: R-squared, Mean Squared Error (MSE).
- **Key Findings**:
    - **Linear Regression (Ridge)**: Achieved R-squared ~0.74-0.75, MSE ~7.12.
    - **Random Forest Regressor (Default)**: Achieved R-squared ~0.86, MSE ~3.80.
    - **Random Forest Regressor (Tuned)**: Optimal hyperparameters found via `RandomizedSearchCV` (`n_estimators=100`, `max_features=0.8`, `max_depth=10`). Performance was R-squared ~0.86, MSE ~3.78, showing marginal improvement over default.
    - **Conclusion**: The **Tuned Random Forest Regressor** is the best model for regression, offering high accuracy and robustness.

### 4. Model Development - Classification (Predicting 'high_spend')

- **Target Variable**: `high_spend` (binary: 1 if `Total Spent` > median, 0 otherwise). Median `Total Spent` was $6.00.
- **Models Explored**: Logistic Regression, K-Nearest Neighbors (KNN), and Naive Bayes.
- **Features Used**: `Quantity`, `Price Per Unit`, and one-hot encoded categorical features.
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score (with K-fold cross-validation).
- **Key Findings**:
    - **Logistic Regression**: Achieved Accuracy ~0.84-0.85, F1-score ~0.84-0.85.
    - **K-Nearest Neighbors (KNN)**: Achieved Accuracy ~0.93-0.94, F1-score ~0.93.
    - **Naive Bayes (Default)**: Achieved Accuracy ~0.67-0.68, F1-score ~0.73, showing high recall for 'high spend' but lower precision.
    - **Naive Bayes (Tuned)**: After tuning `var_smoothing` (optimal `0.5`), achieved Accuracy ~0.93-0.94, F1-score ~0.94.
    - **Conclusion**: Both **KNN** and **Tuned Naive Bayes** are excellent choices for classification, significantly outperforming Logistic Regression. Tuned Naive Bayes offers comparable performance with potential computational efficiency benefits.

## Best Model for Deployment

- **For Regression (Predicting 'Total Spent')**: The `tuned_random_forest_regressor.pkl` is selected due to its superior R-squared (0.86) and low MSE (3.78). This model is saved and used in the Streamlit application.

## Run the Streamlit Application (`app.py`)

1.  **necessary files that i have**:
    - `app.py`
    - `tuned_random_forest_regressor.pkl`
    - `dataset.csv` (the cleaned data used for encoding unique values)
    - `README.md` (this file)
2.## Streamlit Application (`app.py`) Documentation

This `app.py` file contains the code for a simple Streamlit web application designed to interactively predict the 'Total Spent' based on user inputs. It serves as a practical demonstration of the deployed machine learning model.

### Purpose

The primary goal of this application is to provide an intuitive interface for users (e.g., cafe staff, managers) to input details of a potential customer order and instantly receive a prediction of the total amount the customer is expected to spend. This helps in understanding the model's predictions in a real-world context.

### Functionality

The application performs the following key functions:

1.  **Model Loading**: It loads the pre-trained `tuned_random_forest_regressor.pkl` model using `joblib`, ensuring the predictive intelligence is available.
2.  **User Input Interface**: It presents various input fields to the user, allowing them to specify:
    *   `Quantity`: The number of items purchased (numeric input).
    *   `Price Per Unit`: The price of a single unit of the item (numeric input).
    *   `Payment Method`: Checkboxes for 'Credit Card' and 'Digital Wallet' (assuming 'Cash' is the default if neither is selected).
    *   `Location`: A checkbox for 'Takeaway' (assuming 'In-store' if not selected).
    *   `Item`: Checkboxes for different item types like 'Coffee', 'Cookie', 'Juice', 'Salad', 'Sandwich', 'Smoothie', 'Tea'.
3.  **Prediction Logic**: Upon clicking the 'Predict Total Spent' button, the application gathers all user inputs, constructs a Pandas DataFrame in the exact format expected by the trained model (including one-hot encoded features), and then uses the loaded model to make a prediction.
4.  **Result Display**: The predicted 'Total Spent' value is displayed clearly to the user, formatted to two decimal places.

### How it Works 
*   **Libraries**: Utilizes `streamlit` for the web interface, `joblib` for model deserialization, and `pandas` for data handling.
*   **Feature Mapping**: The boolean `True`/`False` values from Streamlit checkboxes are converted to `int` (1/0) to match the model's expected input format for one-hot encoded features.
*   **Model Integration**: The `model.predict()` method takes the prepared input DataFrame and returns the estimated `Total Spent`. 



3.  **Install Streamlit and other dependencies**:
    
    pip install streamlit pandas scikit-learn joblib
    

4.  **Run the application**:
    Navigate to the directory containing `app.py` in terminal and execute:

    streamlit run app.py
    

5.  **Access the application**:
    Streamlit will open a new tab in  web browser with the interactive application.

The Streamlit app will allow users to input transaction details and get a prediction for the 'Total Spent' using the deployed `tuned_random_forest_regressor.pkl` model.

Conclusion: Even though i could have used Tuned Naive Bayes for prediction in application  but instead i opted for linear regression with accuracy below KNN because i wanted output features to be continious not categorical one  but using KNN could be good approach as well.


Author,
Sujan aryal

