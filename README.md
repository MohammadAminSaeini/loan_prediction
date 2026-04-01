# Bank Personal Loan Prediction 🏦 📈

## 🎯 Project Overview
This project focuses on a binary classification problem for a retail bank. The goal is to predict whether a "liability customer" will accept a personal loan offer based on their demographic and financial profile. 

The primary challenge of this dataset is its **highly imbalanced nature**, where only **9.6%** of customers accepted the loan. This makes **F1-Score** and **Recall** much more critical metrics than simple Accuracy.

## 🛠️ Tech Stack
- **Language:** Python
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn (Pipeline, GridSearchCV, K-Fold)

## 📊 Dataset Description
The dataset consists of 5,000 observations with features such as:
- **Demographics:** Age, Family size, Education level.
- **Financials:** Annual Income, Monthly Credit Card Spending (CCAvg), Mortgage.
- **Bank Relationship:** Securities Account, CD Account, Online Banking, Credit Card usage.
- **Target Variable:** `Personal Loan` (0: Not accepted, 1: Accepted).

## 🧪 Methodology & Implementation

### 1. Data Preprocessing & Cleaning
- **Feature Engineering:** Cleaned the `CCAvg` column by converting string formats (e.g., '1/60') into float values.
- **Feature Selection:** Handled **Multicollinearity** by identifying a 0.99 correlation between `Age` and `Experience`.
- **Scaling:** Used `MinMaxScaler` within a Pipeline to ensure no data leakage during cross-validation.

### 2. Model Selection & Tuning
I implemented and compared several models using **GridSearchCV** to find the optimal hyperparameters:
- **Logistic Regression:** Adjusted with `class_weight='balanced'` to handle the minority class.
- **K-Nearest Neighbors (KNN):** Tuned `k` to find the best bias-variance tradeoff.
- **Complement Naive Bayes (CNB):** Tested for its specialized approach to imbalanced data.

### 3. Evaluation Metrics
Since the dataset is imbalanced, the models were evaluated based on:
- **F1-Score:** To balance Precision and Recall.
- **Confusion Matrix:** To visualize Type I and Type II errors.

## 📈 Key Insights
- **Income & Education:** These were found to be the strongest predictors for loan acceptance.
- **The Imbalance Trap:** Standard models often achieved high accuracy (>90%) but failed to identify the actual loan takers (low Recall). Weighting the classes was essential for a usable model.
- **Model Stability:** Removing highly correlated features (like Experience) improved the interpretability of the Logistic Regression coefficients.

