Bank Personal Loan Prediction 🏦 📈
🎯 Project Overview
This project aims to predict whether a liability customer of a bank will accept a personal loan offer. By identifying potential customers more accurately, the bank can optimize its marketing resources and increase the success rate of loan conversions.

Key Challenge: The dataset is highly imbalanced, with only about 9.6% of customers accepting the loan offer.

🛠️ Technologies & Tools
Python (Core Language)

Scikit-learn (Pipeline, GridSearchCV, Modeling)

Pandas & NumPy (Data Manipulation)

Matplotlib & Seaborn (Data Visualization)

📊 Dataset Highlights
The dataset contains 5,000 customer records with 14 attributes, including:

Numerical: Age, Experience, Income, Family, CCAvg, Mortgage.

Categorical/Binary: Education, Securities Account, CD Account, Online, CreditCard.

Target: Personal Loan (1: Accepted, 0: Rejected).

🧪 Models & Methodology
I implemented a robust Machine Learning pipeline that includes:

Data Preprocessing: Handled missing values, corrected data types (e.g., CCAvg), and scaled features using MinMaxScaler.

Handling Imbalanced Data: Utilized class_weight='balanced' and focused on F1-Score as the primary evaluation metric.

Cross-Validation: Used GridSearchCV and K-Fold cross-validation to tune hyperparameters without data leakage.

Models Compared:
Logistic Regression: Baseline model with high interpretability.

K-Nearest Neighbors (KNN): Optimized k to balance the bias-variance tradeoff.

Complement Naive Bayes (CNB): Specifically tested for its performance on imbalanced datasets.

📈 Key Findings & Insights
Multicollinearity: Identified a near-perfect correlation (0.99) between Age and Experience.

Trade-offs: Learned the delicate balance between Precision and Recall when dealing with rare events (loan acceptance).

Optimization: Found that while some models excel in Accuracy, they may fail in F1-score if the minority class is not weighted correctly.

