# Fraud-Detection-on-Imbalanced-Financial-Transactions
💳 Fraud Detection System using Machine Learning
📌 Project Overview

This project detects fraudulent financial transactions using Machine Learning. The goal is to identify suspicious transactions based on attributes like amount, account balances, and transaction type.
Due to a high class imbalance, SMOTE was used to improve fraud detection performance.

📂 Dataset Description

The dataset contains 53,516 transactions after cleaning. Features include:

Feature	Description
step	Time step of the transaction
type	Transaction type (TRANSFER, PAYMENT, CASH_OUT, etc.)
amount	Transaction amount
nameOrig	Sender account ID
oldbalanceOrg	Sender balance before transaction
newbalanceOrig	Sender balance after transaction
nameDest	Receiver account ID
oldbalanceDest	Receiver balance before transaction
newbalanceDest	Receiver balance after transaction
isFraud	Target variable (1 = Fraud, 0 = Normal)
isFlaggedFraud	Flagged fraud indicator

Fraud distribution:

Normal transactions: 53,416 (≈ 99.81%)

Fraud transactions: 100 (≈ 0.19%)

⚙️ Data Preprocessing

Steps applied:

Removed missing values and duplicates

Converted categorical feature type using One-Hot Encoding

Dropped non-informative features: step, nameOrig, nameDest, isFlaggedFraud

Converted amount to numeric

Split dataset into training (70%) and testing (30%)

Balanced the training data using SMOTE

Standardized features for SVM using StandardScaler

📊 Exploratory Data Analysis (EDA)
1️⃣ Correlation Matrix

Displays feature correlations with the target.

Helps identify important features for fraud detection.

2️⃣ Fraud vs Normal Distribution

Shows extreme class imbalance.

Only 0.19% of transactions are fraudulent.

3️⃣ Transaction Amount Boxplot

Compares amount for Normal vs Fraud transactions using log scale.

Fraud transactions usually have small or unusual amounts.

4️⃣ Transaction Amount Histogram

Log-scaled histogram of transaction amounts for both classes.

Highlights differences between fraud and normal patterns.

🤖 Machine Learning Models
1️⃣ Logistic Regression

Handles linear patterns.

Used class_weight='balanced' to handle class imbalance.

Results: High recall for fraud, low precision (many false positives).

2️⃣ Random Forest

Ensemble tree-based model, handles nonlinear patterns.

Best performance for fraud detection among tested models.

Results:

Metric	Fraud
Precision	0.42
Recall	0.67
F1-score	0.51
3️⃣ SVM

Used RBF kernel, scaled features.

High recall for fraud but low precision, similar to logistic regression.

🔧 Hyperparameter Tuning (Random Forest)

Used GridSearchCV:

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

Best parameters:

{'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2}
💾 Model Saving
joblib.dump(best_rf, 'fraud_rf_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

The saved model can predict new transactions without retraining.

🔮 Example Predictions
Transaction	Amount	Sender Balance	Receiver Balance	Result
Transaction 1	5,000	20,000 → 15,000	1,000 → 6,000	Normal ✅
Transaction 2	181	181 → 0	0 → 0	Fraud ⚠️
🛠 Technologies Used

Python

Pandas, NumPy

Scikit-learn

Imbalanced-learn (SMOTE)

Matplotlib, Seaborn

Joblib

📁 Project Structure
fraud-detection-project/
│
├── samples.csv
├── fraud_detection.ipynb
├── fraud_rf_model.pkl
├── scaler.pkl
│
├── plots/
│   ├── correlation_matrix.png
│   ├── fraud_distribution.png
│   ├── amount_boxplot.png
│   └── confusion_matrix.png
│
└── README.md
🚀 Future Improvements

Train with larger datasets

Test XGBoost / LightGBM

Deploy as real-time API (Flask / FastAPI)

Experiment with Deep Learning for better detection

👩‍💻 Author

Waad Sadek
Machine Learning enthusiast, building intelligent systems for fraud detection and predictive analytics.
