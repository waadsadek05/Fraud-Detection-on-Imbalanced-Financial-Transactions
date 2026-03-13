# Fraud-Detection-on-Imbalanced-Financial-Transactions
💳 Fraud Detection System using Machine Learning
📌 Project Overview

This project focuses on detecting fraudulent financial transactions using Machine Learning techniques. The goal is to build a reliable classification model capable of identifying suspicious transactions based on transaction attributes such as amount, account balances, and transaction type.

The dataset contains transaction records where each transaction is labeled as fraudulent or normal. Due to the severe class imbalance in the dataset, special techniques such as SMOTE were applied to improve the model's ability to detect fraud.

📂 Dataset Description

The dataset includes the following features:

Feature	Description
step	Time step of the transaction
type	Type of transaction (TRANSFER, PAYMENT, CASH_OUT, etc.)
amount	Transaction amount
nameOrig	Sender account ID
oldbalanceOrg	Sender balance before transaction
newbalanceOrig	Sender balance after transaction
nameDest	Receiver account ID
oldbalanceDest	Receiver balance before transaction
newbalanceDest	Receiver balance after transaction
isFraud	Target variable (1 = Fraud, 0 = Normal)
isFlaggedFraud	Flagged fraud indicator
⚙️ Data Preprocessing

The following preprocessing steps were performed:

Removed missing values

Removed duplicate records

Converted categorical variable transaction type using One-Hot Encoding

Removed non-informative features:

nameOrig

nameDest

step

isFlaggedFraud

Converted amount to numeric format

Split dataset into training and testing sets

Applied SMOTE (Synthetic Minority Oversampling Technique) to balance the dataset

Standardized features for SVM using StandardScaler

📊 Exploratory Data Analysis (EDA)

Several visualizations were created to better understand the dataset:

1️⃣ Correlation Matrix

Shows relationships between features and the fraud label.

2️⃣ Fraud vs Normal Distribution

Illustrates the severe class imbalance in the dataset.

3️⃣ Transaction Amount Boxplot

Compares transaction amounts between fraud and normal transactions using a log scale.

4️⃣ Transaction Amount Histogram

Displays distribution differences between fraudulent and normal transactions.

🤖 Machine Learning Models

Three models were trained and evaluated:

1️⃣ Logistic Regression

Handles linear relationships

Used class_weight="balanced"

2️⃣ Random Forest

Ensemble learning model

Handles nonlinear patterns well

Performed best among tested models

3️⃣ Support Vector Machine (SVM)

Used RBF Kernel

Applied feature scaling before training

🔍 Model Evaluation

Models were evaluated using:

Confusion Matrix

Precision

Recall

F1-score

Because fraud detection is a highly imbalanced classification problem, the focus was on Recall and F1-score for the fraud class.

Example Random Forest Results
Metric	Value
Precision (Fraud)	0.42
Recall (Fraud)	0.67
F1-score	0.51

Random Forest provided the best balance between detecting fraud and minimizing false positives.

🔧 Hyperparameter Tuning

GridSearchCV was used to optimize Random Forest parameters.

Best parameters:

n_estimators = 100
max_depth = None
min_samples_split = 2
💾 Model Saving

The trained model was saved using Joblib for later deployment.

joblib.dump(best_rf, 'fraud_rf_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
🔮 Example Prediction

Example transactions were tested with the saved model.

Transaction	Result
Transaction 1	Normal
Transaction 2	Fraud

Example Output:

Transaction 1 → Normal
Transaction 2 → Fraud
🛠 Technologies Used

Python

Pandas

NumPy

Scikit-learn

Imbalanced-learn (SMOTE)

Matplotlib

Seaborn

Joblib

📁 Project Structure
fraud-detection-project
│
├── samples.csv
├── fraud_detection.ipynb / fraud_detection.py
├── fraud_rf_model.pkl
├── scaler.pkl
│
├── plots
│   ├── correlation_matrix.png
│   ├── fraud_distribution.png
│   ├── amount_boxplot.png
│   └── confusion_matrix.png
│
└── README.md
🚀 Future Improvements

Use XGBoost or LightGBM

Train on a larger dataset

Deploy the model using Flask or FastAPI

Build a real-time fraud detection API

Implement deep learning models

👩‍💻 Author

Waad Sadek

Machine Learning enthusiast focused on building intelligent systems for real-world problems such as fraud detection and predictive analytics.
