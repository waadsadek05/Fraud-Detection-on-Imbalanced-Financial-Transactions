# Fraud-Detection-on-Imbalanced-Financial-Transactions
# 💳 Fraud Detection System using Machine Learning

## 📌 Project Overview
This project detects **fraudulent financial transactions** using Machine Learning.  
The goal is to identify suspicious transactions based on features like amount, account balances, and transaction type.  

> ⚠️ Due to the severe class imbalance, **SMOTE** was applied to improve fraud detection performance.

---

## 📂 Dataset Description
The dataset contains **53,516 transactions** after cleaning.

| Feature | Description |
|---------|-------------|
| `step` | Time step of the transaction |
| `type` | Transaction type (`TRANSFER`, `PAYMENT`, `CASH_OUT`, etc.) |
| `amount` | Transaction amount |
| `nameOrig` | Sender account ID |
| `oldbalanceOrg` | Sender balance before transaction |
| `newbalanceOrig` | Sender balance after transaction |
| `nameDest` | Receiver account ID |
| `oldbalanceDest` | Receiver balance before transaction |
| `newbalanceDest` | Receiver balance after transaction |
| `isFraud` | Target variable (1 = Fraud, 0 = Normal) |
| `isFlaggedFraud` | Flagged fraud indicator |

**Fraud distribution**:

- Normal transactions: 53,416 (≈ 99.81%)  
- Fraud transactions: 100 (≈ 0.19%)

---

## ⚙️ Data Preprocessing

Steps applied:

1. Removed **missing values** and **duplicates**  
2. Converted categorical feature `type` using **One-Hot Encoding**  
3. Dropped non-informative features: `step`, `nameOrig`, `nameDest`, `isFlaggedFraud`  
4. Converted `amount` to numeric  
5. Split dataset into **training (70%)** and **testing (30%)**  
6. Balanced the training data using **SMOTE**  
7. Standardized features for **SVM** using **StandardScaler**

---

## 📊 Exploratory Data Analysis (EDA)

### 1️⃣ Correlation Matrix
![Correlation Matrix](plots/correlation_matrix.png)

### 2️⃣ Fraud vs Normal Distribution
![Fraud Distribution](plots/fraud_distribution.png)

### 3️⃣ Transaction Amount Boxplot
![Transaction Amount Boxplot](plots/amount_boxplot.png)

### 4️⃣ Confusion Matrices
![Confusion Matrices](plots/confusion_matrix.png)

---

## 🤖 Machine Learning Models

### 1️⃣ Logistic Regression
- Linear model, used `class_weight='balanced'`  
- High recall for fraud but low precision

### 2️⃣ Random Forest
- Ensemble model, handles nonlinear patterns  
- Best performance among tested models

| Metric | Fraud |
|--------|-------|
| Precision | 0.42 |
| Recall | 0.67 |
| F1-score | 0.51 |

### 3️⃣ SVM
- Used RBF kernel and scaled features  
- Similar recall to Logistic Regression, low precision

---

## 🔧 Hyperparameter Tuning (Random Forest)

GridSearchCV parameters:

```python
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

Train on larger datasets

Try XGBoost / LightGBM

Deploy as real-time API (Flask / FastAPI)

Experiment with Deep Learning for better detection

👩‍💻 Author

Waad Sadek
Machine Learning enthusiast building intelligent systems for fraud detection and predictive analytics.
