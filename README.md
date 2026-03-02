# Proactive Fraud Detection for Large-Scale Financial Transactions

This project implements an end-to-end machine learning pipeline to detect fraudulent activity within a massive financial dataset (6.4 million rows). By focusing on "account wipeout" patterns and utilizing high-performance gradient boosting, the system achieves a 92% F1-score.

## 🛠️ Technology Stack
* **Data Processing:** `Pandas`, `NumPy`
* **Feature Engineering:** `Scikit-learn` (RobustScaler, LabelEncoder)
* **Imbalance Handling:** `Imbalanced-learn` (SMOTE)
* **Machine Learning:** `LightGBM` (Microsoft ML Library)
* **Evaluation:** `Scikit-learn` (Classification Report, Confusion Matrix)

---

# The business questions.

### 1. Data cleaning including missing values, outliers and multi-collinearity.
The dataset used is a <mark> **500MB CSV** </mark> containing approximately <mark> **64 lakh rows** and **5 primary columns**. </mark>
* **Missing Values:** There were **no missing values** identified during the data inspection phase.
* **Outliers:** High outliers were present in the numerical data. I first applied **Robust Scaling**, which is critical for financial data as it uses the Interquartile Range (IQR) to scale features, making the model resilient to extreme transaction amounts.
* **Log Transformation:** I then applied a **log1p transformation** across numerical columns. This desensitizes the model to outlier data without removing it, preserving the signal of high-value transactions.
* **Encoding:** **Label Encoding** was applied to handle categorical variables.
* **Class Imbalance:** To handle the significant imbalance between fraudulent and legitimate transactions, I used the **SMOTE (Synthetic Minority Over-sampling Technique)** to generate synthetic data for the minority class during the preprocessing stage.



### 2. Describe your fraud detection model in elaboration.
The model is built using <mark **LightGBM**,</mark>a performance-efficient gradient boosting framework from Microsoft. It was chosen for its high accuracy and ability to process millions of rows with minimal memory overhead. The model achieved a **92% harmonic F1 score**, indicating it is highly reliable for production deployment.



### 3. How did you select variables to be included in the model?
Variable selection was driven by **data correlation** analysis using `pandas`. I prioritized **transformed features** and variables that showed a strong statistical relationship with the target, specifically focusing on the flow of funds between accounts.

### 4. Demonstrate the performance of the model by using best set of tools.
The performance is demonstrated using a **Classification Report**, which provides the best overview of precision, recall, and the F1-score. These tools are essential for evaluating unbalanced datasets. Additionally, a **Confusion Matrix** was utilized to visualize the model's ability to distinguish between actual fraud and legitimate transactions.



### 5. What are the key factors that predict fraudulent customer?
The key predictive factors identified by the model are:
* **Sender balance before** the transaction.
* **Sender balance after** the transaction.
* **Receiver balance before** the transaction.
* **Receiver balance after** the transaction.
* **The volume of the amount**
* **The delta of sent and received**



### 6. Do these factors make sense? If yes, How? If not, How not?
**Yes, these factors make sense.** Most fraudulent transactions aim for a complete **wipeout of the sender's account**. Furthermore, improper transaction results often occur due to **SQL injection** or unauthorized access. Monitoring the delta between the "before" and "after" balances of both parties provides a clear digital signature of such malicious activity.

### 7. What kind of prevention should be adopted while company update its infrastructure?
The company should adopt a **consent-based verification protocol**. Specifically, the system should request explicit consent and secondary verification from the sender whenever a transaction requires more than **50% of the account balance** to be debited.

### 8. Assuming these actions have been implemented, how would you determine if they work?
To determine effectiveness, the model should monitor **demand requests** in real-time. It will check the **before and after balance** of every transaction; if a fraud pattern is detected, the system should automatically **revert the transaction** and return the money to the original account immediately. Success is measured by the reduction in successful "account wipeouts" and the volume of funds successfully reverted.
