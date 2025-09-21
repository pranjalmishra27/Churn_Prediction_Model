# Telco Customer Churn Prediction

This repository contains a **Machine Learning project** for predicting customer churn in the telecommunications industry using the **Telco Customer Churn dataset** from Kaggle. The project leverages **XGBoost**, **feature engineering**, and **hyperparameter tuning** to build a robust churn prediction model.

---

## **Dataset**

- Source: [Telco Customer Churn Dataset - Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn)  
- Description: This dataset contains **7043 customer records** with **21 features** including demographic info, account info, and services subscribed.  
- Target Variable: `Churn` (1 = customer churned, 0 = customer retained)  

**Key Features Include:**  
- `gender`, `SeniorCitizen`, `Partner`, `Dependents`  
- `tenure`, `MonthlyCharges`, `TotalCharges`  
- `Contract`, `PaymentMethod`, `InternetService`, `TechSupport`, etc.

---

## **Project Steps**

1. **Data Cleaning**  
   - Converted `TotalCharges` to numeric and filled missing values.  
   - Dropped `customerID` column.  

2. **Encoding Categorical Features**  
   - Binary columns encoded with `LabelEncoder`.  
   - Multi-category columns encoded with `One-Hot Encoding`.  

3. **Handling Imbalanced Classes**  
   - Used **SMOTE** to oversample minority class (churned customers).  

4. **Train-Test Split & Scaling**  
   - Split data 80:20 for training and testing.  
   - Scaled features using `StandardScaler` (optional for XGBoost).  

5. **Model Training**  
   - Used **XGBoost Classifier** with hyperparameter tuning using **GridSearchCV**.  
   - Tuned parameters include:
     - `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`  

6. **Evaluation Metrics**  
   - Accuracy, F1-Score, ROC-AUC, Confusion Matrix, Classification Report.  

7. **Feature Importance & Explainability**  
   - Visualized top features using `xgboost.plot_importance`.  
   - Optional: SHAP explainability for individual predictions.  

---

## **Results**

- Tuned model achieved improved **accuracy and ROC-AUC** compared to default parameters.  
- Top contributing features include:  
  - `Contract`, `tenure`, `MonthlyCharges`, `TotalCharges`, `TechSupport`  

---

## **Usage**

1. Clone the repository:

```bash
git clone <your-repo-url>
cd telco-churn-prediction
