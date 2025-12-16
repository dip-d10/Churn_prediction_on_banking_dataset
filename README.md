# Bank Customer Churn Prediction using Machine Learning

##  Project Overview

Customer churn is a major challenge for banking institutions, as retaining existing customers is significantly more cost-effective than acquiring new ones. This project develops a **machine learning–based customer churn prediction system** that identifies customers who are likely to leave the bank, enabling **proactive, data-driven retention strategies**.

The project follows a **complete end-to-end data science lifecycle**, covering data exploration, preprocessing, model development, evaluation, comparison, interpretation, and business recommendation formulation.

---

##  Objectives

The primary objectives of this project are:

- Analyze customer behavior and financial patterns associated with churn  
- Build and compare multiple machine learning classification models  
- Address class imbalance by prioritizing **Recall and F1-score**  
- Select a **production-ready model** suitable for real-world deployment  
- Extract **actionable business insights** from model interpretation  

---

##  Problem Statement

Given structured banking customer data, the goal is to:

> **Predict whether a customer will churn (Exited = 1) or remain with the bank (Exited = 0)**

This is a **binary classification problem** with an **imbalanced target distribution**, where correctly identifying churned customers is more important than maximizing overall accuracy.

---

##  Dataset Description

- **Total Records:** 10,000  
- **Total Features:** 14 (before preprocessing)  
- **Target Variable:** `Exited`  

### Feature Categories
- **Demographic:** Age, Gender, Geography  
- **Financial:** CreditScore, Balance, EstimatedSalary  
- **Engagement:** Tenure, NumOfProducts, IsActiveMember, HasCrCard  

### Data Quality
- No missing values  
- No duplicate records  
- Clean and consistent feature types  

---

##  Technologies & Libraries Used

### Programming Language
- Python

### Libraries
- **Pandas, NumPy** – Data manipulation and numerical computation  
- **Matplotlib, Seaborn** – Data visualization  
- **Scikit-learn** – Preprocessing, model training, and evaluation  
- **Joblib** – Model persistence and deployment readiness  

### Machine Learning Models
- Logistic Regression  
- Support Vector Classifier (SVC)  
- K-Nearest Neighbors (KNN)  
- Decision Tree  
- Random Forest  
- Gradient Boosting  

---

##  Project Workflow

1. Data Loading and Exploration  
2. Data Cleaning and Feature Selection  
3. Encoding of Categorical Variables  
4. Feature Scaling  
5. Train–Test Split (80/20)  
6. Model Training  
7. Model Evaluation using multiple metrics  
8. Model Comparison and Selection  
9. Feature Importance Analysis  
10. Business Interpretation and Recommendations  

This structured workflow ensures **reproducibility**, **interpretability**, and **alignment with business objectives**.

---

##  Data Preprocessing Summary

- Removed non-informative identifier columns:
  - `RowNumber`
  - `CustomerId`
  - `Surname`
- Encoding:
  - **Gender:** Label Encoding  
  - **Geography:** One-Hot Encoding  
- Feature Scaling:
  - Standardization using `StandardScaler`
- Feature–Target Split:
  - **X:** 11 input features  
  - **y:** Churn label (`Exited`)  

---

##  Model Evaluation Metrics

Due to the imbalanced nature of the dataset, the following metrics were used:

- **Accuracy** – Overall correctness  
- **Precision** – Correct churn predictions among predicted churn  
- **Recall** – Ability to identify actual churners (**critical metric**)  
- **F1-score** – Balance between Precision and Recall  

---

##  Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-score |
|------|--------|----------|--------|----------|
| Logistic Regression | 0.811 | 0.55 | 0.20 | 0.29 |
| Support Vector Classifier | 0.786 | 0.47 | 0.75 | 0.58 |
| K-Nearest Neighbors | 0.830 | 0.61 | 0.37 | 0.46 |
| Decision Tree | 0.746 | 0.42 | 0.79 | 0.55 |
| **Random Forest** | **0.833** | **0.56** | **0.73** | **0.63** |
| Gradient Boosting | 0.867 | 0.75 | 0.48 | 0.58 |

---

##  Final Model Selection: Random Forest

### Why Random Forest?

- Balanced performance across Precision, Recall, and F1-score  
- Strong recall for churned customers  
- Ensemble approach reduces overfitting  
- Provides feature importance for interpretability  
- Stable and reliable for real-world deployment  

---

##  Feature Importance Insights

Top churn-driving features identified by the Random Forest model:

1. Age  
2. Number of Products  
3. Account Balance  
4. IsActiveMember  
5. Geography (Germany)  
6. CreditScore  
7. EstimatedSalary  

> **Key Insight:** Customer engagement and product usage dominate churn behavior more than purely demographic features.

---

##  Business Recommendations

- **Increase Product Engagement:** Cross-sell and bundle banking products  
- **Target Inactive Customers:** Early intervention through personalized campaigns  
- **Age-Specific Retention Strategies:** Tailored offers based on customer age groups  
- **Balance-Based Risk Segmentation:** Customized financial services  
- **Region-Specific Retention Strategies:** Localized engagement initiatives  

These strategies enable a shift from **reactive churn handling to proactive customer retention**.

---

##  Deployment Readiness

- Model serialized using `joblib`  
- Compatible with REST API deployment  
- Scalable for integration with CRM systems  
- Cloud-ready (AWS / Azure / GCP)  

---

##  Limitations

- Class imbalance impacts churn recall stability  
- Dataset represents a static snapshot in time  
- Lack of advanced behavioral data (transactions, app usage)  
- Tree-based models provide limited instance-level interpretability  

---

##  Future Enhancements

- Advanced imbalance handling techniques (SMOTE, ADASYN)  
- Hyperparameter optimization (GridSearchCV, Bayesian Optimization)  
- Temporal modeling using time-series data  
- Explainable AI (SHAP, LIME)  
- Real-time churn scoring pipelines  

---

##  Executive Summary

This project demonstrates a **production-oriented machine learning solution** for predicting customer churn in the banking sector. By combining rigorous preprocessing, multi-model evaluation, and business-focused interpretation, the system enables **early identification of high-risk customers** and supports **data-driven retention strategies**.

---


