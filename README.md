# Telco Customer Churn Analysis

**CadetX Data Analyst Internship — Sprint 1 Deliverable**

## Project Overview

Analysis of customer churn patterns for a telecommunications company using Python, Pandas, and Scikit-learn. This project identifies key churn drivers and builds predictive models to help reduce customer attrition.

## Dataset

- **Source:** Telco Customer Churn Dataset
- **Records:** 7,043 customers
- **Features:** 22 (demographics, services, account info, churn status)

## Tasks Completed

### Task 1: Data Loading & Inspection
- Loaded dataset with Pandas
- Explored data shape, types, and missing values
- Cleaned and preprocessed data for analysis

### Task 2: Exploratory Data Analysis (EDA)
- Analyzed churn rates by contract type, tenure, and internet service
- Created visualizations for churn distribution
- Identified key churn drivers:
  - Month-to-month contracts: 42.71% churn rate
  - 0-1 year tenure: 47.68% churn rate
  - Fiber optic users: highest churn segment

### Task 3: Data Preprocessing
- Handled missing values in TotalCharges
- Encoded categorical variables
- Feature engineering for model inputs

### Task 4: Machine Learning Models
- **Logistic Regression:** 77.3% accuracy
- **Random Forest:** 75.4% accuracy
- Generated confusion matrices and classification reports

### Task 5: Decision Thresholds & Cost-Benefit Analysis
- Optimal threshold: 0.180
- Precision-Recall tradeoff analysis
- Cost-benefit analysis at different threshold levels

## Technologies Used

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Key Findings

1. **Contract type** is the strongest predictor of churn
2. **New customers** (0-1 year) have highest churn risk
3. **Fiber optic service** correlates with higher churn
4. Optimal model threshold balances precision and recall

## Files

- `telco-churn-analysis.ipynb` — Main analysis notebook
- `telco_churn_task1.py` — Data loading script
- `telco_churn_task2.py` — EDA script
- `telco_churn_task3.py` — Preprocessing script
- `telco_churn_task4_fixed.py` — ML model training
- `telco_churn_task5_fixed.py` — Threshold analysis
- `Telco-Customer-Churn.csv` — Dataset

## Author

**Rehan Ali Haider**
- CadetX ID: CX-2026-89LY
- Track: Data Analyst
- Internship Period: Aug 2026 — Nov 2026

## License

This project is part of the CadetX Data Analyst Internship program.
