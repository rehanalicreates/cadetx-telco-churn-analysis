# Telco Customer Churn Analysis — Final Report

**Author:** Rehan Ali Haider (CX-2026-89LY)
**Date:** September 11, 2026
**Internship:** CadetX Data Analyst Program

---

## Executive Summary

This analysis examined customer churn patterns for a telecommunications company using a dataset of 7,043 customers. The goal was to identify key churn drivers and build predictive models to reduce customer attrition.

**Key Finding:** Month-to-month contracts, new customers (0-1 year tenure), and fiber optic service users are the highest churn risk segments.

---

## Business Problem

Customer churn costs telecom companies significantly more than retaining existing customers. This project identifies:
1. Which customers are most likely to churn
2. What factors drive churn decisions
3. How to optimize retention strategies

---

## Methodology

### Data Source
- **Dataset:** Telco Customer Churn (7,043 records, 22 features)
- **Tools:** Python, Pandas, Scikit-learn, Matplotlib

### Analysis Steps
1. Data cleaning and preprocessing
2. Exploratory Data Analysis (EDA)
3. Feature engineering
4. Model training (Logistic Regression & Random Forest)
5. Threshold optimization
6. Cost-benefit analysis

---

## Key Findings

### 1. Contract Type is the Strongest Predictor
| Contract | Churn Rate |
|----------|------------|
| Month-to-month | 42.71% |
| One year | 11.27% |
| Two year | 2.83% |

### 2. New Customers Leave Fast
| Tenure | Churn Rate |
|--------|------------|
| 0-1 year | 47.68% |
| 1-2 years | 28.71% |
| 2-4 years | 20.39% |
| 4+ years | 9.51% |

### 3. Fiber Optic Users Churn More
- Fiber optic customers have the highest churn rate
- DSL customers are more stable

---

## Model Performance

| Model | Accuracy | Precision (Churn) | Recall (Churn) |
|-------|----------|-------------------|----------------|
| Logistic Regression | 77.3% | 60% | 42% |
| Random Forest | 75.4% | 55% | 44% |

**Optimal Threshold:** 0.180 (balances precision and recall)

---

## Recommendations

1. **Target month-to-month customers** with loyalty discounts
2. **Improve onboarding** for first-year customers
3. **Review fiber optic pricing** — consider bundled offers
4. **Implement early warning system** using the predictive model

---

## Business Impact

At optimal threshold (0.180):
- **Precision:** 42.6% — identify churners accurately
- **Recall:** 77.0% — catch most at-risk customers
- **Net Benefit:** Rs. 93,750 per 1,000 customers contacted

---

## Conclusion

The analysis successfully identified key churn drivers and built predictive models with 77% accuracy. Implementing the recommended strategies could reduce churn by 15-20% in the first year.

---

## Files Delivered

| File | Description |
|------|-------------|
| telco-churn-analysis.ipynb | Complete analysis notebook |
| telco_churn_task1.py | Data loading script |
| telco_churn_task2.py | EDA script |
| telco_churn_task3.py | Preprocessing script |
| telco_churn_task4_fixed.py | Feature engineering |
| telco_churn_task5_fixed.py | Threshold analysis |
| Telco-Customer-Churn.csv | Raw dataset |
| README.md | Project documentation |

---

**CadetX ID:** CX-2026-89LY
**Track:** Data Analyst
**Status:** Sprint 10 Complete
