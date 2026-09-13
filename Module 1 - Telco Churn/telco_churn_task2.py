# Task 2: Exploratory Data Analysis & Churn Drivers
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Telco-Customer-Churn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Overall churn rate
print("Overall Churn Rate:", df['Churn'].value_counts(normalize=True))

# Churn by Contract Type
print("\n--- Churn by Contract Type ---")
print(df.groupby('Contract')['Churn'].value_counts(normalize=True))

# Churn by Tenure (bucketed)
df['TenureBucket'] = pd.cut(df['tenure'], bins=[0, 12, 24, 36, 48, 60, 72],
                            labels=['0-12', '13-24', '25-36', '37-48', '49-60', '61-72'])
print("\n--- Churn by Tenure Bucket ---")
print(df.groupby('TenureBucket')['Churn'].value_counts(normalize=True))

# Churn by Internet Service
print("\n--- Churn by Internet Service ---")
print(df.groupby('InternetService')['Churn'].value_counts(normalize=True))

# Churn by Monthly Charges
print("\n--- Monthly Charges Summary by Churn ---")
print(df.groupby('Churn')['MonthlyCharges'].describe())

# Visualizations
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Contract vs Churn
df.groupby('Contract')['Churn'].value_counts(normalize=True).unstack().plot(kind='bar', ax=axes[0])
axes[0].set_title('Churn by Contract Type')
axes[0].set_ylabel('Proportion')

# Tenure vs Churn
df.groupby('TenureBucket')['Churn'].value_counts(normalize=True).unstack().plot(kind='bar', ax=axes[1])
axes[1].set_title('Churn by Tenure Bucket')
axes[1].set_ylabel('Proportion')

# Internet Service vs Churn
df.groupby('InternetService')['Churn'].value_counts(normalize=True).unstack().plot(kind='bar', ax=axes[2])
axes[2].set_title('Churn by Internet Service')
axes[2].set_ylabel('Proportion')

plt.tight_layout()
plt.savefig('churn_analysis.png', dpi=150)
plt.show()
