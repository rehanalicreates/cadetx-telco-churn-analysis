# Task 1: Load & Inspect
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('Telco-Customer-Churn.csv')

# Inspect schema
print("Shape:", df.shape)
print("\nColumn Types:\n", df.dtypes)
print("\nFirst 5 rows:\n", df.head())
print("\nMissing values:\n", df.isnull().sum())

# Convert TotalCharges to numeric (strings with spaces become NaN)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Check missing after conversion
print("\nMissing TotalCharges:", df['TotalCharges'].isnull().sum())

# Impute missing TotalCharges with median
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

print("\nFinal missing values:\n", df.isnull().sum())
print("\nCleaned data shape:", df.shape)
