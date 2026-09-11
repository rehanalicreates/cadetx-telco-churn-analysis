# Task 3: Preprocessing & Feature Encoding
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

df = pd.read_csv('Telco-Customer-Churn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Drop unnecessary columns
df.drop(['customerID', 'TenureBucket'], axis=1, inplace=True, errors='ignore')

# Binary encode target
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Identify columns
binary_cols = [col for col in df.columns if df[col].nunique() == 2 and col != 'Churn']
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_cols = [col for col in df.columns if df[col].nunique() > 2 and col not in numeric_cols and col != 'Churn']

# Binary encode
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})

print("Binary columns:", binary_cols)
print("Numeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)

# One-hot encode categorical, scale numeric
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols),
        ('pass', 'passthrough', binary_cols)
    ]
)

X = df.drop('Churn', axis=1)
y = df['Churn']

X_processed = preprocessor.fit_transform(X)

# Get feature names
cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
all_feature_names = numeric_cols + list(cat_feature_names) + binary_cols

print("\nProcessed shape:", X_processed.shape)
print("Features:", all_feature_names)
print("\nSample:\n", X_processed[:2])
