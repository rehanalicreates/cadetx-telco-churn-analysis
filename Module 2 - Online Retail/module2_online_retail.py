# Module 2: Online Retail Analytics — RFM Segmentation & Market Basket
# CadetX Data Analyst Internship — Rehan Ali Haider (CX-2026-89LY)

import pandas as pd
import numpy as np
from datetime import datetime

# ============================================================
# TASK 1: Order Sanitization
# ============================================================
print("=" * 60)
print("TASK 1: Order Sanitization")
print("=" * 60)

# Load dataset
df = pd.read_excel('online_retail.xlsx')

print(f"Dataset Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nMissing Values:\n{df.isnull().sum()}")

# Clean missing CustomerIDs
df = df.dropna(subset=['CustomerID'])
print(f"\nAfter dropping missing CustomerID: {df.shape}")

# Convert CustomerID to integer
df['CustomerID'] = df['CustomerID'].astype(int)

# Filter cancelled orders (Invoice starting with 'C')
cancelled_mask = df['InvoiceNo'].astype(str).str.startswith('C')
cancelled_orders = df[cancelled_mask]
print(f"Cancelled orders found: {len(cancelled_orders)}")

# Keep only non-cancelled orders
df = df[~cancelled_mask]
print(f"After removing cancelled orders: {df.shape}")

print("\nTask 1 Complete: Data loaded, missing CustomerIDs removed, cancelled orders filtered.\n")


# ============================================================
# TASK 2: Calculate Spend
# ============================================================
print("=" * 60)
print("TASK 2: Calculate Spend")
print("=" * 60)

# Filter invalid non-positive quantities
print(f"Negative quantities found: {(df['Quantity'] <= 0).sum()}")
df = df[df['Quantity'] > 0]
print(f"After filtering non-positive quantities: {df.shape}")

# Calculate Total Sales
df['TotalSales'] = df['Quantity'] * df['UnitPrice']
print(f"\nTotal Sales Statistics:")
print(f"  Mean: ${df['TotalSales'].mean():.2f}")
print(f"  Median: ${df['TotalSales'].median():.2f}")
print(f"  Max: ${df['TotalSales'].max():.2f}")
print(f"  Total Revenue: ${df['TotalSales'].sum():.2f}")

print("\nTask 2 Complete: Total Sales calculated, invalid quantities filtered.\n")


# ============================================================
# TASK 3: Monthly Analytics
# ============================================================
print("=" * 60)
print("TASK 3: Monthly Analytics")
print("=" * 60)

# Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Extract month and year
df['YearMonth'] = df['InvoiceDate'].dt.to_period('M')

# Monthly Revenue Analysis
monthly_revenue = df.groupby('YearMonth')['TotalSales'].sum().reset_index()
monthly_revenue.columns = ['Month', 'Revenue']
print("Monthly Revenue:")
print(monthly_revenue.to_string(index=False))

# Top Performing Countries
country_revenue = df.groupby('Country')['TotalSales'].sum().sort_values(ascending=False).head(10)
print(f"\nTop 10 Countries by Revenue:")
print(country_revenue.to_string())

# Revenue by Country
country_stats = df.groupby('Country').agg({
    'TotalSales': ['sum', 'count'],
    'CustomerID': 'nunique'
}).round(2)
country_stats.columns = ['TotalRevenue', 'TransactionCount', 'UniqueCustomers']
country_stats = country_stats.sort_values('TotalRevenue', ascending=False).head(10)
print(f"\nCountry Statistics:")
print(country_stats.to_string())

print("\nTask 3 Complete: Monthly revenue and top countries analyzed.\n")


# ============================================================
# TASK 4: RFM Scoring
# ============================================================
print("=" * 60)
print("TASK 4: RFM Customer Segmentation")
print("=" * 60)

# Set analysis date (one day after last transaction)
analysis_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
print(f"Analysis Date: {analysis_date}")

# Calculate RFM metrics
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (analysis_date - x.max()).days,  # Recency
    'InvoiceNo': 'count',  # Frequency
    'TotalSales': 'sum'  # Monetary
}).reset_index()

rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

# Create RFM scores (1-5 scale)
rfm['R_Score'] = pd.qcut(rfm['Recency'], q=5, labels=[5, 4, 3, 2, 1])
rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5])
rfm['M_Score'] = pd.qcut(rfm['Monetary'], q=5, labels=[1, 2, 3, 4, 5])

# Combine scores
rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

# Customer Segments
def segment_customer(row):
    if row['R_Score'] == '5' and row['F_Score'] == '5' and row['M_Score'] == '5':
        return 'Champions'
    elif row['R_Score'] == '5' and row['F_Score'] == '4':
        return 'Loyal Customers'
    elif row['R_Score'] == '4' and row['F_Score'] == '4':
        return 'Potential Loyalists'
    elif row['R_Score'] == '5' and row['F_Score'] == '1':
        return 'New Customers'
    elif row['R_Score'] == '1' and row['F_Score'] == '5':
        return 'At Risk'
    elif row['R_Score'] == '1' and row['F_Score'] == '1':
        return 'Lost'
    else:
        return 'Others'

rfm['Segment'] = rfm.apply(segment_customer, axis=1)

print(f"\nRFM Summary Statistics:")
print(rfm.describe().round(2))
print(f"\nCustomer Segments:")
print(rfm['Segment'].value_counts())
print(f"\nTop 10 Customers by RFM Score:")
print(rfm.nlargest(10, 'Monetary')[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'Segment']].to_string(index=False))

print("\nTask 4 Complete: RFM segmentation calculated.\n")


# ============================================================
# TASK 5: Market Basket Analysis
# ============================================================
print("=" * 60)
print("TASK 5: Market Basket Analysis")
print("=" * 60)

from itertools import combinations
from collections import Counter

# Create basket data (products per transaction)
basket = df.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().fillna(0)

# Convert to binary (1 if purchased, 0 if not)
basket_binary = basket.applymap(lambda x: 1 if x > 0 else 0)

print(f"Basket Shape: {basket_binary.shape}")
print(f"Number of transactions: {basket_binary.shape[0]}")
print(f"Number of products: {basket_binary.shape[1]}")

# Find frequent itemsets (products bought together)
def get_frequent_pairs(basket_df, min_support=0.01):
    n_transactions = len(basket_df)
    product_counts = basket_df.sum().sort_values(ascending=False)
    
    # Get frequent individual items
    frequent_items = product_counts[product_counts / n_transactions >= min_support]
    
    # Get frequent pairs
    pair_counts = Counter()
    for idx in basket_df.index:
        purchased = basket_df.columns[basket_df.loc[idx] == 1].tolist()
        for pair in combinations(sorted(purchased), 2):
            pair_counts[pair] += 1
    
    # Filter by minimum support
    frequent_pairs = {pair: count for pair, count in pair_counts.items() 
                      if count / n_transactions >= min_support}
    
    return frequent_items, frequent_pairs

frequent_items, frequent_pairs = get_frequent_pairs(basket_binary, min_support=0.01)

print(f"\nTop 10 Frequent Individual Products:")
for product, count in list(frequent_items.items())[:10]:
    print(f"  {product[:50]}: {count} transactions ({count/len(basket_binary)*100:.1f}%)")

print(f"\nTop 10 Frequent Product Pairs:")
sorted_pairs = sorted(frequent_pairs.items(), key=lambda x: x[1], reverse=True)
for pair, count in sorted_pairs[:10]:
    print(f"  {pair[0][:30]} + {pair[1][:30]}: {count} transactions ({count/len(basket_binary)*100:.1f}%)")

# Calculate lift for top pairs
def calculate_lift(basket_df, product_a, product_b):
    n = len(basket_df)
    a_count = basket_df[product_a].sum()
    b_count = basket_df[product_b].sum()
    ab_count = ((basket_df[product_a] == 1) & (basket_df[product_b] == 1)).sum()
    
    if a_count == 0 or b_count == 0:
        return 0
    
    lift = (ab_count / n) / ((a_count / n) * (b_count / n))
    return lift

print(f"\nLift Analysis for Top Pairs:")
for pair, count in sorted_pairs[:5]:
    lift = calculate_lift(basket_binary, pair[0], pair[1])
    print(f"  {pair[0][:30]} + {pair[1][:30]}: Lift = {lift:.2f}")

print("\nTask 5 Complete: Market basket analysis finished.\n")

# ============================================================
# SUMMARY
# ============================================================
print("=" * 60)
print("MODULE 2 COMPLETE")
print("=" * 60)
print(f"Total Customers Analyzed: {rfm['CustomerID'].nunique()}")
print(f"Total Transactions: {df['InvoiceNo'].nunique()}")
print(f"Total Revenue: ${df['TotalSales'].sum():,.2f}")
print(f"Date Range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")
print("=" * 60)
