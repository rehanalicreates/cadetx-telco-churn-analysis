# Module 3: NHS Hospital Waiting Times Performance Analysis
# CadetX Data Analyst Internship — Rehan Ali Haider (CX-2026-89LY)

import pandas as pd
import numpy as np

# ============================================================
# TASK 1: Clean NHS Schema
# ============================================================
print("=" * 60)
print("TASK 1: Clean NHS Schema")
print("=" * 60)

# Load dataset
df = pd.read_csv('NHS_RTT_MAY2026.csv')

print(f"Dataset Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nMissing Values:\n{df.isnull().sum()}")

# Filter out summary rows (check for aggregate/summary indicators)
summary_indicators = ['Total', 'All', 'Summary', 'Overall', 'NHS England']
for col in df.columns:
    if df[col].dtype == 'object':
        mask = df[col].str.contains('|'.join(summary_indicators), case=False, na=False)
        if mask.any():
            print(f"\nSummary rows found in '{col}': {mask.sum()}")
            df = df[~mask]

print(f"\nAfter filtering summary rows: {df.shape}")

# Identify key columns
print(f"\nColumn Names:")
for i, col in enumerate(df.columns):
    print(f"  {i}: {col}")

print("\nTask 1 Complete: NHS data loaded and cleaned.\n")


# ============================================================
# TASK 2: 18-Week Targets
# ============================================================
print("=" * 60)
print("TASK 2: 18-Week Targets")
print("=" * 60)

# Identify relevant columns for 18-week wait analysis
# Common NHS RTT columns: Total patients, Patients within 18 weeks, etc.
# Adjust column names based on actual dataset structure

# Find columns containing '18' or 'week'
week_cols = [col for col in df.columns if '18' in str(col) or 'week' in str(col).lower()]
print(f"Columns related to 18-week waits: {week_cols}")

# Find total patients column
total_cols = [col for col in df.columns if 'total' in str(col).lower() or 'all' in str(col).lower()]
print(f"Total patients columns: {total_cols}")

# Calculate 18-week compliance percentage
# This will vary based on actual column names
try:
    # Try common column name patterns
    total_col = [col for col in df.columns if 'total' in str(col).lower() and 'patient' in str(col).lower()]
    within_18_col = [col for col in df.columns if '18' in str(col) and 'week' in str(col).lower()]
    
    if total_col and within_18_col:
        total_patients = df[total_col[0]]
        within_18_weeks = df[within_18_col[0]]
        
        df['Compliance_Rate'] = (within_18_weeks / total_patients * 100).round(2)
        print(f"\n18-Week Compliance Rate Statistics:")
        print(df['Compliance_Rate'].describe())
        
        # Identify trusts exceeding target (typically 92%)
        target = 92
        compliant_trusts = df[df['Compliance_Rate'] >= target]
        non_compliant_trusts = df[df['Compliance_Rate'] < target]
        
        print(f"\nTrusts meeting 92% target: {len(compliant_trusts)}")
        print(f"Trusts below 92% target: {len(non_compliant_trusts)}")
    else:
        print("Could not identify total patients and 18-week columns")
        print("Available columns:", df.columns.tolist())
except Exception as e:
    print(f"Error calculating compliance: {e}")
    print("Available columns:", df.columns.tolist())

print("\nTask 2 Complete: 18-week compliance calculated.\n")


# ============================================================
# TASK 3: Specialty Bottlenecks
# ============================================================
print("=" * 60)
print("TASK 3: Specialty Bottlenecks")
print("=" * 60)

# Find specialty column
specialty_cols = [col for col in df.columns if 'special' in str(col).lower() or 'type' in str(col).lower()]
print(f"Specialty columns: {specialty_cols}")

if specialty_cols:
    specialty_col = specialty_cols[0]
    
    # Calculate average wait by specialty
    wait_cols = [col for col in df.columns if 'wait' in str(col).lower() or 'mean' in str(col).lower() or 'median' in str(col).lower()]
    
    if wait_cols:
        wait_col = wait_cols[0]
        specialty_stats = df.groupby(specialty_col).agg({
            wait_col: ['mean', 'max', 'count']
        }).round(2)
        specialty_stats.columns = ['Average_Wait', 'Max_Wait', 'Trust_Count']
        specialty_stats = specialty_stats.sort_values('Average_Wait', ascending=False)
        
        print(f"\nTop 10 Specialties by Average Wait Time:")
        print(specialty_stats.head(10).to_string())
        
        # Identify bottleneck specialties
        bottleneck_threshold = specialty_stats['Average_Wait'].quantile(0.9)
        bottlenecks = specialty_stats[specialty_stats['Average_Wait'] >= bottleneck_threshold]
        print(f"\nBottleneck Specialties (90th percentile wait):")
        print(bottlenecks.to_string())
    else:
        print("No wait time columns found")
else:
    print("No specialty column found")
    print("Available columns:", df.columns.tolist())

print("\nTask 3 Complete: Specialty bottlenecks identified.\n")


# ============================================================
# TASK 4: Regional Aggregations
# ============================================================
print("=" * 60)
print("TASK 4: Regional Aggregations")
print("=" * 60)

# Find region column
region_cols = [col for col in df.columns if 'region' in str(col).lower() or 'area' in str(col).lower() or 'trust' in str(col).lower()]
print(f"Region columns: {region_cols}")

if region_cols:
    region_col = region_cols[0]
    
    # Regional statistics
    if 'Compliance_Rate' in df.columns:
        regional_stats = df.groupby(region_col).agg({
            'Compliance_Rate': ['mean', 'min', 'max', 'count']
        }).round(2)
        regional_stats.columns = ['Avg_Compliance', 'Min_Compliance', 'Max_Compliance', 'Trust_Count']
        regional_stats = regional_stats.sort_values('Avg_Compliance', ascending=False)
        
        print(f"\nRegional Compliance Statistics:")
        print(regional_stats.to_string())
        
        # Regional distribution
        print(f"\nRegional Distribution:")
        print(df[region_col].value_counts().head(10).to_string())
else:
    print("No region column found")

# Waiting times distribution
if 'Compliance_Rate' in df.columns:
    print(f"\nWaiting Times Distribution:")
    print(f"  Mean: {df['Compliance_Rate'].mean():.2f}%")
    print(f"  Median: {df['Compliance_Rate'].median():.2f}%")
    print(f"  Std Dev: {df['Compliance_Rate'].std():.2f}%")
    print(f"  Min: {df['Compliance_Rate'].min():.2f}%")
    print(f"  Max: {df['Compliance_Rate'].max():.2f}%")

print("\nTask 4 Complete: Regional aggregations analyzed.\n")


# ============================================================
# TASK 5: Executive Report
# ============================================================
print("=" * 60)
print("TASK 5: Executive Summary & Recommendations")
print("=" * 60)

print("\n" + "=" * 60)
print("NHS HOSPITAL WAITING TIMES - EXECUTIVE SUMMARY")
print("=" * 60)

print(f"\nDataset Overview:")
print(f"  Total Records: {len(df)}")
print(f"  Date Range: May 2026")

if 'Compliance_Rate' in df.columns:
    print(f"\nKey Performance Metrics:")
    print(f"  Average 18-Week Compliance: {df['Compliance_Rate'].mean():.2f}%")
    print(f"  Trusts Meeting Target (92%): {(df['Compliance_Rate'] >= 92).sum()}")
    print(f"  Trusts Below Target: {(df['Compliance_Rate'] < 92).sum()}")
    
    print(f"\nCritical Findings:")
    worst_trusts = df.nsmallest(5, 'Compliance_Rate')
    print(f"  Worst Performing Trusts:")
    for idx, row in worst_trusts.iterrows():
        print(f"    - {row.get('Trust Name', row.get('Organisation', 'Unknown'))}: {row['Compliance_Rate']:.1f}%")
    
    print(f"\nRecommendations:")
    print(f"  1. Immediate review of trusts below 92% compliance target")
    print(f"  2. Resource reallocation to bottleneck specialties")
    print(f"  3. Cross-regional support for worst-performing areas")
    print(f"  4. Monthly monitoring of improvement targets")
    print(f"  5. Patient pathway optimization for long-wait patients")

print("\n" + "=" * 60)
print("MODULE 3 COMPLETE")
print("=" * 60)
