# Module 4: Rossmann Store Sales Time-Series Forecasting
# CadetX Data Analyst Internship — Rehan Ali Haider (CX-2026-89LY)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# TASK 1: Merge Store Metadata
# ============================================================
print("=" * 60)
print("TASK 1: Merge Store Metadata")
print("=" * 60)

# Load datasets
train_df = pd.read_csv('rossmann_train.csv')
store_df = pd.read_csv('rossmann_store.csv')
test_df = pd.read_csv('rossmann_test.csv')

print(f"Train Shape: {train_df.shape}")
print(f"Store Shape: {store_df.shape}")
print(f"Test Shape: {test_df.shape}")

print(f"\nTrain Columns: {train_df.columns.tolist()}")
print(f"\nStore Columns: {store_df.columns.tolist()}")

print(f"\nTrain First 5 rows:\n{train_df.head()}")
print(f"\nStore First 5 rows:\n{store_df.head()}")

# Merge train with store metadata
train_merged = pd.merge(train_df, store_df, on='Store', how='left')
print(f"\nMerged Train Shape: {train_merged.shape}")

# Merge test with store metadata
test_merged = pd.merge(test_df, store_df, on='Store', how='left')
print(f"Merged Test Shape: {test_merged.shape}")

print(f"\nMerged Columns: {train_merged.columns.tolist()}")

print("\nTask 1 Complete: Store metadata merged.\n")


# ============================================================
# TASK 2: Clean Closed Days
# ============================================================
print("=" * 60)
print("TASK 2: Clean Closed Days")
print("=" * 60)

# Check for closed stores
print(f"Closed stores (Open == 0): {(train_merged['Open'] == 0).sum()}")
print(f"Open stores (Open == 1): {(train_merged['Open'] == 1).sum()}")

# Filter closed stores
train_clean = train_merged[train_merged['Open'] == 1].copy()
print(f"\nAfter removing closed stores: {train_clean.shape}")

# Filter zero-sales days
print(f"Zero sales days: {(train_clean['Sales'] == 0).sum()}")
train_clean = train_clean[train_clean['Sales'] > 0]
print(f"After removing zero-sales days: {train_clean.shape}")

# Convert date to datetime
train_clean['Date'] = pd.to_datetime(train_clean['Date'])
test_merged['Date'] = pd.to_datetime(test_merged['Date'])

# Sort by date
train_clean = train_clean.sort_values(['Store', 'Date']).reset_index(drop=True)

print(f"\nDate Range: {train_clean['Date'].min()} to {train_clean['Date'].max()}")
print(f"Number of Stores: {train_clean['Store'].nunique()}")

print("\nTask 2 Complete: Closed stores and zero-sales days removed.\n")


# ============================================================
# TASK 3: Time-Based Split
# ============================================================
print("=" * 60)
print("TASK 3: Time-Based Split")
print("=" * 60)

# Sort by date for proper time-based split
train_clean = train_clean.sort_values('Date')

# Use last 6 weeks as validation set
split_date = train_clean['Date'].max() - pd.Timedelta(weeks=6)
print(f"Split Date: {split_date}")

train_split = train_clean[train_clean['Date'] < split_date].copy()
val_split = train_clean[train_clean['Date'] >= split_date].copy()

print(f"Training Set: {train_split.shape}")
print(f"Validation Set: {val_split.shape}")

print(f"\nTraining Date Range: {train_split['Date'].min()} to {train_split['Date'].max()}")
print(f"Validation Date Range: {val_split['Date'].min()} to {val_split['Date'].max()}")

print("\nTask 3 Complete: Time-based split created.\n")


# ============================================================
# TASK 4: Feature Engineering
# ============================================================
print("=" * 60)
print("TASK 4: Feature Engineering")
print("=" * 60)

def create_features(df):
    """Create features for sales forecasting"""
    df = df.copy()
    
    # Date features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
    
    # Lagged sales (for each store)
    df['Sales_Lag_7'] = df.groupby('Store')['Sales'].shift(7)
    df['Sales_Lag_14'] = df.groupby('Store')['Sales'].shift(14)
    df['Sales_Lag_28'] = df.groupby('Store')['Sales'].shift(28)
    
    # Rolling averages
    df['Sales_Rolling_7'] = df.groupby('Store')['Sales'].transform(lambda x: x.rolling(7).mean())
    df['Sales_Rolling_14'] = df.groupby('Store')['Sales'].transform(lambda x: x.rolling(14).mean())
    df['Sales_Rolling_28'] = df.groupby('Store')['Sales'].transform(lambda x: x.rolling(28).mean())
    
    # Promo duration
    df['Promo_Duration'] = df.groupby('Store')['Promo'].transform(lambda x: x.cumsum())
    
    # Competitor features
    df['Competition_Age'] = ((df['Date'] - pd.to_datetime(df['CompetitionOpenSinceMonth'].astype(str) + '/' + 
                                                           df['CompetitionOpenSinceYear'].astype(str), 
                                                           errors='coerce')).dt.days / 30).fillna(0)
    
    return df

# Apply feature engineering
train_features = create_features(train_split)
val_features = create_features(val_split)

print(f"Training features shape: {train_features.shape}")
print(f"Validation features shape: {val_features.shape}")

# Select features for modeling
feature_columns = ['Store', 'DayOfWeek', 'Promo', 'SchoolHoliday', 'Year', 'Month', 
                   'Day', 'WeekOfYear', 'IsWeekend', 'Sales_Lag_7', 'Sales_Lag_14', 
                   'Sales_Lag_28', 'Sales_Rolling_7', 'Sales_Rolling_14', 'Sales_Rolling_28',
                   'CompetitionDistance', 'Promo2']

# Filter to available columns
available_features = [col for col in feature_columns if col in train_features.columns]
print(f"\nFeatures used: {available_features}")

print("\nTask 4 Complete: Features engineered.\n")


# ============================================================
# TASK 5: 6-Week Sales Forecast
# ============================================================
print("=" * 60)
print("TASK 5: 6-Week Sales Forecast & Model Training")
print("=" * 60)

# Prepare training data
X_train = train_features[available_features].fillna(0)
y_train = train_features['Sales']

X_val = val_features[available_features].fillna(0)
y_val = val_features['Sales']

# Train Random Forest model
print("Training Random Forest model...")
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# Make predictions on validation set
val_predictions = rf_model.predict(X_val)

# Calculate metrics
rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
mae = mean_absolute_error(y_val, val_predictions)
mape = np.mean(np.abs((y_val - val_predictions) / y_val)) * 100

print(f"\nModel Performance:")
print(f"  RMSE: {rmse:.2f}")
print(f"  MAE: {mae:.2f}")
print(f"  MAPE: {mape:.2f}%")

# Baseline: moving average
baseline_predictions = train_features.groupby('Store')['Sales'].transform(lambda x: x.rolling(7).mean()).fillna(0)
baseline_rmse = np.sqrt(mean_squared_error(y_val, baseline_predictions[X_val.index]))

print(f"\nBaseline (7-day moving average) RMSE: {baseline_rmse:.2f}")
print(f"Improvement over baseline: {(baseline_rmse - rmse) / baseline_rmse * 100:.1f}%")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': available_features,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\nTop 10 Feature Importance:")
print(feature_importance.head(10).to_string(index=False))

# Forecast for next 6 weeks (using test set)
print(f"\nTest Set Shape: {test_merged.shape}")

# Create features for test set
test_features = create_features(test_merged)
X_test = test_features[available_features].fillna(0)

# Make predictions
test_predictions = rf_model.predict(X_test)
test_features['Predicted_Sales'] = test_predictions

print(f"\n6-Week Forecast Summary:")
print(f"  Mean Daily Sales per Store: ${test_predictions.mean():,.2f}")
print(f"  Total Predicted Sales (6 weeks): ${test_predictions.sum():,.2f}")

# Confidence interval
std_pred = test_predictions.std()
print(f"  Prediction Std Dev: ${std_pred:,.2f}")
print(f"  95% Confidence Range: ${test_predictions.mean() - 1.96*std_pred:,.2f} to ${test_predictions.mean() + 1.96*std_pred:,.2f}")

print("\nTask 5 Complete: 6-week sales forecast generated.\n")

# ============================================================
# SUMMARY
# ============================================================
print("=" * 60)
print("MODULE 4 COMPLETE")
print("=" * 60)
print(f"Total Stores: {train_clean['Store'].nunique()}")
print(f"Training Period: {train_clean['Date'].min()} to {train_clean['Date'].max()}")
print(f"Model RMSE: {rmse:.2f}")
print(f"Model MAPE: {mape:.2f}%")
print(f"Improvement over baseline: {(baseline_rmse - rmse) / baseline_rmse * 100:.1f}%")
print("=" * 60)
