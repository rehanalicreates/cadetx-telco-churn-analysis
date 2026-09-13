# Module 5: Spotify Artist Data Cleaning & Analytics
# CadetX Data Analyst Internship — Rehan Ali Haider (CX-2026-89LY)

import pandas as pd
import numpy as np
import re

# ============================================================
# TASK 1: Load Data
# ============================================================
print("=" * 60)
print("TASK 1: Load Data")
print("=" * 60)

# Load dataset
df = pd.read_csv('spotify_artists.csv')

print(f"Dataset Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nColumn Names (cleaned):")
for i, col in enumerate(df.columns):
    print(f"  {i}: '{col}'")

print("\nTask 1 Complete: Data loaded and inspected.\n")


# ============================================================
# TASK 2: Clean Numbers
# ============================================================
print("=" * 60)
print("TASK 2: Clean Numbers")
print("=" * 60)

# Identify numeric columns that need cleaning
numeric_columns = df.select_dtypes(include=['object']).columns.tolist()
print(f"Object columns to clean: {numeric_columns}")

# Function to clean numeric strings
def clean_numeric(value):
    """Convert messy numeric strings to valid floats"""
    if pd.isna(value) or value == '' or value is None:
        return np.nan
    
    # Convert to string
    value = str(value)
    
    # Remove common non-numeric characters
    value = re.sub(r'[?!]', '', value)
    
    # Handle European number format (comma as decimal)
    if ',' in value and '.' in value:
        # If both comma and dot, comma is thousands separator
        value = value.replace(',', '')
    elif ',' in value:
        # If only comma, it's decimal separator
        value = value.replace(',', '.')
    
    # Remove any remaining non-numeric characters except dot and minus
    value = re.sub(r'[^\d.\-]', '', value)
    
    try:
        return float(value)
    except:
        return np.nan

# Apply cleaning to numeric columns
for col in numeric_columns:
    if any(keyword in col.lower() for keyword in ['stream', 'follower', 'popularity', 'duration', 'release', 'date']):
        print(f"\nCleaning column: {col}")
        print(f"  Sample values before: {df[col].head(10).tolist()}")
        df[col] = df[col].apply(clean_numeric)
        print(f"  Sample values after: {df[col].head(10).tolist()}")
        print(f"  Missing after cleaning: {df[col].isnull().sum()}")

print("\nTask 2 Complete: Numeric values cleaned.\n")


# ============================================================
# TASK 3: Standardize Text
# ============================================================
print("=" * 60)
print("TASK 3: Standardize Text")
print("=" * 60)

# Find text columns to standardize
text_columns = df.select_dtypes(include=['object']).columns.tolist()
print(f"Text columns: {text_columns}")

# Function to standardize text
def standardize_text(value):
    """Standardize text values"""
    if pd.isna(value) or value == '':
        return value
    
    # Convert to string and strip whitespace
    value = str(value).strip()
    
    # Standardize common variations
    value_lower = value.lower()
    
    # Gender standardization
    gender_map = {
        'f': 'Female', 'female': 'Female', 'woman': 'Female', 'w': 'Female',
        'm': 'Male', 'male': 'Male', 'man': 'Male', 'b': 'Male',
        'non-binary': 'Non-Binary', 'nonbinary': 'Non-Binary', 'nb': 'Non-Binary'
    }
    if value_lower in gender_map:
        return gender_map[value_lower]
    
    # Country standardization
    country_map = {
        'us': 'United States', 'usa': 'United States', 'united states': 'United States',
        'uk': 'United Kingdom', 'united kingdom': 'United Kingdom', 'england': 'United Kingdom',
        'deutschland': 'Germany', 'germany': 'Germany',
        'brasil': 'Brazil', 'brazil': 'Brazil',
        'españa': 'Spain', 'spain': 'Spain',
        'france': 'France', 'francia': 'France',
        'italia': 'Italy', 'italy': 'Italy',
        'japan': 'Japan', 'japon': 'Japan',
        'korea': 'South Korea', 'korean': 'South Korea',
    }
    if value_lower in country_map:
        return country_map[value_lower]
    
    # Genre standardization (common variations)
    genre_map = {
        'hip hop': 'Hip-Hop', 'hip-hop': 'Hip-Hop', 'hiphop': 'Hip-Hop',
        'r&b': 'R&B', 'rnb': 'R&B', 'r and b': 'R&B',
        'pop': 'Pop', 'pop music': 'Pop',
        'rock': 'Rock', 'rock music': 'Rock',
        'electronic': 'Electronic', 'edm': 'Electronic', 'dance': 'Electronic',
        'latin': 'Latin', 'reggaeton': 'Latin',
        'country': 'Country', 'country music': 'Country',
        'jazz': 'Jazz', 'jazz music': 'Jazz',
        'classical': 'Classical', 'classic': 'Classical',
    }
    if value_lower in genre_map:
        return genre_map[value_lower]
    
    # Return title case for other text
    return value.title()

# Apply standardization to text columns
for col in text_columns:
    if any(keyword in col.lower() for keyword in ['sex', 'gender', 'country', 'genre', 'type', 'origin']):
        print(f"\nStandardizing column: {col}")
        print(f"  Unique values before: {df[col].nunique()}")
        df[col] = df[col].apply(standardize_text)
        print(f"  Unique values after: {df[col].nunique()}")
        print(f"  Top values: {df[col].value_counts().head(10).to_dict()}")

print("\nTask 3 Complete: Text values standardized.\n")


# ============================================================
# TASK 4: Impute & Dedupe
# ============================================================
print("=" * 60)
print("TASK 4: Impute & Dedupe")
print("=" * 60)

# Find stream/follower columns for imputation
stream_cols = [col for col in df.columns if 'stream' in str(col).lower() or 'follower' in str(col).lower()]
print(f"Stream/follower columns: {stream_cols}")

# Impute missing stream values
for col in stream_cols:
    if col in df.columns:
        missing_before = df[col].isnull().sum()
        if missing_before > 0:
            # Impute with median (more robust to outliers)
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)
            print(f"Imputed {col}: {missing_before} missing values replaced with median ({median_value:,.0f})")

# Find artist name column for deduplication
artist_cols = [col for col in df.columns if 'artist' in str(col).lower() or 'name' in str(col).lower()]
print(f"\nArtist columns: {artist_cols}")

if artist_cols:
    artist_col = artist_cols[0]
    
    # Check for duplicates
    print(f"\nDuplicate check:")
    print(f"  Total rows: {len(df)}")
    print(f"  Unique artists: {df[artist_col].nunique()}")
    
    # Find potential duplicates (case-insensitive)
    df['artist_lower'] = df[artist_col].str.lower().str.strip()
    duplicate_mask = df.duplicated(subset=['artist_lower'], keep='first')
    print(f"  Potential duplicates: {duplicate_mask.sum()}")
    
    # Show some examples
    if duplicate_mask.sum() > 0:
        print(f"\nSample duplicates:")
        duplicate_artists = df[duplicate_mask]['artist_lower'].unique()[:10]
        for artist in duplicate_artists:
            matches = df[df['artist_lower'] == artist][artist_col].tolist()
            print(f"  '{artist}': {matches}")
    
    # Remove duplicates (keep first occurrence)
    df = df.drop_duplicates(subset=['artist_lower'], keep='first')
    df = df.drop(columns=['artist_lower'])
    print(f"\nAfter removing duplicates: {df.shape}")

print("\nTask 4 Complete: Missing values imputed, duplicates removed.\n")


# ============================================================
# TASK 5: Summary & Export
# ============================================================
print("=" * 60)
print("TASK 5: Summary & Export")
print("=" * 60)

print("\n" + "=" * 60)
print("SPOTIFY ARTIST DATA - CLEANING SUMMARY")
print("=" * 60)

print(f"\nDataset Overview:")
print(f"  Final Shape: {df.shape}")
print(f"  Total Artists: {df[artist_col].nunique() if artist_cols else 'N/A'}")

print(f"\nMissing Values After Cleaning:")
missing = df.isnull().sum()
print(missing[missing > 0].to_string() if missing.any() else "  No missing values")

print(f"\nData Types After Cleaning:")
print(df.dtypes.to_string())

print(f"\nSample Cleaned Data:")
print(df.head(10).to_string())

# Export cleaned dataset
output_file = 'cleaned_spotify_artists.csv'
df.to_csv(output_file, index=False)
print(f"\nCleaned dataset exported to: {output_file}")

print("\n" + "=" * 60)
print("MODULE 5 COMPLETE")
print("=" * 60)
print(f"Tasks Completed:")
print(f"  1. Data loaded and inspected")
print(f"  2. Numeric values cleaned")
print(f"  3. Text values standardized")
print(f"  4. Missing values imputed, duplicates removed")
print(f"  5. Summary generated and data exported")
print("=" * 60)
