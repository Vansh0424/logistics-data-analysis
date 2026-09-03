"""
Week 2 - Data Collection, Cleaning and Preprocessing
Logistics Data Analysis Project

Dataset:
DataCo Smart Supply Chain Dataset
Expected input:
data/DataCoSupplyChainDataset.csv
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

INPUT_FILE = "data/DataCoSupplyChainDataset.csv"
OUTPUT_FILE = "data/cleaned_logistics_data.csv"

# 1. Load data
df = pd.read_csv(INPUT_FILE, encoding="latin1")

print("Original shape:", df.shape)

# 2. Basic inspection
print("\nData types:")
print(df.dtypes.value_counts())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nMissing values:")
print(df.isna().sum().sort_values(ascending=False).head(10))

# 3. Remove fields that are not useful for the operational analysis.
# Product Description is completely missing; Order Zipcode is mostly missing.
# Customer contact/password fields are not required and should not be used.
drop_columns = [
    "Product Description",
    "Order Zipcode",
    "Customer Email",
    "Customer Password",
    "Product Image"
]

df = df.drop(columns=[c for c in drop_columns if c in df.columns])

# 4. Handle small amounts of missing customer information.
# These fields are not central to the logistics model, so missing rows are removed.
for col in ["Customer Lname", "Customer Zipcode"]:
    if col in df.columns:
        df = df.dropna(subset=[col])

# 5. Remove duplicate records
df = df.drop_duplicates()

# 6. Convert date fields
date_columns = ["order date (DateOrders)", "shipping date (DateOrders)"]
for col in date_columns:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

# 7. Create a useful operational feature
df["delivery_delay_days"] = (
    df["Days for shipping (real)"]
    - df["Days for shipment (scheduled)"]
)

# 8. IQR-based outlier detection.
# We flag outliers for review rather than automatically deleting valid transactions.
numeric_cols = [
    "Benefit per order",
    "Sales per customer",
    "Order Item Discount",
    "Order Item Product Price",
    "Order Item Quantity",
    "Sales",
    "Order Item Total",
    "Order Profit Per Order",
    "delivery_delay_days"
]

outlier_summary = {}

for col in numeric_cols:
    if col not in df.columns:
        continue

    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    mask = (df[col] < lower) | (df[col] > upper)
    outlier_summary[col] = int(mask.sum())

print("\nPotential outliers by IQR:")
for col, count in outlier_summary.items():
    print(f"{col}: {count}")

# 9. Min-Max normalization for selected continuous variables.
# Keep the original variables and create normalized versions for later ML use.
scale_cols = [
    "Sales",
    "Order Item Total",
    "Order Profit Per Order",
    "Order Item Discount"
]

scale_cols = [c for c in scale_cols if c in df.columns]

if scale_cols:
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(df[scale_cols])
    normalized_df = pd.DataFrame(
        normalized,
        columns=[f"{c}_normalized" for c in scale_cols],
        index=df.index
    )
    df = pd.concat([df, normalized_df], axis=1)

# 10. Final quality check
print("\nFinal shape:", df.shape)
print("Remaining duplicate rows:", df.duplicated().sum())
print("\nRemaining missing values (top 10):")
print(df.isna().sum().sort_values(ascending=False).head(10))

# 11. Save cleaned/preprocessed dataset
df.to_csv(OUTPUT_FILE, index=False)

print(f"\nCleaned dataset saved to: {OUTPUT_FILE}")
