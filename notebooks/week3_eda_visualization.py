"""
Week 3 - Advanced Data Analysis and Visualization in Logistics
Hypothetical logistics dataset, EDA, visualization and insights.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Create reproducible hypothetical logistics data
rng = np.random.default_rng(42)
n = 500

regions = rng.choice(["North","South","East","West"], n)
modes = rng.choice(["Road","Rail","Air","Sea"], n, p=[0.55,0.18,0.12,0.15])
categories = rng.choice(["Electronics","Apparel","Food","Industrial"], n)

shipment_volume = np.clip(rng.normal(520, 180, n), 80, 1200)
distance_km = np.clip(rng.gamma(3.0, 220, n), 50, 1800)

base_days = 1.2 + distance_km / 450
mode_adj = pd.Series(modes).map(
    {"Road":0.7, "Rail":1.8, "Air":-1.0, "Sea":3.0}
).to_numpy()

delivery_time = np.clip(
    base_days + mode_adj + shipment_volume / 900 +
    rng.normal(0, 1.2, n), 0.5, None
)

transport_cost = np.clip(
    120 + distance_km * 0.72 + shipment_volume * 0.42 +
    pd.Series(modes).map(
        {"Road":0, "Rail":-45, "Air":210, "Sea":-85}
    ).to_numpy() + rng.normal(0, 90, n),
    80, None
)

delay_risk = np.clip(
    0.10 + (delivery_time > np.percentile(delivery_time, 65)) * 0.18 +
    (shipment_volume > 700) * 0.10 + rng.normal(0, 0.07, n),
    0.01, 0.95
)

df = pd.DataFrame({
    "Shipment_ID": np.arange(10001, 10001+n),
    "Region": regions,
    "Transport_Mode": modes,
    "Product_Category": categories,
    "Shipment_Volume_kg": shipment_volume,
    "Distance_km": distance_km,
    "Delivery_Time_Days": delivery_time,
    "Transportation_Cost_INR": transport_cost,
    "Delay_Risk": delay_risk
})

# 2. Basic EDA
print("Shape:", df.shape)
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isna().sum())
print("\nDuplicates:", df.duplicated().sum())
print("\nDescriptive statistics:\n", df.describe())

# 3. Central tendency
for col in ["Shipment_Volume_kg","Distance_km","Delivery_Time_Days",
            "Transportation_Cost_INR","Delay_Risk"]:
    print(f"{col}: mean={df[col].mean():.2f}, "
          f"median={df[col].median():.2f}, "
          f"std={df[col].std():.2f}")

# 4. Distribution
plt.figure(figsize=(8,5))
plt.hist(df["Delivery_Time_Days"], bins=25, edgecolor="black")
plt.xlabel("Delivery Time (days)")
plt.ylabel("Number of Shipments")
plt.title("Distribution of Delivery Times")
plt.tight_layout()
plt.show()

# 5. Relationship: distance and cost
plt.figure(figsize=(8,5))
plt.scatter(df["Distance_km"], df["Transportation_Cost_INR"], alpha=0.45)
plt.xlabel("Distance (km)")
plt.ylabel("Transportation Cost (INR)")
plt.title("Distance vs Transportation Cost")
plt.tight_layout()
plt.show()

# 6. Compare transport modes
mode_summary = df.groupby("Transport_Mode").agg(
    Avg_Delivery_Days=("Delivery_Time_Days","mean"),
    Avg_Cost_INR=("Transportation_Cost_INR","mean"),
    Avg_Delay_Risk=("Delay_Risk","mean")
)
print("\nMode summary:\n", mode_summary)

mode_summary["Avg_Delivery_Days"].plot(kind="bar")
plt.ylabel("Average Delivery Time (days)")
plt.title("Average Delivery Time by Transport Mode")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 7. Volume vs delivery time
plt.figure(figsize=(8,5))
plt.scatter(df["Shipment_Volume_kg"], df["Delivery_Time_Days"], alpha=0.45)
plt.xlabel("Shipment Volume (kg)")
plt.ylabel("Delivery Time (days)")
plt.title("Shipment Volume vs Delivery Time")
plt.tight_layout()
plt.show()

# 8. Correlation analysis
numeric_cols = ["Shipment_Volume_kg","Distance_km","Delivery_Time_Days",
                "Transportation_Cost_INR","Delay_Risk"]
corr = df[numeric_cols].corr()
print("\nCorrelation matrix:\n", corr)

plt.figure(figsize=(8,6))
plt.imshow(corr, aspect="auto")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
plt.yticks(range(len(corr.columns)), corr.columns)
plt.colorbar(label="Correlation")
plt.title("Correlation Matrix")
for i in range(len(corr)):
    for j in range(len(corr)):
        plt.text(j, i, f"{corr.iloc[i,j]:.2f}",
                 ha="center", va="center", fontsize=8)
plt.tight_layout()
plt.show()

# 9. Example operational insight
print("\nRecommendation:")
print("- Use transport-mode comparisons to balance speed and cost.")
print("- Prioritize long-distance shipments for route/cost optimization.")
print("- Monitor high-volume shipments because they can increase delivery time.")
