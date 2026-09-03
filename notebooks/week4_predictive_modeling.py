"""
Week 4 - Predictive Modeling and Optimization in Logistics
Predict delivery time using logistics features and compare models.
"""

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Week 3 hypothetical dataset
df = pd.read_csv("data/week3_hypothetical_logistics_dataset.csv")

target = "Delivery_Time_Days"
features = [
    "Region", "Transport_Mode", "Product_Category",
    "Shipment_Volume_kg", "Distance_km"
]

X = df[features]
y = df[target]

categorical = ["Region", "Transport_Mode", "Product_Category"]
numeric = ["Shipment_Volume_kg", "Distance_km"]

preprocess = ColumnTransformer([
    ("num", StandardScaler(), numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
])

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, max_depth=12, random_state=42
    )
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

for name, model in models.items():
    pipe = Pipeline([("preprocess", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    r2 = r2_score(y_test, pred)

    cv_rmse = -cross_val_score(
        pipe, X_train, y_train, cv=5,
        scoring="neg_root_mean_squared_error"
    ).mean()

    print(f"\n{name}")
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R2: {r2:.3f}")
    print(f"5-fold CV RMSE: {cv_rmse:.3f}")

# Optimization idea:
# For a shipment, test each available transport mode and select
# the mode with the lowest predicted delivery time subject to
# business constraints such as budget and service level.
