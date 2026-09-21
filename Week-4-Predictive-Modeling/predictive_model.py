"""
Week 4 - Predictive Modeling for Logistics
Dataset: hypothetical_logistics_dataset.csv

Models:
1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

Evaluation:
- MAE
- RMSE
- R²
- 5-Fold Cross-Validation
- Random Forest Hyperparameter Tuning

Target:
    delivery_time_days
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------
# 1. Load Dataset
# ---------------------------------------------------------

DATA_FILE = "hypothetical_logistics_dataset.csv"

df = pd.read_csv(DATA_FILE)

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())


# ---------------------------------------------------------
# 2. Basic Data Preparation
# ---------------------------------------------------------

df["shipment_date"] = pd.to_datetime(df["shipment_date"], errors="coerce")

# Create useful time-based features
df["month"] = df["shipment_date"].dt.month
df["day_of_week"] = df["shipment_date"].dt.dayofweek

target = "delivery_time_days"

# Exclude delivery_delay_days and delivery_status because
# they are outcomes related to the delivery process and can
# cause target leakage when predicting delivery time.
features = [
    "region",
    "transport_mode",
    "shipment_volume",
    "distance_km",
    "transport_cost",
    "customer_satisfaction",
    "month",
    "day_of_week"
]

X = df[features]
y = df[target]


# ---------------------------------------------------------
# 3. Train-Test Split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ---------------------------------------------------------
# 4. Preprocessing
# ---------------------------------------------------------

numeric_features = [
    "shipment_volume",
    "distance_km",
    "transport_cost",
    "customer_satisfaction",
    "month",
    "day_of_week"
]

categorical_features = [
    "region",
    "transport_mode"
]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# ---------------------------------------------------------
# 5. Define Models
# ---------------------------------------------------------

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42,
        max_depth=5
    ),

    "Random Forest": RandomForestRegressor(
        random_state=42,
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=2
    )
}


# ---------------------------------------------------------
# 6. Train and Evaluate Models
# ---------------------------------------------------------

results = []

for model_name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)
    print("MAE :", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R²  :", round(r2, 4))


results_df = pd.DataFrame(results)

print("\nModel Comparison:")
print(results_df)


# ---------------------------------------------------------
# 7. 5-Fold Cross-Validation
# ---------------------------------------------------------

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("\n5-Fold Cross-Validation Results:")

for model_name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="neg_root_mean_squared_error"
    )

    rmse_scores = -cv_scores

    print(
        f"{model_name}: "
        f"Mean RMSE = {rmse_scores.mean():.4f}, "
        f"Std = {rmse_scores.std():.4f}"
    )


# ---------------------------------------------------------
# 8. Random Forest Hyperparameter Tuning
# ---------------------------------------------------------

rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(random_state=42))
])

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [5, 8, None],
    "model__min_samples_leaf": [1, 2, 4]
}

grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\nBest Random Forest Parameters:")
print(grid_search.best_params_)

print(
    "Best Cross-Validation RMSE:",
    round(-grid_search.best_score_, 4)
)


# ---------------------------------------------------------
# 9. Evaluate Tuned Random Forest
# ---------------------------------------------------------

best_model = grid_search.best_estimator_

tuned_predictions = best_model.predict(X_test)

tuned_mae = mean_absolute_error(
    y_test,
    tuned_predictions
)

tuned_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        tuned_predictions
    )
)

tuned_r2 = r2_score(
    y_test,
    tuned_predictions
)

print("\nTuned Random Forest Test Results:")
print("MAE :", round(tuned_mae, 4))
print("RMSE:", round(tuned_rmse, 4))
print("R²  :", round(tuned_r2, 4))


# ---------------------------------------------------------
# 10. Example Delivery-Time Predictions
# ---------------------------------------------------------

sample_predictions = pd.DataFrame({
    "Actual_Delivery_Time": y_test.values,
    "Predicted_Delivery_Time": tuned_predictions
})

print("\nSample Predictions:")
print(sample_predictions.head(10))


# ---------------------------------------------------------
# 11. Save Predictions
# ---------------------------------------------------------

sample_predictions.to_csv(
    "delivery_time_predictions.csv",
    index=False
)

print("\nPrediction file saved as delivery_time_predictions.csv")
