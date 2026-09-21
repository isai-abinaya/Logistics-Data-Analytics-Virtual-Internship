"""
Week 4 - Logistics Optimization

Purpose:
Use predicted delivery times and shipment information to
identify practical logistics optimization opportunities.

The script demonstrates:
1. High-delay shipment identification
2. Resource allocation
3. Transport-mode comparison
4. Cost minimization logic
5. Simple route-priority scoring

This is a decision-support example, not a production routing engine.
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------
# 1. Load Dataset and Predictions
# ---------------------------------------------------------

DATA_FILE = "hypothetical_logistics_dataset.csv"
PREDICTION_FILE = "delivery_time_predictions.csv"

df = pd.read_csv(DATA_FILE)
predictions = pd.read_csv(PREDICTION_FILE)


# Add prediction values to the test-prediction table.
# In a real system, predictions would be joined using
# a unique shipment/order ID.
print("Dataset loaded:", df.shape)
print("Prediction records:", predictions.shape)


# ---------------------------------------------------------
# 2. Identify High-Delay Shipments
# ---------------------------------------------------------

if "delivery_delay_days" in df.columns:

    high_delay_threshold = df["delivery_delay_days"].quantile(0.75)

    high_delay_shipments = df[
        df["delivery_delay_days"] >= high_delay_threshold
    ].copy()

    print("\nHigh-delay shipment threshold:",
          round(high_delay_threshold, 2))

    print("\nHigh-delay shipments:")
    print(high_delay_shipments.head())


# ---------------------------------------------------------
# 3. Resource Allocation
# ---------------------------------------------------------

# Calculate average shipment volume by transport mode.
resource_summary = (
    df.groupby("transport_mode")
      .agg(
          total_shipments=("shipment_volume", "sum"),
          average_volume=("shipment_volume", "mean"),
          average_cost=("transport_cost", "mean"),
          average_delivery_time=("delivery_time_days", "mean")
      )
      .reset_index()
)

print("\nResource Allocation Summary:")
print(resource_summary)


# ---------------------------------------------------------
# 4. Transport Mode Cost Analysis
# ---------------------------------------------------------

mode_cost = (
    df.groupby("transport_mode")["transport_cost"]
      .mean()
      .sort_values()
)

print("\nAverage Cost by Transport Mode:")
print(mode_cost)


# ---------------------------------------------------------
# 5. Route / Shipment Priority Score
# ---------------------------------------------------------

# A simple priority score can combine distance, shipment
# volume and delivery delay.

df["priority_score"] = (
    0.4 * df["distance_km"].rank(pct=True)
    + 0.3 * df["shipment_volume"].rank(pct=True)
    + 0.3 * df["delivery_delay_days"].rank(pct=True)
)

priority_shipments = df.sort_values(
    "priority_score",
    ascending=False
)

print("\nHigh-priority shipments:")
print(
    priority_shipments[
        [
            "region",
            "transport_mode",
            "shipment_volume",
            "distance_km",
            "delivery_delay_days",
            "priority_score"
        ]
    ].head(10)
)


# ---------------------------------------------------------
# 6. Cost Minimization Scenario
# ---------------------------------------------------------

# Estimate potential savings by comparing the current
# average transport cost with the lowest average cost
# transport mode.

average_cost_by_mode = (
    df.groupby("transport_mode")["transport_cost"]
      .mean()
)

lowest_cost_mode = average_cost_by_mode.idxmin()
lowest_average_cost = average_cost_by_mode.min()
current_average_cost = df["transport_cost"].mean()

potential_saving_per_shipment = (
    current_average_cost - lowest_average_cost
)

print("\nCost Optimization:")
print("Current average cost:",
      round(current_average_cost, 2))

print("Lowest average-cost mode:",
      lowest_cost_mode)

print("Lowest average cost:",
      round(lowest_average_cost, 2))

print("Potential saving per shipment:",
      round(potential_saving_per_shipment, 2))


# ---------------------------------------------------------
# 7. Optimization Recommendations
# ---------------------------------------------------------

print("\nLogistics Optimization Recommendations:")
print("1. Prioritize shipments with high delay and long distance.")
print("2. Allocate additional resources to high-volume regions.")
print("3. Compare transport modes using cost and delivery time.")
print("4. Use predicted delivery time to identify potential delays early.")
print("5. Consolidate suitable shipments where possible to reduce cost.")
print("6. Review high-cost routes and investigate alternative transport modes.")


# ---------------------------------------------------------
# 8. Optimization Pseudocode
# ---------------------------------------------------------

optimization_pseudocode = """
FOR each shipment:
    predict delivery time

    IF predicted delivery time > target:
        mark shipment as high priority

GROUP shipments by region and transport mode

FOR each region:
    calculate shipment volume
    calculate average delivery time
    calculate average transport cost

ALLOCATE additional resources to high-volume/high-delay regions

FOR each route:
    compare transport cost
    compare expected delivery time
    compare available capacity

SELECT feasible option that minimizes:
    total logistics cost + delay penalty

RETURN optimized resource and route recommendations
"""

print("\nOptimization Pseudocode:")
print(optimization_pseudocode)
