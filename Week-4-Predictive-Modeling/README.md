# Week 4 – Predictive Modeling and Optimization

This folder contains the technical implementation for Week 4 of the Logistics Data Analytics Virtual Internship.

## Files

| File | Description |
|---|---|
| `predictive_model.py` | Builds and evaluates Linear Regression, Decision Tree, and Random Forest models |
| `optimization.py` | Performs logistics analysis and demonstrates resource allocation, route-priority, and cost-minimization logic |
| `Week4_Predictive_Modeling_Logistics_Report.docx` | Complete Week 4 report |

## Predictive Modeling

The target variable is:

`delivery_time_days`

The model workflow includes:

- Data preprocessing
- Feature engineering
- Train/test split
- Linear Regression
- Decision Tree Regression
- Random Forest Regression
- MAE
- RMSE
- R²
- 5-fold cross-validation
- Random Forest hyperparameter tuning

## Optimization

The optimization script demonstrates:

- High-delay shipment identification
- Resource allocation
- Transport-mode cost analysis
- Shipment priority scoring
- Cost-minimization logic
- Route and logistics recommendations

## Dataset

The project uses the hypothetical logistics dataset from the internship, containing shipment date, region, transport mode, shipment volume, distance, delivery time, delivery delay, transport cost, customer satisfaction, and delivery status.

## How to Run

Place these files in the same folder:

```text
hypothetical_logistics_dataset.csv
predictive_model.py
optimization.py
```

Run:

```bash
python predictive_model.py
```

This creates:

```text
delivery_time_predictions.csv
```

Then run:

```bash
python optimization.py
```

## Python Libraries

```text
pandas
numpy
scikit-learn
```
