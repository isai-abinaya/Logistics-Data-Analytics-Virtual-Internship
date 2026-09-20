# Week 2 – Data Collection, Cleaning and Preprocessing

## 📌 Overview

This week focuses on data collection, data cleaning, and preprocessing for logistics data analysis.

The **Brazilian E-Commerce Public Dataset by Olist** is used to simulate a logistics data preprocessing workflow for a last-mile e-commerce delivery scenario.

The main goal is to prepare a clean and reliable dataset for further exploratory analysis, visualization, and predictive modeling.

## 🎯 Objectives

- Collect and inspect the logistics dataset.
- Understand the structure and quality of the data.
- Identify missing values and duplicate records.
- Detect invalid and inconsistent values.
- Identify potential outliers.
- Clean and transform the data.
- Aggregate item-level data into an order-level dataset.
- Create useful delivery-related features.
- Standardize selected numerical variables.
- Validate the processed dataset.

## 🛠️ Tools and Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Jupyter Notebook

## 📊 Dataset

**Dataset:** Brazilian E-Commerce Public Dataset by Olist

The dataset contains information related to:

- Orders
- Customers
- Sellers
- Products
- Order items
- Reviews
- Freight
- Delivery dates
- Geolocation

## 🧹 Data Preprocessing Steps

The following preprocessing techniques were implemented:

1. **Data Loading**
   - Loaded the Olist CSV datasets using Pandas.

2. **Data Inspection**
   - Checked columns, data types, dimensions, and sample records.

3. **Date and Time Conversion**
   - Converted logistics timestamp columns into proper datetime format.

4. **Missing-Value Analysis**
   - Identified missing values and considered appropriate treatment based on the meaning of each field.

5. **Duplicate Handling**
   - Checked for exact duplicate records and duplicate order IDs.

6. **Data Validation**
   - Checked for invalid values and logical inconsistencies.

7. **Outlier Detection**
   - Used the Interquartile Range (IQR) method to identify potential outliers in numerical variables such as price and freight value.

8. **Numeric Cleaning**
   - Converted numerical columns to appropriate data types and handled invalid negative values.

9. **Data Aggregation**
   - Aggregated order-item information to the order level.

10. **Feature Engineering**
    - Created:
      - `item_count`
      - `total_price`
      - `total_freight_value`
      - `review_score`
      - `delivery_time_days`
      - `delivery_delay_days`

11. **Standardization**
    - Applied `StandardScaler` to selected numerical features for machine-learning preparation.

12. **Data Quality Validation**
    - Performed final checks for duplicates, missing order IDs, and invalid delivery values.

## 📁 Files

### `data_preprocessing.ipynb`

Jupyter Notebook containing the complete Python implementation of the Week 2 preprocessing workflow.

### `Week2_Report.docx`

Detailed report explaining the data collection, cleaning methods, preprocessing techniques, methodology, and impact of data quality on logistics analysis.

## 🔄 Workflow

```text
Data Collection
       ↓
Data Inspection
       ↓
Data Cleaning
       ↓
Missing Value Analysis
       ↓
Duplicate Handling
       ↓
Invalid Value Validation
       ↓
Outlier Detection
       ↓
Data Aggregation
       ↓
Feature Engineering
       ↓
Standardization
       ↓
Final Validation
       ↓
Clean Dataset
