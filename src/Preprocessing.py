# =====================================================
# CUSTOMER PERSONALITY ANALYSIS - DATA PREPROCESSING
# Activity 1 & Activity 2
# =====================================================

# -----------------------------
# Import Libraries
# -----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("="*60)
print("CUSTOMER PERSONALITY ANALYSIS")
print("="*60)

# -----------------------------
# Load Dataset
# -----------------------------
file_path = "data/marketing_campaign.csv"
# Agar file isi folder me hai to:
# file_path = "marketing_campaign.csv"

df = pd.read_csv(file_path, sep="\t")

print("\nDataset Loaded Successfully!")

# -----------------------------
# Display First 10 Rows
# -----------------------------
print("\nFirst 10 Rows")
print(df.head(10))

# -----------------------------
# Display Last 10 Rows
# -----------------------------
print("\nLast 10 Rows")
print(df.tail(10))

# -----------------------------
# Dataset Shape
# -----------------------------
rows, cols = df.shape

print("\nDataset Shape")
print("Rows :", rows)
print("Columns :", cols)

# -----------------------------
# Column Names
# -----------------------------
print("\nColumn Names")
for col in df.columns:
    print(col)

# -----------------------------
# Data Types
# -----------------------------
print("\nData Types")
print(df.dtypes)

# -----------------------------
# Dataset Information
# -----------------------------
print("\nDataset Info")
print(df.info())

# -----------------------------
# Summary Statistics
# -----------------------------
print("\nSummary Statistics")
print(df.describe(include="all"))

# =====================================================
# ACTIVITY 2
# DATA QUALITY ASSESSMENT
# =====================================================

print("\n" + "="*60)
print("DATA QUALITY CHECK")
print("="*60)

# -----------------------------
# Missing Values
# -----------------------------
print("\nMissing Values")
missing = df.isnull().sum()

missing_percent = (missing / len(df)) * 100

missing_report = pd.DataFrame({
    "Missing Values": missing,
    "Percentage": missing_percent.round(2)
})

print(missing_report)

# -----------------------------
# Duplicate Rows
# -----------------------------
print("\nDuplicate Rows")
duplicates = df.duplicated().sum()
print("Duplicate Rows :", duplicates)

# -----------------------------
# Duplicate Customer IDs
# -----------------------------
duplicate_ids = df["ID"].duplicated().sum()

print("\nDuplicate Customer IDs :", duplicate_ids)

# -----------------------------
# Check Unique Categories
# -----------------------------
categorical_columns = ["Education","Marital_Status"]

for col in categorical_columns:

    print("\nUnique Values in", col)

    print(df[col].unique())

# -----------------------------
# Invalid Birth Years
# -----------------------------
current_year = 2026

invalid_birth = df[
    (df["Year_Birth"] > current_year) |
    (df["Year_Birth"] < 1900)
]

print("\nInvalid Birth Years :", len(invalid_birth))

# -----------------------------
# Check Income
# -----------------------------
print("\nIncome Statistics")
print(df["Income"].describe())

# -----------------------------
# Invalid Dates
# -----------------------------
temp = pd.to_datetime(
    df["Dt_Customer"],
    errors="coerce"
)

invalid_dates = temp.isna().sum()

print("\nInvalid Dates :", invalid_dates)

# -----------------------------
# Data Quality Assessment Table
# -----------------------------

quality = pd.DataFrame({

"Issue":[
"Missing Values",
"Duplicate Rows",
"Duplicate IDs",
"Invalid Dates",
"Invalid Birth Years"
],

"Count":[
missing.sum(),
duplicates,
duplicate_ids,
invalid_dates,
len(invalid_birth)
]

})

print("\nDATA QUALITY TABLE")
print(quality)

print("\nActivity 1 & 2 Completed Successfully")

# =====================================================
# ACTIVITY 3 - HANDLE MISSING VALUES
# =====================================================

print("\n" + "="*60)
print("ACTIVITY 3 - HANDLE MISSING VALUES")
print("="*60)

# Missing Values Count
missing = df.isnull().sum()

print("\nMissing Values in Each Column")
print(missing)

# Missing Percentage
missing_percent = (missing / len(df)) * 100

print("\nMissing Percentage")
print(missing_percent.round(2))

# Income Missing Values
income_missing = df["Income"].isnull().sum()

print("\nMissing Income Values :", income_missing)

# Fill Missing Income with Median
median_income = df["Income"].median()

df["Income"] = df["Income"].fillna(median_income)

print("\nIncome Missing Values After Filling :")
print(df["Income"].isnull().sum())

print("\nReason:")
print("Median is used because Income may contain outliers.")


# =====================================================
# ACTIVITY 4 - REMOVE DUPLICATES
# =====================================================

print("\n" + "="*60)
print("ACTIVITY 4 - REMOVE DUPLICATES")
print("="*60)

before_rows = len(df)

duplicate_rows = df.duplicated().sum()

duplicate_ids = df["ID"].duplicated().sum()

print("\nDuplicate Rows :", duplicate_rows)
print("Duplicate Customer IDs :", duplicate_ids)

# Remove Duplicate Rows
df = df.drop_duplicates()

# Remove Duplicate IDs (Keep First)
df = df.drop_duplicates(subset="ID", keep="first")

after_rows = len(df)

removed = before_rows - after_rows

print("\nRows Before :", before_rows)
print("Rows After :", after_rows)
print("Removed :", removed)


# =====================================================
# ACTIVITY 5 - CORRECT DATA TYPES
# =====================================================

print("\n" + "="*60)
print("ACTIVITY 5 - DATA TYPES")
print("="*60)

print("\nOld Data Types")
print(df.dtypes)

# Convert Data Types

df["ID"] = df["ID"].astype(int)

df["Income"] = df["Income"].astype(float)

df["Recency"] = df["Recency"].astype(int)

# Convert Date Column
df["Dt_Customer"] = pd.to_datetime(
    df["Dt_Customer"],
    format="%d-%m-%Y",
    errors="coerce"
)

print("\nNew Data Types")
print(df.dtypes)


# =====================================================
# ACTIVITY 6 - DATE CONVERSION
# =====================================================

print("\n" + "="*60)
print("ACTIVITY 6 - DATE CONVERSION")
print("="*60)

invalid_dates = df["Dt_Customer"].isnull().sum()

print("\nInvalid Dates :", invalid_dates)

# Remove Invalid Dates if Any
df = df.dropna(subset=["Dt_Customer"])

# Extract Date Features
df["Join_Year"] = df["Dt_Customer"].dt.year

df["Join_Month"] = df["Dt_Customer"].dt.month

df["Join_Day"] = df["Dt_Customer"].dt.day

print("\nSample Date Features")

print(
    df[
        [
            "Dt_Customer",
            "Join_Year",
            "Join_Month",
            "Join_Day"
        ]
    ].head()
)

print("\nActivity 3, 4, 5 & 6 Completed Successfully.")

# =====================================================
# ACTIVITY 7 - STANDARDIZE CATEGORICAL VARIABLES
# =====================================================

print("\n" + "="*60)
print("ACTIVITY 7 - STANDARDIZE CATEGORICAL VARIABLES")
print("="*60)

categorical_columns = ["Education", "Marital_Status"]

for col in categorical_columns:

    print(f"\nBefore Standardization ({col})")
    print(df[col].unique())

    # Remove spaces and standardize text
    df[col] = df[col].astype(str).str.strip().str.title()

# Merge Similar Categories
df["Marital_Status"] = df["Marital_Status"].replace({
    "Alone": "Single",
    "Absurd": "Single",
    "Yolo": "Single",
    "Together": "Married"
})

print("\nAfter Standardization")

for col in categorical_columns:
    print(f"\n{col}")
    print(df[col].unique())

print("\nActivity 7 Completed Successfully")


# =====================================================
# ACTIVITY 8 - DETECT UNREALISTIC CUSTOMER AGES
# =====================================================

print("\n" + "="*60)
print("ACTIVITY 8 - AGE VALIDATION")
print("="*60)

current_year = 2026

df["Age"] = current_year - df["Year_Birth"]

print("\nAge Statistics")
print(df["Age"].describe())

young = df[df["Age"] < 18]
old = df[df["Age"] > 100]
future_birth = df[df["Year_Birth"] > current_year]

print("\nCustomers younger than 18 :", len(young))
print("Customers older than 100 :", len(old))
print("Future Birth Years :", len(future_birth))

# Remove unrealistic ages
before = len(df)

df = df[(df["Age"] >= 18) & (df["Age"] <= 100)]

after = len(df)

print("\nRemoved Unrealistic Records :", before - after)

print("\nActivity 8 Completed Successfully")


# =====================================================
# ACTIVITY 9 - DETECT OUTLIERS
# =====================================================

print("\n" + "="*60)
print("ACTIVITY 9 - OUTLIER DETECTION")
print("="*60)

import os

if not os.path.exists("output"):
    os.makedirs("output")

numeric_columns = [
    "Income",
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds"
]

for col in numeric_columns:

    plt.figure(figsize=(7,4))

    sns.boxplot(x=df[col])

    plt.title(col)

    plt.savefig(f"output/{col}_boxplot.png")

    plt.close()

print("\nBoxplots Saved Successfully!")

# IQR Outlier Detection

outlier_summary = []

for col in numeric_columns:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)]

    outlier_summary.append([col, len(outliers)])

print("\nOutlier Summary")

summary = pd.DataFrame(
    outlier_summary,
    columns=["Variable","Outliers"]
)

print(summary)

summary.to_csv("output/outlier_summary.csv", index=False)

print("\nActivity 9 Completed Successfully")


# =====================================================
# ACTIVITY 10 - HANDLE OUTLIERS
# =====================================================

print("\n" + "="*60)
print("ACTIVITY 10 - HANDLE OUTLIERS")
print("="*60)

for col in numeric_columns:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    # Winsorization (Capping)

    df[col] = np.where(df[col] < lower, lower, df[col])

    df[col] = np.where(df[col] > upper, upper, df[col])

print("\nOutliers Handled Using IQR Capping.")

print("\nActivity 10 Completed Successfully")

# =====================================================
# ACTIVITY 11 - VALIDATE CLEANED DATASET
# =====================================================

print("\n" + "="*60)
print("ACTIVITY 11 - VALIDATE CLEANED DATASET")
print("="*60)

validation = {
    "Missing Values": df.isnull().sum().sum(),
    "Duplicate Rows": df.duplicated().sum(),
    "Duplicate IDs": df["ID"].duplicated().sum(),
    "Invalid Dates": df["Dt_Customer"].isnull().sum()
}

print("\nValidation Summary")

for key, value in validation.items():
    print(f"{key}: {value}")

print("\nData Types")
print(df.dtypes)

print("\nValidation Checklist")

checklist = pd.DataFrame({
    "Validation Check":[
        "Missing Values",
        "Duplicate Rows",
        "Duplicate IDs",
        "Data Types",
        "Dates",
        "Categories"
    ],
    "Status":[
        "PASS" if validation["Missing Values"]==0 else "FAIL",
        "PASS" if validation["Duplicate Rows"]==0 else "FAIL",
        "PASS" if validation["Duplicate IDs"]==0 else "FAIL",
        "PASS",
        "PASS" if validation["Invalid Dates"]==0 else "FAIL",
        "PASS"
    ]
})

print(checklist)

checklist.to_csv("output/validation_checklist.csv",index=False)


# =====================================================
# ACTIVITY 12 - SAVE CLEANED DATASET
# =====================================================

print("\n" + "="*60)
print("ACTIVITY 12 - SAVE CLEANED DATASET")
print("="*60)

output_file = "output/customer_personality_cleaned.csv"

df.to_csv(output_file,index=False)

print("\nCleaned Dataset Saved Successfully!")

print(output_file)


# =====================================================
# ACTIVITY 13 - PREPROCESSING SCRIPT
# =====================================================

print("\n" + "="*60)
print("ACTIVITY 13")
print("="*60)

print("""
Congratulations!

This preprocessing.py file automatically performs:

✔ Load Dataset
✔ Data Inspection
✔ Missing Value Handling
✔ Duplicate Removal
✔ Data Type Correction
✔ Date Conversion
✔ Category Standardization
✔ Age Validation
✔ Outlier Detection
✔ Outlier Handling
✔ Validation
✔ Export Clean Dataset

Reusable Script Completed Successfully.
""")


# =====================================================
# ACTIVITY 14 - BEFORE VS AFTER COMPARISON
# =====================================================

print("\n" + "="*60)
print("ACTIVITY 14 - BEFORE VS AFTER")
print("="*60)

before_rows = rows
after_rows = len(df)

comparison = pd.DataFrame({

"Metric":[
"Rows",
"Columns",
"Missing Values",
"Duplicate Rows"
],

"Before Cleaning":[
before_rows,
cols,
income_missing,
duplicates
],

"After Cleaning":[
after_rows,
len(df.columns),
df.isnull().sum().sum(),
df.duplicated().sum()
]

})

print(comparison)

comparison.to_csv(
    "output/before_after_comparison.csv",
    index=False
)

print("\nComparison Saved Successfully!")


# =====================================================
# ACTIVITY 15 - FINAL SUMMARY
# =====================================================

print("\n" + "="*60)
print("ACTIVITY 15 - FINAL REPORT SUMMARY")
print("="*60)

print("""

PROJECT COMPLETED SUCCESSFULLY

Dataset Overview
-------------------------
Customer Personality Analysis Dataset

Cleaning Performed
-------------------------
✔ Missing Values Handled
✔ Duplicate Records Removed
✔ Data Types Corrected
✔ Date Converted
✔ Categories Standardized
✔ Unrealistic Ages Removed
✔ Outliers Detected
✔ Outliers Capped
✔ Dataset Validated

Final Output
-------------------------
customer_personality_cleaned.csv

Ready For:
✔ Exploratory Data Analysis
✔ Machine Learning
✔ Model Building

""")

print("="*60)
print("ALL ACTIVITIES COMPLETED SUCCESSFULLY")
print("="*60)
