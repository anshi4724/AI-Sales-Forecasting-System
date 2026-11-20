
import pandas as pd

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    STORE_ID,
    PRODUCT_FAMILY
)

# 1. load dataset
def load_train_data():
    """
    Load the historical sales dataset.
    """

    file_path = RAW_DATA_DIR / "train.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}\n"
            "Please place train.csv inside data/raw/"
        )

    df = pd.read_csv(file_path)

    return df

# 2. INSPECT COMPLETE DATASET
def inspect_dataset(df):
    """
    Display basic information about the dataset.
    """

    print("\n========== DATASET SHAPE ==========")
    print(df.shape)

    print("\n========== FIRST 5 ROWS ==========")
    print(df.head())

    print("\n========== COLUMN NAMES ==========")
    print(df.columns.tolist())

    print("\n========== DATA TYPES ==========")
    print(df.dtypes)

    print("\n========== DATASET INFORMATION ==========")
    df.info()

    print("\n========== MISSING VALUES ==========")
    print(df.isnull().sum())

    print("\n========== DUPLICATE ROWS ==========")
    print(df.duplicated().sum())

    print("\n========== NUMERICAL SUMMARY ==========")
    print(df.describe())

# 3. ANALYZE SELECTED STORE AND PRODUCT FAMILY
def analyze_selected_data(df):
    """
    Filter one store and one product family.
    """

    # Check whether required columns exist
    required_columns = [
        "date",
        "store_nbr",
        "family",
        "sales",
        "onpromotion"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    # Filter selected store and product family
    selected_df = df[
        (df["store_nbr"] == STORE_ID) &
        (df["family"] == PRODUCT_FAMILY)
    ].copy()

    # Check whether filtered data is empty
    if selected_df.empty:
        raise ValueError(
            f"No records found for Store {STORE_ID} "
            f"and Product Family {PRODUCT_FAMILY}"
        )

    # Sort records chronologically
    selected_df = selected_df.sort_values(
        by="date"
    ).reset_index(drop=True)

    # Check duplicate dates
    if selected_df["date"].duplicated().any():
        raise ValueError("Duplicate dates found in selected dataset!")


    # Check whether every calendar date is present
    expected_dates = pd.date_range(
        start=selected_df["date"].min(),
        end=selected_df["date"].max(),
        freq="D"
    )

    actual_dates = pd.DatetimeIndex(selected_df["date"])

    missing_dates = expected_dates.difference(actual_dates)

    print("\n========== DATE CONTINUITY CHECK ==========")
    print("Expected calendar days:", len(expected_dates))
    print("Actual records:", len(selected_df))
    print("Missing dates:", len(missing_dates))
    print("Missing date list:", missing_dates.tolist())

    print(
        "Duplicate dates:",
        selected_df["date"].duplicated().sum()
    )

    # Analyze zero-sales records
    print("\n========== ZERO SALES DATES ==========")

    zero_sales = selected_df[
        selected_df["sales"] == 0
    ]

    print(zero_sales[
        ["date", "sales", "onpromotion"]
    ])

    print("\nTotal zero-sales records:", len(zero_sales))

    # Find highest-sales record
    print("\n========== HIGHEST SALES RECORD ==========")

    highest_sales = selected_df.loc[
        selected_df["sales"].idxmax()
    ]

    print(
        highest_sales[
            ["date", "sales", "onpromotion"]
        ]
    )

    # Display selected dataset information
    print("\n========== SELECTED DATASET ==========")

    print("Store:", STORE_ID)
    print("Product Family:", PRODUCT_FAMILY)

    print("\nSelected dataset shape:")
    print(selected_df.shape)

    print("\nFirst 5 selected records:")
    print(selected_df.head())

    print("\nDate range:")
    print("Start:", selected_df["date"].min())
    print("End:", selected_df["date"].max())

    print("\nSales statistics:")
    print(selected_df["sales"].describe())

    print("\nPromotion statistics:")
    print(selected_df["onpromotion"].describe())

    return selected_df

# 4. SAVE PROCESSED DATASET
def save_processed_data(selected_df):
    """
    Save the filtered dataset inside data/processed/.
    """

    output_path = (
        PROCESSED_DATA_DIR / "store_1_grocery_sales.csv"
    )

    selected_df.to_csv(
        output_path,
        index=False
    )

    print("\n========== DATA SAVING ==========")
    print("Processed dataset saved successfully!")
    print("File location:", output_path)

# 5. MAIN FUNCTION
def main():
    # Load original dataset
    df = load_train_data()

    # Convert date column into datetime format
    df["date"] = pd.to_datetime(
        df["date"],
        errors="raise"
    )

    # Inspect complete dataset
    inspect_dataset(df)

    # Filter and analyze selected data
    selected_df = analyze_selected_data(df)

    # Save processed dataset
    save_processed_data(selected_df)

# 6. EXECUTION ENTRY POINT

if __name__ == "__main__":
    main()

"""summary:- phase 2
WHAT THIS CODE DOES
===================

1. load_train_data()
   Loads the original CSV.

2. inspect_dataset()
   Checks shape, columns, data types, missing values
   and duplicates.

3. analyze_selected_data()
   Filters Store 1 and GROCERY I and analyzes dates
   and sales.

4. save_processed_data()
   Saves the filtered CSV.

5. main()
   Runs the complete workflow.
"""

"""
============================================================
              AI SALES FORECASTING PROJECT- phase steps
============================================================

Project Phase                  Libraries Used
------------------------------------------------------------

1. Data Loading                Pandas, pathlib

2. Data Inspection             Pandas, NumPy

3. Exploratory Data Analysis   Pandas, Matplotlib, Seaborn

4. Data Preprocessing          Pandas, NumPy, Scikit-learn

5. Baseline Model              Scikit-learn

6. LSTM Development            PyTorch

7. Model Training              PyTorch

8. Model Evaluation            Scikit-learn, Matplotlib

9. Model Persistence           PyTorch, Joblib

10. Backend Development        FastAPI, Pydantic, Uvicorn

11. Frontend Development       React, Axios

12. Testing and Deployment     Pytest, Docker, Git

============================================================
"""