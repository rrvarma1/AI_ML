"""
Create a modeling-ready dataset from data/processed/features.csv.

Outputs:
    data/processed/model_table.csv
    data/processed/model_table.parquet
"""

# Summary:
# 1. Purpose: Converts engineered patient features into a modeling-ready table.
# 2. What it does: Removes identifiers, fills missing values, and one-hot encodes categories.
# 3. Invoked by: Data preparation operators after src/build_features.py completes.
# 4. Main functions/classes: build_model_table and main.
# 5. Validations/controls: Uses numeric medians and an Unknown category before writing CSV and Parquet.

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "features.csv"
DEFAULT_CSV_OUTPUT = PROJECT_ROOT / "data" / "processed" / "model_table.csv"
DEFAULT_PARQUET_OUTPUT = PROJECT_ROOT / "data" / "processed" / "model_table.parquet"


def build_model_table(features: pd.DataFrame) -> pd.DataFrame:
    model_df = features.copy()

    drop_columns = [
        "patient_id",
        "registration_date",
        "first_visit_date",
        "last_visit_date",
    ]
    model_df = model_df.drop(columns=[c for c in drop_columns if c in model_df.columns])

    categorical_columns = [
        c
        for c in ["gender", "city", "insurance_provider"]
        if c in model_df.columns
    ]

    numeric_columns = model_df.select_dtypes(include=["number", "bool"]).columns.tolist()
    for column in numeric_columns:
        model_df[column] = model_df[column].fillna(model_df[column].median())

    for column in categorical_columns:
        model_df[column] = model_df[column].fillna("Unknown")

    model_df = pd.get_dummies(
        model_df,
        columns=categorical_columns,
        drop_first=False,
        dtype=int,
    )

    return model_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build modeling dataset from data/processed/features.csv."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input feature CSV path.")
    parser.add_argument("--csv-output", default=DEFAULT_CSV_OUTPUT, help="Output CSV path.")
    parser.add_argument(
        "--parquet-output",
        default=DEFAULT_PARQUET_OUTPUT,
        help="Output Parquet path.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    csv_output_path = Path(args.csv_output)
    parquet_output_path = Path(args.parquet_output)

    csv_output_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_output_path.parent.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(input_path)
    model_table = build_model_table(features)

    model_table.to_csv(csv_output_path, index=False)
    model_table.to_parquet(parquet_output_path, index=False)

    print(f"Modeling CSV created: {csv_output_path}")
    print(f"Modeling Parquet created: {parquet_output_path}")
    print(f"Rows: {model_table.shape[0]}")
    print(f"Columns: {model_table.shape[1]}")
    print("Preview:")
    print(model_table.head(10))


if __name__ == "__main__":
    main()
