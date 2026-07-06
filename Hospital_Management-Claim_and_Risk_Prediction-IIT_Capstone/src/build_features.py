"""
Build patient-level engineered features from cleaned Oracle tables.

Default workflow:
    1. Execute notebooks/01_eda.ipynb so the source tables are cleaned.
    2. Read PATIENTS, VISITS, and BILLING from the Oracle schema.
    3. Engineer visit frequency, average length of stay per patient,
       provider rejection rate, days since registration, and time-based features.
    4. Write the feature dataset to data/processed/features.csv.

Oracle connection defaults are defined as header variables:
    USERNAME
    PASSWORD
    HOSTNAME
    PORT
    SERVICE_NAME
    SCHEMA

These can still be overridden with command-line arguments.
"""

# Summary:
# 1. Purpose: Builds patient-level engineered features from cleaned Oracle source tables.
# 2. What it does: Runs EDA, reads tables, creates aggregates, and writes features.csv.
# 3. Invoked by: Data preparation operators or the project feature-building workflow.
# 4. Main functions/classes: Oracle helpers, feature builders, parse_args, and main.
# 5. Validations/controls: Checks notebook, connection settings, source dates, and missing aggregates.

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import oracledb
import pandas as pd
from sqlalchemy import create_engine


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EDA_NOTEBOOK = PROJECT_ROOT / "notebooks" / "01_eda.ipynb"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "features.csv"

# Replace Username, Password, Hostname, port and Service_name with the actual DB credentials
USERNAME = USERNAME
PASSWORD = PASSWORD
HOSTNAME = HOSTNAME
PORT = PORT
SERVICE_NAME = SERVICE_NAME
SCHEMA = USERNAME


def execute_eda_notebook(notebook_path: Path) -> None:
    """Execute the EDA notebook that applies data-quality corrections."""
    if not notebook_path.exists():
        raise FileNotFoundError(f"EDA notebook not found: {notebook_path}")

    command = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        str(notebook_path),
    ]

    print(f"Executing EDA notebook first: {notebook_path}")
    subprocess.run(command, check=True)
    print("EDA notebook execution completed.")


def build_engine(user: str, password: str, hostname: str, port: int, service_name: str):
    """Create a SQLAlchemy Oracle engine using python-oracledb."""
    connection_url = (
        f"oracle+oracledb://{user}:{password}@{hostname}:{port}/"
        f"?service_name={service_name}"
    )
    return create_engine(connection_url)


def create_oracle_connection(user: str, password: str, hostname: str, port: int, service_name: str):
    """Create and validate a direct Oracle DB connection."""
    dsn = f"{hostname}:{port}/{service_name}"
    conn = oracledb.connect(user=user, password=password, dsn=dsn)
    print("Connected successfully")
    return conn


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Oracle column names to lowercase for feature logic."""
    df = df.copy()
    df.columns = df.columns.str.lower()
    return df


def read_cleaned_tables(engine, schema: str | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read cleaned source tables from Oracle into pandas DataFrames."""
    prefix = f"{schema}." if schema else ""

    patients = pd.read_sql(f"SELECT * FROM {prefix}patients", con=engine)
    visits = pd.read_sql(f"SELECT * FROM {prefix}visits", con=engine)
    billing = pd.read_sql(f"SELECT * FROM {prefix}billing", con=engine)

    patients = normalize_columns(patients)
    visits = normalize_columns(visits)
    billing = normalize_columns(billing)

    patients["registration_date"] = pd.to_datetime(patients["registration_date"])
    visits["visit_date"] = pd.to_datetime(visits["visit_date"])
    billing["billing_date"] = pd.to_datetime(billing["billing_date"])

    return patients, visits, billing


def resolve_reference_date(
    patients: pd.DataFrame, visits: pd.DataFrame, billing: pd.DataFrame, reference_date: str | None
) -> pd.Timestamp:
    """Use the provided reference date or one day after the latest available date."""
    if reference_date:
        return pd.Timestamp(reference_date)

    max_date = max(
        patients["registration_date"].max(),
        visits["visit_date"].max(),
        billing["billing_date"].max(),
    )
    return pd.Timestamp(max_date) + pd.Timedelta(days=1)


def add_patient_base_features(patients: pd.DataFrame, reference_date: pd.Timestamp) -> pd.DataFrame:
    features = patients.copy()

    features["days_since_registration"] = (
        reference_date - features["registration_date"]
    ).dt.days
    features["registration_year"] = features["registration_date"].dt.year
    features["registration_month"] = features["registration_date"].dt.month
    features["registration_quarter"] = features["registration_date"].dt.quarter
    features["registration_day_of_week"] = features["registration_date"].dt.dayofweek
    features["registered_on_weekend"] = (
        features["registration_day_of_week"].isin([5, 6]).astype(int)
    )

    return features


def build_visit_features(visits: pd.DataFrame, reference_date: pd.Timestamp) -> pd.DataFrame:
    visit_base = visits.copy()

    visit_base["visit_year"] = visit_base["visit_date"].dt.year
    visit_base["visit_month"] = visit_base["visit_date"].dt.month
    visit_base["visit_quarter"] = visit_base["visit_date"].dt.quarter
    visit_base["visit_day_of_week"] = visit_base["visit_date"].dt.dayofweek
    visit_base["visit_on_weekend"] = visit_base["visit_day_of_week"].isin([5, 6]).astype(int)

    patient_visit_features = (
        visit_base.groupby("patient_id", as_index=False)
        .agg(
            visit_frequency=("visit_id", "count"),
            first_visit_date=("visit_date", "min"),
            last_visit_date=("visit_date", "max"),
            avg_length_of_stay_hours=("length_of_stay_hours", "mean"),
            weekend_visit_count=("visit_on_weekend", "sum"),
        )
    )

    patient_visit_features["days_since_last_visit"] = (
        reference_date - patient_visit_features["last_visit_date"]
    ).dt.days
    patient_visit_features["days_between_first_last_visit"] = (
        patient_visit_features["last_visit_date"] - patient_visit_features["first_visit_date"]
    ).dt.days
    patient_visit_features["weekend_visit_rate"] = (
        patient_visit_features["weekend_visit_count"]
        / patient_visit_features["visit_frequency"]
    )

    return patient_visit_features


def build_provider_features(
    patients: pd.DataFrame, visits: pd.DataFrame, billing: pd.DataFrame
) -> pd.DataFrame:
    claims = (
        billing.merge(visits[["visit_id", "patient_id"]], on="visit_id", how="left")
        .merge(
            patients[["patient_id", "insurance_provider"]],
            on="patient_id",
            how="left",
        )
    )

    claims["is_rejected_claim"] = (
        claims["claim_status"].str.upper() == "REJECTED"
    ).astype(int)
    provider_features = (
        claims.groupby("insurance_provider", as_index=False)
        .agg(
            provider_total_claims=("bill_id", "count"),
            provider_rejected_claims=("is_rejected_claim", "sum"),
        )
    )

    provider_features["provider_rejection_rate"] = (
        provider_features["provider_rejected_claims"]
        / provider_features["provider_total_claims"]
    )
    return provider_features


def build_features_from_tables(
    patients: pd.DataFrame,
    visits: pd.DataFrame,
    billing: pd.DataFrame,
    reference_date: str | None,
) -> pd.DataFrame:
    reference_timestamp = resolve_reference_date(patients, visits, billing, reference_date)

    features = add_patient_base_features(patients, reference_timestamp)
    visit_features = build_visit_features(visits, reference_timestamp)
    provider_features = build_provider_features(patients, visits, billing)

    features = features.merge(visit_features, on="patient_id", how="left")
    features = features.merge(provider_features, on="insurance_provider", how="left")

    count_columns = [
        "visit_frequency",
        "weekend_visit_count",
        "provider_total_claims",
        "provider_rejected_claims",
    ]
    for column in count_columns:
        if column in features.columns:
            features[column] = features[column].fillna(0).astype(int)

    rate_columns = [
        "weekend_visit_rate",
        "provider_rejection_rate",
    ]
    for column in rate_columns:
        if column in features.columns:
            features[column] = features[column].fillna(0)

    return features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build engineered features from cleaned Oracle PATIENTS, VISITS, and BILLING tables."
    )
    parser.add_argument("--user", default=USERNAME, help="Oracle username.")
    parser.add_argument(
        "--password", default=PASSWORD, help="Oracle password."
    )
    parser.add_argument(
        "--hostname",
        default=HOSTNAME,
        help="Oracle database hostname.",
    )
    parser.add_argument(
        "--port",
        default=PORT,
        type=int,
        help="Oracle listener port.",
    )
    parser.add_argument(
        "--service-name",
        default=SERVICE_NAME,
        help="Oracle service name.",
    )
    parser.add_argument(
        "--schema",
        default=SCHEMA,
        help="Optional Oracle schema name. Example: DSJFCPM26USR1",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output feature CSV path.")
    parser.add_argument(
        "--reference-date",
        default=None,
        help="Optional date used for recency features, in YYYY-MM-DD format. Defaults to one day after the latest source date.",
    )
    parser.add_argument(
        "--eda-notebook",
        default=DEFAULT_EDA_NOTEBOOK,
        help="EDA notebook to execute before feature extraction.",
    )
    parser.add_argument(
        "--skip-eda",
        action="store_true",
        help="Skip executing the EDA notebook if the Oracle tables are already cleaned.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    missing = [
        name
        for name, value in {
            "user": args.user,
            "password": args.password,
            "hostname": args.hostname,
            "port": args.port,
            "service_name": args.service_name,
        }.items()
        if not value
    ]
    if missing:
        raise ValueError(
            "Missing Oracle connection values: "
            + ", ".join(missing)
            + ". Pass them as arguments or environment variables."
        )

    if not args.skip_eda:
        execute_eda_notebook(Path(args.eda_notebook))

    conn = create_oracle_connection(
        user=args.user,
        password=args.password,
        hostname=args.hostname,
        port=args.port,
        service_name=args.service_name,
    )
    conn.close()

    engine = build_engine(
        user=args.user,
        password=args.password,
        hostname=args.hostname,
        port=args.port,
        service_name=args.service_name,
    )
    patients, visits, billing = read_cleaned_tables(engine, schema=args.schema)
    features = build_features_from_tables(
        patients=patients,
        visits=visits,
        billing=billing,
        reference_date=args.reference_date,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)

    print(f"Feature dataset created: {output_path}")
    print(f"Rows: {features.shape[0]}")
    print(f"Columns: {features.shape[1]}")
    print("Preview:")
    print(features.head(10))


if __name__ == "__main__":
    main()
