# Module -21 - Capstone Project - Healthcare Business   
# ----------------------------------------------------
# I have used Oracle 26 AI DB for creating the tables, loading the csv data, performing EDA and capturing the drift statistics.


Hospital operations, data-quality, feature-engineering, and machine-learning project using Oracle `PATIENTS`, `VISITS`, and `BILLING` tables.

## Open in VS Code

Open `Hospital_Analytics_Capstone.code-workspace` in Visual Studio Code, or run:

```bash
code Hospital_Analytics_Capstone.code-workspace
```

Install the recommended Python and Jupyter extensions when VS Code prompts.

## Project Structure

```text
notebooks/       EDA, SQL, risk-model, and claim-model notebooks
src/             Reusable Python feature/model-table builders
sql/             Standalone SQL analysis queries
data/raw/        Original CSV source files
data/processed/  Engineered features and modeling tables
models/risk/     Visit-risk model artifacts
models/claim/    Claim-outcome model artifacts
schemas/         Feature schema JSON files
metrics/         Saved model evaluation metrics
reports/         Project statements and generated DOCX reports
.vscode/         Shared VS Code settings and extension recommendations
```

## Environment Setup

From the project root:

```bash
python3 -m pip install -r requirements.txt
```

Select the same Python interpreter in VS Code using `Python: Select Interpreter`.

## Main Workflow

1. Run `notebooks/01_eda.ipynb` to validate and clean the Oracle tables.
2. Build patient-level features:

```bash
python3 src/build_features.py --skip-eda
```

Remove `--skip-eda` when the script should execute `notebooks/01_eda.ipynb` first.

3. Build the processed modeling table:

```bash
python3 src/build_model_table.py
```

4. Run model notebooks from the VS Code workspace root:

```text
notebooks/02_risk_model_v1.ipynb
notebooks/02_risk_model_v2.ipynb
notebooks/03_claim_model_v1.ipynb
notebooks/03_claim_model_v2.ipynb
```

Notebook artifacts are written to `models/`, `schemas/`, and `metrics/`.

## Oracle Connection

The current notebooks and `src/build_features.py` contain project-specific Oracle connection values. Before sharing or publishing the project, move credentials to environment variables or a local `.env` file that is excluded from version control.
