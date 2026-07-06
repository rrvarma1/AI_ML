"""Training-only aggregate interaction features for the visit-risk model.

The transformer learns department and provider averages during ``fit`` only.
It then applies those frozen mappings during validation, testing, and inference.
"""

# Summary:
# 1. Purpose: Adds aggregate interaction features and resampling targets for Risk V5.
# 2. What it does: Learns frozen averages, creates ratios, and supplies SMOTE/undersampling targets.
# 3. Invoked by: Risk model notebooks and app/model_registry.py when Risk V5 is loaded.
# 4. Main functions/classes: RiskInteractionFeatureTransformer and two resampling helper functions.
# 5. Validations/controls: Checks fitted state, DataFrame type, required columns, and zero denominators.

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


def smote_to_75_percent_of_majority(y):
    """Return per-class targets that raise minorities to 75% of the majority."""
    counts = pd.Series(y).value_counts()
    target = int(np.floor(counts.max() * 0.75))
    return {
        label: target
        for label, count in counts.items()
        if count < target
    }


def undersample_to_minority_count(y):
    """Return per-class targets that reduce larger classes to the minority count."""
    counts = pd.Series(y).value_counts()
    target = int(counts.min())
    return {
        label: target
        for label, count in counts.items()
        if count > target
    }


class RiskInteractionFeatureTransformer(BaseEstimator, TransformerMixin):
    """Add leakage-safe department, provider, billing, and stay interactions."""

    def __init__(
        self,
        department_column="department",
        provider_column="insurance_provider",
        billed_amount_column="billed_amount",
        length_of_stay_column="length_of_stay_hours",
    ):
        self.department_column = department_column
        self.provider_column = provider_column
        self.billed_amount_column = billed_amount_column
        self.length_of_stay_column = length_of_stay_column

    def fit(self, X, y=None):
        frame = self._as_frame(X)
        self._validate_columns(frame)

        billed_amount = pd.to_numeric(
            frame[self.billed_amount_column], errors="coerce"
        )
        length_of_stay = pd.to_numeric(
            frame[self.length_of_stay_column], errors="coerce"
        )
        aggregate_frame = pd.DataFrame(
            {
                self.department_column: frame[self.department_column],
                self.provider_column: frame[self.provider_column],
                self.billed_amount_column: billed_amount,
                self.length_of_stay_column: length_of_stay,
            },
            index=frame.index,
        )

        self.department_billed_amount_means_ = aggregate_frame.groupby(
            self.department_column, dropna=True
        )[self.billed_amount_column].mean()
        self.department_length_of_stay_means_ = aggregate_frame.groupby(
            self.department_column, dropna=True
        )[self.length_of_stay_column].mean()
        self.provider_billed_amount_means_ = aggregate_frame.groupby(
            self.provider_column, dropna=True
        )[self.billed_amount_column].mean()

        self.global_billed_amount_mean_ = billed_amount.mean()
        self.global_length_of_stay_mean_ = length_of_stay.mean()
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        return self

    def transform(self, X):
        check_is_fitted(
            self,
            [
                "department_billed_amount_means_",
                "department_length_of_stay_means_",
                "provider_billed_amount_means_",
            ],
        )
        frame = self._as_frame(X).copy()
        self._validate_columns(frame)

        billed_amount = pd.to_numeric(
            frame[self.billed_amount_column], errors="coerce"
        )
        length_of_stay = pd.to_numeric(
            frame[self.length_of_stay_column], errors="coerce"
        )
        department_bill_average = frame[self.department_column].map(
            self.department_billed_amount_means_
        ).fillna(self.global_billed_amount_mean_)
        department_stay_average = frame[self.department_column].map(
            self.department_length_of_stay_means_
        ).fillna(self.global_length_of_stay_mean_)
        provider_bill_average = frame[self.provider_column].map(
            self.provider_billed_amount_means_
        ).fillna(self.global_billed_amount_mean_)

        frame["department_avg_billed_amount"] = department_bill_average
        frame["bill_vs_department_avg"] = billed_amount - department_bill_average
        frame["bill_to_department_avg_ratio"] = self._safe_ratio(
            billed_amount, department_bill_average
        )
        frame["department_avg_length_of_stay"] = department_stay_average
        frame["los_to_department_avg_ratio"] = self._safe_ratio(
            length_of_stay, department_stay_average
        )
        frame["billed_amount_per_stay_hour"] = self._safe_ratio(
            billed_amount, length_of_stay.clip(lower=1)
        )
        frame["provider_avg_billed_amount"] = provider_bill_average
        frame["bill_vs_provider_avg"] = billed_amount - provider_bill_average
        frame["bill_to_provider_avg_ratio"] = self._safe_ratio(
            billed_amount, provider_bill_average
        )
        return frame

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")
        interaction_features = np.asarray(
            [
                "department_avg_billed_amount",
                "bill_vs_department_avg",
                "bill_to_department_avg_ratio",
                "department_avg_length_of_stay",
                "los_to_department_avg_ratio",
                "billed_amount_per_stay_hour",
                "provider_avg_billed_amount",
                "bill_vs_provider_avg",
                "bill_to_provider_avg_ratio",
            ],
            dtype=object,
        )
        return np.concatenate([self.feature_names_in_, interaction_features])

    @staticmethod
    def _as_frame(X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "RiskInteractionFeatureTransformer requires a pandas DataFrame."
            )
        return X

    def _validate_columns(self, frame):
        required = {
            self.department_column,
            self.provider_column,
            self.billed_amount_column,
            self.length_of_stay_column,
        }
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"Missing interaction source columns: {missing}")

    @staticmethod
    def _safe_ratio(numerator, denominator):
        denominator = denominator.replace(0, np.nan)
        return numerator / denominator
