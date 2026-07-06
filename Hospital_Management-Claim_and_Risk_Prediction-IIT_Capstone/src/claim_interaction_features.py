"""Training-only aggregate interactions for the claim classification model."""

# Summary:
# 1. Purpose: Adds aggregate and interaction features for claim-model training pipelines.
# 2. What it does: Learns department/provider averages and creates ratios and chronic interactions.
# 3. Invoked by: Claim interaction-feature notebooks and artifacts that include this transformer.
# 4. Main functions/classes: ClaimInteractionFeatureTransformer and its fit/transform helpers.
# 5. Validations/controls: Requires a DataFrame, checks source columns, and avoids divide-by-zero errors.

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class ClaimInteractionFeatureTransformer(BaseEstimator, TransformerMixin):
    """Add leakage-safe billing, stay, lag, provider, and chronic interactions."""

    def __init__(
        self,
        department_column="department",
        provider_column="insurance_provider",
        billed_amount_column="billed_amount",
        length_of_stay_column="length_of_stay_hours",
        billing_lag_column="days_from_visit_to_billing",
        age_column="age",
        chronic_column="chronic_flag",
    ):
        self.department_column = department_column
        self.provider_column = provider_column
        self.billed_amount_column = billed_amount_column
        self.length_of_stay_column = length_of_stay_column
        self.billing_lag_column = billing_lag_column
        self.age_column = age_column
        self.chronic_column = chronic_column

    def fit(self, X, y=None):
        frame = self._as_frame(X)
        self._validate_columns(frame)
        numeric = self._numeric_frame(frame)
        aggregate_frame = pd.concat(
            [
                frame[[self.department_column, self.provider_column]],
                numeric[[
                    self.billed_amount_column,
                    self.length_of_stay_column,
                    self.billing_lag_column,
                ]],
            ],
            axis=1,
        )

        department_group = aggregate_frame.groupby(
            self.department_column, dropna=True
        )
        provider_group = aggregate_frame.groupby(self.provider_column, dropna=True)

        self.department_billed_amount_means_ = department_group[
            self.billed_amount_column
        ].mean()
        self.department_length_of_stay_means_ = department_group[
            self.length_of_stay_column
        ].mean()
        self.department_billing_lag_means_ = department_group[
            self.billing_lag_column
        ].mean()
        self.provider_billed_amount_means_ = provider_group[
            self.billed_amount_column
        ].mean()
        self.provider_billing_lag_means_ = provider_group[
            self.billing_lag_column
        ].mean()

        self.global_billed_amount_mean_ = numeric[self.billed_amount_column].mean()
        self.global_length_of_stay_mean_ = numeric[
            self.length_of_stay_column
        ].mean()
        self.global_billing_lag_mean_ = numeric[self.billing_lag_column].mean()
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        return self

    def transform(self, X):
        check_is_fitted(
            self,
            [
                "department_billed_amount_means_",
                "department_length_of_stay_means_",
                "department_billing_lag_means_",
                "provider_billed_amount_means_",
                "provider_billing_lag_means_",
            ],
        )
        frame = self._as_frame(X).copy()
        self._validate_columns(frame)
        numeric = self._numeric_frame(frame)

        billed_amount = numeric[self.billed_amount_column]
        length_of_stay = numeric[self.length_of_stay_column]
        billing_lag = numeric[self.billing_lag_column]
        age = numeric[self.age_column]
        chronic = numeric[self.chronic_column]

        department_bill_average = frame[self.department_column].map(
            self.department_billed_amount_means_
        ).fillna(self.global_billed_amount_mean_)
        department_stay_average = frame[self.department_column].map(
            self.department_length_of_stay_means_
        ).fillna(self.global_length_of_stay_mean_)
        department_lag_average = frame[self.department_column].map(
            self.department_billing_lag_means_
        ).fillna(self.global_billing_lag_mean_)
        provider_bill_average = frame[self.provider_column].map(
            self.provider_billed_amount_means_
        ).fillna(self.global_billed_amount_mean_)
        provider_lag_average = frame[self.provider_column].map(
            self.provider_billing_lag_means_
        ).fillna(self.global_billing_lag_mean_)

        frame["department_avg_billed_amount"] = department_bill_average
        frame["bill_vs_department_avg"] = billed_amount - department_bill_average
        frame["bill_to_department_avg_ratio"] = self._safe_ratio(
            billed_amount, department_bill_average
        )
        frame["provider_avg_billed_amount"] = provider_bill_average
        frame["bill_vs_provider_avg"] = billed_amount - provider_bill_average
        frame["bill_to_provider_avg_ratio"] = self._safe_ratio(
            billed_amount, provider_bill_average
        )
        frame["department_avg_length_of_stay"] = department_stay_average
        frame["los_to_department_avg_ratio"] = self._safe_ratio(
            length_of_stay, department_stay_average
        )
        frame["billed_amount_per_stay_hour"] = self._safe_ratio(
            billed_amount, length_of_stay.clip(lower=1)
        )
        frame["department_avg_billing_lag"] = department_lag_average
        frame["billing_lag_vs_department_avg"] = billing_lag - department_lag_average
        frame["billing_lag_to_department_avg_ratio"] = self._safe_ratio(
            billing_lag, department_lag_average
        )
        frame["provider_avg_billing_lag"] = provider_lag_average
        frame["billing_lag_vs_provider_avg"] = billing_lag - provider_lag_average
        frame["billing_lag_to_provider_avg_ratio"] = self._safe_ratio(
            billing_lag, provider_lag_average
        )
        frame["age_chronic_interaction"] = age * chronic
        frame["los_chronic_interaction"] = length_of_stay * chronic
        frame["billed_amount_chronic_interaction"] = billed_amount * chronic
        return frame

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")
        return np.concatenate(
            [self.feature_names_in_, np.asarray(self.interaction_feature_names())]
        )

    @staticmethod
    def interaction_feature_names():
        return [
            "department_avg_billed_amount",
            "bill_vs_department_avg",
            "bill_to_department_avg_ratio",
            "provider_avg_billed_amount",
            "bill_vs_provider_avg",
            "bill_to_provider_avg_ratio",
            "department_avg_length_of_stay",
            "los_to_department_avg_ratio",
            "billed_amount_per_stay_hour",
            "department_avg_billing_lag",
            "billing_lag_vs_department_avg",
            "billing_lag_to_department_avg_ratio",
            "provider_avg_billing_lag",
            "billing_lag_vs_provider_avg",
            "billing_lag_to_provider_avg_ratio",
            "age_chronic_interaction",
            "los_chronic_interaction",
            "billed_amount_chronic_interaction",
        ]

    def _numeric_frame(self, frame):
        columns = [
            self.billed_amount_column,
            self.length_of_stay_column,
            self.billing_lag_column,
            self.age_column,
            self.chronic_column,
        ]
        return frame[columns].apply(pd.to_numeric, errors="coerce")

    @staticmethod
    def _as_frame(X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "ClaimInteractionFeatureTransformer requires a pandas DataFrame."
            )
        return X

    def _validate_columns(self, frame):
        required = {
            self.department_column,
            self.provider_column,
            self.billed_amount_column,
            self.length_of_stay_column,
            self.billing_lag_column,
            self.age_column,
            self.chronic_column,
        }
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"Missing interaction source columns: {missing}")

    @staticmethod
    def _safe_ratio(numerator, denominator):
        return numerator / denominator.replace(0, np.nan)
