"""Borusyak, Jaravel & Spiess (2021) Imputation Estimator.

This module implements the imputation-based estimator for staggered
adoption designs, providing robust inference for treatment effects.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

# Constants
SIGNIFICANCE_LEVEL = 0.05
MIN_PRE_PERIODS = 2
BOOTSTRAP_ITERATIONS = 100


class BorusyakImputation:
    """Imputation-based estimator for staggered adoption.

    Implements Borusyak, Jaravel & Spiess (2021) approach
    for robust DiD estimation with staggered treatment.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        unit_col: str = "unit",
        time_col: str = "time",
        treatment_col: str | None = None,
    ):
        """Initialize Borusyak imputation estimator.

        Args:
            data: Panel data DataFrame
            unit_col: Column name for unit identifier
            time_col: Column name for time period
            treatment_col: Optional column indicating treatment status
        """
        self.data = data.copy()
        self.unit_col = unit_col
        self.time_col = time_col
        self.treatment_col = treatment_col

        # Validate data
        self._validate_data()

        # Store units and time periods
        self.units = sorted(data[unit_col].unique())
        self.time_periods = sorted(data[time_col].unique())

        # Results storage
        self.imputed_data = None
        self.treatment_effects = None
        self.pre_trend_test = None

    def _validate_data(self) -> None:
        """Validate input data structure."""
        required_cols = [self.unit_col, self.time_col]
        if self.treatment_col:
            required_cols.append(self.treatment_col)

        missing_cols = [col for col in required_cols if col not in self.data.columns]

        if missing_cols:
            msg = f"Missing required columns: {missing_cols}"
            raise ValueError(msg)

    def _identify_treatment_timing(self) -> pd.Series:
        """Identify treatment timing for each unit.

        Returns:
            Series with treatment timing for each unit
        """
        if self.treatment_col and self.treatment_col in self.data.columns:
            # Find first treatment time for each unit
            treated_data = self.data[self.data[self.treatment_col] == 1]
            if len(treated_data) > 0:
                return treated_data.groupby(self.unit_col)[
                    self.time_col
                ].min()

        # Return empty series if no treatment timing found
        return pd.Series(dtype="int64")

    def impute_counterfactuals(
        self, outcome: str, covariates: list[str] | None = None, method: str = "fe"
    ) -> pd.DataFrame:
        """Impute Y(0) counterfactuals for treated units.

        Args:
            outcome: Outcome variable column name
            covariates: Optional list of control variables
            method: Imputation method ('fe' for fixed effects, 'linear' for model)

        Returns:
            DataFrame with imputed counterfactuals
        """
        if outcome not in self.data.columns:
            msg = f"Outcome column '{outcome}' not found"
            raise ValueError(msg)

        # Identify treatment timing
        treatment_timing = self._identify_treatment_timing()

        # Create a copy for imputation
        imputed_df = self.data.copy()
        imputed_df["Y0_imputed"] = imputed_df[outcome].copy()
        imputed_df["is_treated"] = False

        # Mark treated observations if treatment timing exists
        if len(treatment_timing) > 0:
            for unit, treat_time in treatment_timing.items():
                unit_mask = imputed_df[self.unit_col] == unit
                post_mask = imputed_df[self.time_col] >= treat_time
                imputed_df.loc[unit_mask & post_mask, "is_treated"] = True
        else:
            warnings.warn(
                "No treated units identified for imputation",
                UserWarning,
                stacklevel=2
            )

        if method == "fe":
            # Fixed effects imputation
            imputed_df = self._impute_with_fixed_effects(
                imputed_df, outcome, covariates
            )
        elif method == "linear":
            # Linear model imputation
            imputed_df = self._impute_with_linear_model(imputed_df, outcome, covariates)
        else:
            msg = f"Unknown imputation method: {method}"
            raise ValueError(msg)

        self.imputed_data = imputed_df
        return imputed_df

    def _impute_with_fixed_effects(
        self, df: pd.DataFrame, outcome: str, covariates: list[str] | None = None
    ) -> pd.DataFrame:
        """Impute using two-way fixed effects on control units.

        Args:
            df: DataFrame with treatment indicators
            outcome: Outcome variable column name
            covariates: Optional control variables

        Returns:
            DataFrame with imputed values
        """
        # Get control observations
        control_data = df[~df["is_treated"]].copy()

        # Prepare features for fixed effects model
        unit_dummies = pd.get_dummies(
            control_data[self.unit_col], prefix="unit", drop_first=True
        )
        time_dummies = pd.get_dummies(
            control_data[self.time_col], prefix="time", drop_first=True
        )

        x = pd.concat([unit_dummies, time_dummies], axis=1)

        # Add covariates if specified
        if covariates:
            x = pd.concat([x, control_data[covariates]], axis=1)

        y = control_data[outcome]

        # Fit model on control units
        model = OLS(y, add_constant(x), missing="drop")
        result = model.fit()

        # Predict counterfactuals for treated units
        treated_data = df[df["is_treated"]].copy()

        if len(treated_data) > 0:
            # Prepare features for treated units
            treated_unit_dummies = pd.get_dummies(
                treated_data[self.unit_col], prefix="unit", drop_first=True
            )
            treated_time_dummies = pd.get_dummies(
                treated_data[self.time_col], prefix="time", drop_first=True
            )

            x_treated = pd.concat([treated_unit_dummies, treated_time_dummies], axis=1)

            # Ensure columns match
            for col in x.columns:
                if col not in x_treated.columns:
                    x_treated[col] = 0
            x_treated = x_treated[x.columns]

            if covariates:
                x_treated = pd.concat([x_treated, treated_data[covariates]], axis=1)

            # Predict Y(0) for treated units
            y0_pred = result.predict(add_constant(x_treated))

            # Update imputed values
            df.loc[df["is_treated"], "Y0_imputed"] = y0_pred.values

        # Calculate treatment effects
        df["treatment_effect"] = np.where(
            df["is_treated"], df[outcome] - df["Y0_imputed"], 0
        )

        return df

    def _impute_with_linear_model(
        self, df: pd.DataFrame, outcome: str, covariates: list[str] | None = None
    ) -> pd.DataFrame:
        """Impute using linear regression on control units.

        Args:
            df: DataFrame with treatment indicators
            outcome: Outcome variable column name
            covariates: Optional control variables

        Returns:
            DataFrame with imputed values
        """
        # Get control observations
        control_data = df[~df["is_treated"]].copy()

        # Prepare features
        features = [self.unit_col, self.time_col]
        if covariates:
            features.extend(covariates)

        # Encode categorical variables
        x_control = pd.get_dummies(control_data[features])
        y_control = control_data[outcome]

        # Fit linear model
        model = LinearRegression()
        model.fit(x_control, y_control)

        # Predict for treated units
        treated_data = df[df["is_treated"]].copy()

        if len(treated_data) > 0:
            x_treated = pd.get_dummies(treated_data[features])

            # Ensure columns match
            for col in x_control.columns:
                if col not in x_treated.columns:
                    x_treated[col] = 0
            x_treated = x_treated[x_control.columns]

            # Predict Y(0)
            y0_pred = model.predict(x_treated)

            # Update imputed values
            df.loc[df["is_treated"], "Y0_imputed"] = y0_pred

        # Calculate treatment effects
        df["treatment_effect"] = np.where(
            df["is_treated"], df[outcome] - df["Y0_imputed"], 0
        )

        return df

    def estimate_effects(
        self,
        level: str = "average",
        bootstrap: bool = True,
        n_bootstrap: int = BOOTSTRAP_ITERATIONS,
    ) -> dict:
        """Estimate treatment effects from imputed counterfactuals.

        Args:
            level: Aggregation level ('average', 'unit', 'time', 'cohort')
            bootstrap: Whether to use bootstrap for inference
            n_bootstrap: Number of bootstrap iterations

        Returns:
            Dictionary with treatment effect estimates
        """
        if self.imputed_data is None:
            msg = "Must run impute_counterfactuals first"
            raise ValueError(msg)

        treated_data = self.imputed_data[self.imputed_data["is_treated"]]

        if len(treated_data) == 0:
            warnings.warn(
                "No treated observations found", UserWarning, stacklevel=2
            )
            return {"att": np.nan, "se": np.nan}

        if level == "average":
            # Overall ATT
            att = treated_data["treatment_effect"].mean()

            if bootstrap:
                # Bootstrap standard errors
                bootstrap_atts = []
                for _ in range(n_bootstrap):
                    sample = treated_data.sample(
                        n=len(treated_data), replace=True
                    )
                    bootstrap_atts.append(sample["treatment_effect"].mean())

                se = np.std(bootstrap_atts)
            else:
                # Analytical standard error
                se = (
                    treated_data["treatment_effect"].std()
                    / np.sqrt(len(treated_data))
                )

            return {
                "att": att,
                "se": se,
                "t_stat": att / se if se > 0 else np.nan,
                "p_value": 2 * (1 - np.abs(np.minimum(1, np.abs(att / se))))
                if se > 0
                else np.nan,
                "n_treated": len(treated_data),
            }

        if level == "unit":
            # Unit-specific effects
            unit_effects = treated_data.groupby(self.unit_col)[
                "treatment_effect"
            ].agg(["mean", "std", "count"])

            return unit_effects.to_dict("index")

        if level == "time":
            # Time-specific effects
            time_effects = treated_data.groupby(self.time_col)[
                "treatment_effect"
            ].agg(["mean", "std", "count"])

            return time_effects.to_dict("index")

        if level == "cohort":
            # Cohort-specific effects (by treatment timing)
            treatment_timing = self._identify_treatment_timing()

            # Add cohort information
            treated_data = treated_data.copy()
            treated_data["cohort"] = treated_data[self.unit_col].map(
                treatment_timing
            )

            cohort_effects = treated_data.groupby("cohort")[
                "treatment_effect"
            ].agg(["mean", "std", "count"])

            return cohort_effects.to_dict("index")

        msg = f"Unknown aggregation level: {level}"
        raise ValueError(msg)

    def test_pre_trends(self, outcome: str, window: int = MIN_PRE_PERIODS) -> dict:
        """Test for pre-treatment trends.

        Args:
            outcome: Outcome variable column name
            window: Number of pre-treatment periods to test

        Returns:
            Dictionary with pre-trend test results
        """
        treatment_timing = self._identify_treatment_timing()

        # Collect pre-treatment observations
        pre_treatment_data = []

        for unit, treat_time in treatment_timing.items():
            unit_data = self.data[self.data[self.unit_col] == unit].copy()
            pre_data = unit_data[
                unit_data[self.time_col] < treat_time
            ].tail(window)

            if len(pre_data) >= MIN_PRE_PERIODS:
                pre_data["relative_time"] = (
                    pre_data[self.time_col] - treat_time
                )
                pre_treatment_data.append(pre_data)

        if not pre_treatment_data:
            return {
                "test_statistic": np.nan,
                "p_value": np.nan,
                "reject_parallel": False,
                "message": "Insufficient pre-treatment data",
            }

        pre_df = pd.concat(pre_treatment_data)

        # Test for differential trends
        # Regression of outcome on relative time
        x = add_constant(pre_df["relative_time"])
        y = pre_df[outcome]

        model = OLS(y, x, missing="drop")
        result = model.fit()

        # Test if trend coefficient is zero
        trend_coef = result.params["relative_time"]
        trend_se = result.bse["relative_time"]
        t_stat = trend_coef / trend_se if trend_se > 0 else np.nan
        p_value = result.pvalues["relative_time"]

        self.pre_trend_test = {
            "test_statistic": t_stat,
            "p_value": p_value,
            "reject_parallel": p_value < SIGNIFICANCE_LEVEL,
            "trend_coefficient": trend_coef,
            "trend_se": trend_se,
        }

        return self.pre_trend_test

    def plot_diagnostics(self, outcome: str, save_path: str | None = None) -> None:
        """Generate diagnostic plots.

        Args:
            outcome: Outcome variable column name
            save_path: Optional path to save figure
        """
        if self.imputed_data is None:
            msg = "Must run impute_counterfactuals first"
            raise ValueError(msg)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot 1: Actual vs Imputed for treated units
        treated = self.imputed_data[self.imputed_data["is_treated"]]
        axes[0, 0].scatter(treated["Y0_imputed"], treated[outcome], alpha=0.5)
        axes[0, 0].plot(
            [treated["Y0_imputed"].min(), treated["Y0_imputed"].max()],
            [treated["Y0_imputed"].min(), treated["Y0_imputed"].max()],
            "r--",
        )
        axes[0, 0].set_xlabel("Imputed Y(0)")
        axes[0, 0].set_ylabel("Actual Y(1)")
        axes[0, 0].set_title("Actual vs Imputed Outcomes")

        # Plot 2: Distribution of treatment effects
        axes[0, 1].hist(treated["treatment_effect"], bins=30, edgecolor="black")
        axes[0, 1].axvline(
            treated["treatment_effect"].mean(),
            color="red",
            linestyle="--",
            label=f"ATT = {treated['treatment_effect'].mean():.3f}",
        )
        axes[0, 1].set_xlabel("Treatment Effect")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_title("Distribution of Treatment Effects")
        axes[0, 1].legend()

        # Plot 3: Treatment effects over time
        time_effects = treated.groupby(self.time_col)["treatment_effect"].mean()
        axes[1, 0].plot(time_effects.index, time_effects.values, marker="o")
        axes[1, 0].set_xlabel("Time")
        axes[1, 0].set_ylabel("Average Treatment Effect")
        axes[1, 0].set_title("Treatment Effects Over Time")
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Pre-treatment fit quality
        control = self.imputed_data[~self.imputed_data["is_treated"]]
        residuals = control[outcome] - control["Y0_imputed"]
        axes[1, 1].scatter(control[self.time_col], residuals, alpha=0.3)
        axes[1, 1].axhline(0, color="red", linestyle="--")
        axes[1, 1].set_xlabel("Time")
        axes[1, 1].set_ylabel("Residuals (Control Units)")
        axes[1, 1].set_title("Model Fit Quality (Control Units)")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()
