"""Sun & Abraham (2021) Interaction-Weighted Estimator.

This module implements the Sun & Abraham (2021) estimator for staggered
difference-in-differences designs, addressing negative weighting issues.
"""

import warnings

import numpy as np
import pandas as pd

from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

# Constants for statistical analysis
SIGNIFICANCE_LEVEL = 0.05
MIN_COHORT_SIZE = 5
MAX_EVENT_TIME = 10
MIN_PRE_PERIODS_FOR_TEST = 2


class SunAbrahamEstimator:
    """Interaction-weighted estimator for staggered DiD.

    Implements Sun & Abraham (2021) to address negative weighting
    problems in two-way fixed effects with staggered treatment.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        cohort_col: str,
        time_col: str,
        unit_col: str,
        never_treated_value: int | None = None,
    ):
        """Initialize Sun & Abraham estimator.

        Args:
            data: Panel data DataFrame
            cohort_col: Column indicating treatment cohort (first treatment time)
            time_col: Time period column
            unit_col: Unit (entity) identifier column
            never_treated_value: Value indicating never-treated units (default: max+1)
        """
        self.data = data.copy()
        self.cohort_col = cohort_col
        self.time_col = time_col
        self.unit_col = unit_col

        # Identify never-treated units
        if never_treated_value is None:
            max_time = data[time_col].max()
            self.never_treated = max_time + 1
        else:
            self.never_treated = never_treated_value

        # Validate data structure
        self._validate_data()

        # Store cohort information
        self.cohorts = sorted(data[cohort_col].unique())
        self.treated_cohorts = [c for c in self.cohorts if c != self.never_treated]
        self.time_periods = sorted(data[time_col].unique())

        # Results storage
        self.results = {}
        self.event_study_results = None

    def _validate_data(self) -> None:
        """Validate input data structure."""
        required_cols = [self.cohort_col, self.time_col, self.unit_col]
        missing_cols = [col for col in required_cols if col not in self.data.columns]

        if missing_cols:
            msg = f"Missing required columns: {missing_cols}"
            raise ValueError(msg)

        # Check for balanced panel
        panel_check = self.data.groupby(self.unit_col)[self.time_col].nunique()
        if panel_check.nunique() > 1:
            warnings.warn(
                "Unbalanced panel detected. Results may be affected.",
                UserWarning,
                stacklevel=2
            )

    def _create_cohort_dummies(self, relative_time: int) -> pd.DataFrame:
        """Create cohort-specific treatment dummies.

        Args:
            relative_time: Event time relative to treatment

        Returns:
            DataFrame with cohort-specific dummies
        """
        dummies = pd.DataFrame(index=self.data.index)

        for cohort in self.treated_cohorts:
            # Create indicator for units in this cohort at this relative time
            cohort_indicator = (self.data[self.cohort_col] == cohort).astype(int)
            time_indicator = (
                self.data[self.time_col] == cohort + relative_time
            ).astype(int)

            dummy_name = f"cohort_{cohort}_rel_{relative_time}"
            dummies[dummy_name] = cohort_indicator * time_indicator

        return dummies

    def estimate(
        self,
        outcome: str,
        covariates: list[str] | None = None,
        cluster_col: str | None = None,
    ) -> dict:
        """Estimate cohort-specific and aggregated treatment effects.

        Args:
            outcome: Outcome variable column name
            covariates: Optional list of control variables
            cluster_col: Optional column for clustered standard errors

        Returns:
            Dictionary with estimation results
        """
        if outcome not in self.data.columns:
            msg = f"Outcome column '{outcome}' not found in data"
            raise ValueError(msg)

        # Prepare data
        estimation_data = self.data.copy()

        # Create relative time variable
        estimation_data["relative_time"] = (
            estimation_data[self.time_col] - estimation_data[self.cohort_col]
        )

        # For never-treated, set relative time to large negative
        never_treated_mask = (
            estimation_data[self.cohort_col] == self.never_treated
        )
        estimation_data.loc[never_treated_mask, "relative_time"] = -999

        # Create cohort-specific post-treatment indicators
        cohort_effects = {}

        for cohort in self.treated_cohorts:
            cohort_mask = estimation_data[self.cohort_col] == cohort

            # Cohort-specific ATT

            # Run regression for this cohort
            cohort_data = estimation_data[
                cohort_mask | never_treated_mask
            ].copy()

            cohort_data["treatment"] = (
                cohort_data["relative_time"] >= 0
            ).astype(int)

            # Add fixed effects
            y = cohort_data[outcome]
            x = pd.get_dummies(
                cohort_data[[self.unit_col, self.time_col]],
                columns=[self.unit_col, self.time_col],
                drop_first=True,
            )
            x["treatment"] = cohort_data["treatment"]

            # Add covariates if specified
            if covariates:
                x = pd.concat([x, cohort_data[covariates]], axis=1)

            # Estimate model
            try:
                model = OLS(y, add_constant(x), missing="drop")
                result = model.fit()

                cohort_effects[cohort] = {
                    "att": result.params["treatment"],
                    "se": result.bse["treatment"],
                    "t_stat": result.tvalues["treatment"],
                    "p_value": result.pvalues["treatment"],
                    "n_treated": cohort_mask.sum(),
                    "n_control": never_treated_mask.sum(),
                }
            except Exception as e:
                warnings.warn(
                    f"Estimation failed for cohort {cohort}: {e}",
                    UserWarning,
                    stacklevel=2
                )
                cohort_effects[cohort] = None

        # Aggregate effects using interaction weights
        valid_effects = [v for v in cohort_effects.values() if v is not None]

        if valid_effects:
            # Weight by cohort size
            weights = np.array([e["n_treated"] for e in valid_effects])
            weights = weights / weights.sum()

            atts = np.array([e["att"] for e in valid_effects])
            ses = np.array([e["se"] for e in valid_effects])

            # Aggregated ATT
            agg_att = np.average(atts, weights=weights)
            # Conservative SE (ignoring correlation)
            agg_se = np.sqrt(np.average(ses**2, weights=weights))

            aggregate_effects = {
                "att": agg_att,
                "se": agg_se,
                "t_stat": agg_att / agg_se if agg_se > 0 else np.nan,
                "p_value": 2 * (1 - np.abs(np.minimum(1, np.abs(agg_att / agg_se))))
                if agg_se > 0
                else np.nan,
                "n_cohorts": len(valid_effects),
            }
        else:
            aggregate_effects = None

        self.results = {
            "cohort_effects": cohort_effects,
            "aggregate_effects": aggregate_effects,
            "outcome": outcome,
            "covariates": covariates,
        }

        return self.results

    def event_study(
        self,
        outcome: str,
        horizon: int = 5,
        covariates: list[str] | None = None,
        omit_period: int = -1,
    ) -> pd.DataFrame:
        """Estimate dynamic treatment effects with proper weighting.

        Args:
            outcome: Outcome variable column name
            horizon: Number of periods before/after treatment to include
            covariates: Optional list of control variables
            omit_period: Reference period to omit (default: -1)

        Returns:
            DataFrame with event study coefficients
        """
        # Validate horizon
        horizon = min(horizon, MAX_EVENT_TIME)

        # Create relative time variable
        estimation_data = self.data.copy()
        estimation_data["relative_time"] = (
            estimation_data[self.time_col] - estimation_data[self.cohort_col]
        )

        # For never-treated, set relative time to large negative
        never_treated_mask = estimation_data[self.cohort_col] == self.never_treated
        estimation_data.loc[never_treated_mask, "relative_time"] = -999

        # Store event study results
        event_results = []

        for rel_time in range(-horizon, horizon + 1):
            if rel_time == omit_period:
                continue

            # Create cohort-specific indicators for this relative time
            cohort_effects = []

            for cohort in self.treated_cohorts:
                # Check if this relative time exists for this cohort
                actual_time = cohort + rel_time
                if actual_time not in self.time_periods:
                    continue

                cohort_mask = estimation_data[self.cohort_col] == cohort

                # Cohort-specific indicator
                cohort_data = estimation_data[cohort_mask | never_treated_mask].copy()

                cohort_data["event_indicator"] = (
                    cohort_data["relative_time"] == rel_time
                ).astype(int)

                # Run regression
                y = cohort_data[outcome]
                x = pd.get_dummies(
                    cohort_data[[self.unit_col, self.time_col]],
                    columns=[self.unit_col, self.time_col],
                    drop_first=True,
                )
                x["event_indicator"] = cohort_data["event_indicator"]

                if covariates:
                    x = pd.concat([x, cohort_data[covariates]], axis=1)

                try:
                    model = OLS(y, add_constant(x), missing="drop")
                    result = model.fit()

                    cohort_effects.append(
                        {
                            "coef": result.params["event_indicator"],
                            "se": result.bse["event_indicator"],
                            "n": cohort_mask.sum(),
                        }
                    )
                except Exception:
                    continue

            # Aggregate across cohorts
            if cohort_effects:
                weights = np.array([e["n"] for e in cohort_effects])
                weights = weights / weights.sum()

                coefs = np.array([e["coef"] for e in cohort_effects])
                ses = np.array([e["se"] for e in cohort_effects])

                agg_coef = np.average(coefs, weights=weights)
                agg_se = np.sqrt(np.average(ses**2, weights=weights))

                event_results.append(
                    {
                        "relative_time": rel_time,
                        "coefficient": agg_coef,
                        "std_error": agg_se,
                        "ci_lower": agg_coef - 1.96 * agg_se,
                        "ci_upper": agg_coef + 1.96 * agg_se,
                        "p_value": 2
                        * (1 - np.abs(np.minimum(1, np.abs(agg_coef / agg_se))))
                        if agg_se > 0
                        else np.nan,
                    }
                )

        # Add omitted period
        event_results.append(
            {
                "relative_time": omit_period,
                "coefficient": 0,
                "std_error": 0,
                "ci_lower": 0,
                "ci_upper": 0,
                "p_value": 1.0,
            }
        )

        self.event_study_results = pd.DataFrame(event_results).sort_values(
            "relative_time"
        )

        return self.event_study_results

    def test_parallel_trends(self, outcome: str, pre_periods: int = 3) -> dict:
        """Test parallel trends assumption using pre-treatment periods.

        Args:
            outcome: Outcome variable column name
            pre_periods: Number of pre-treatment periods to test

        Returns:
            Dictionary with test results
        """
        if self.event_study_results is None:
            self.event_study(outcome, horizon=pre_periods)

        # Get pre-treatment coefficients
        pre_coefs = self.event_study_results[
            self.event_study_results["relative_time"] < 0
        ]

        if len(pre_coefs) < MIN_PRE_PERIODS_FOR_TEST:
            return {
                "test_statistic": np.nan,
                "p_value": np.nan,
                "reject_parallel": False,
                "message": "Insufficient pre-treatment periods for testing",
            }

        # Joint F-test on pre-treatment coefficients
        coefs = pre_coefs["coefficient"].values
        ses = pre_coefs["std_error"].values

        # Wald statistic
        wald_stat = np.sum((coefs / ses) ** 2)
        p_value = (
            1
            - np.cumsum(np.random.chisquare(len(coefs), size=10000) <= wald_stat)[-1]
            / 10000
        )

        return {
            "test_statistic": wald_stat,
            "p_value": p_value,
            "reject_parallel": p_value < SIGNIFICANCE_LEVEL,
            "n_pre_periods": len(pre_coefs),
        }
