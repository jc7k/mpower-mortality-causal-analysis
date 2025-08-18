"""Data Preparation Utilities for MPOWER Analysis.

This module provides specialized functions for preparing WHO MPOWER data
for causal inference analysis, including treatment definition, cohort creation,
and panel data structuring.
"""

import warnings

from typing import Any, Literal

import numpy as np
import pandas as pd

from pandas import DataFrame


class MPOWERDataPrep:
    """Data preparation utilities for MPOWER causal analysis.

    Handles the specific requirements for preparing WHO MPOWER tobacco control
    policy data for causal inference analysis, including:
    - Treatment definition and cohort creation
    - Panel data balancing and validation
    - Control variable preparation
    - Missing data handling

    Parameters:
        data (DataFrame): Raw MPOWER data
        country_col (str): Column name for country identifier
        year_col (str): Column name for year

    Example:
        >>> prep = MPOWERDataPrep(data=raw_data, country_col='country', year_col='year')
        >>> panel_data = prep.create_treatment_cohorts(
        ...     mpower_col='mpower_total_score',
        ...     threshold=25,
        ...     min_years_high=2
        ... )
        >>> analysis_data = prep.prepare_for_analysis(
        ...     outcome_cols=['mortality_rate'],
        ...     control_cols=['gdp_log', 'urban_pct']
        ... )
    """

    def __init__(
        self, data: DataFrame, country_col: str = "country", year_col: str = "year"
    ):
        """Initialize MPOWER Data Preparation."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame")

        if data.empty:
            raise ValueError("Data cannot be empty")

        self.data = data.copy()
        self.country_col = country_col
        self.year_col = year_col

        # Validate required columns
        required_cols = [country_col, year_col]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Store preparation results
        self._treatment_cohorts = None
        self._panel_data = None

        # Basic data validation
        self._validate_basic_structure()

    def _validate_basic_structure(self) -> None:
        """Validate basic data structure requirements."""
        # Check for duplicate country-year combinations
        duplicates = self.data.duplicated(subset=[self.country_col, self.year_col])
        if duplicates.any():
            n_duplicates = duplicates.sum()
            warnings.warn(
                f"Found {n_duplicates} duplicate country-year combinations",
                stacklevel=2,
            )

        # Check year range
        years = self.data[self.year_col].unique()
        year_range = [years.min(), years.max()]
        if year_range[1] - year_range[0] < 5:
            warnings.warn(
                f"Short time series: {year_range[1] - year_range[0] + 1} years",
                stacklevel=2,
            )

        # Check country coverage
        countries = self.data[self.country_col].unique()
        if len(countries) < 10:
            warnings.warn(f"Few countries in dataset: {len(countries)}", stacklevel=2)

    def create_treatment_cohorts(
        self,
        mpower_col: str,
        treatment_definition: Literal[
            "binary_threshold", "continuous_change", "component_based"
        ] = "binary_threshold",
        threshold: float | None = 25.0,
        min_years_high: int = 1,
        component_cols: list[str] | None = None,
        baseline_years: list[int] | None = None,
    ) -> DataFrame:
        """Create treatment cohorts based on MPOWER policy adoption.

        Args:
            mpower_col (str): Column with MPOWER total score
            treatment_definition (str): How to define treatment
                - 'binary_threshold': Countries achieving high MPOWER score
                - 'continuous_change': Based on score changes
                - 'component_based': Based on individual MPOWER components
            threshold (float): Threshold for high MPOWER score (for binary definition)
            min_years_high (int): Minimum years above threshold to be considered treated
            component_cols (List[str]): MPOWER component columns (for component-based)
            baseline_years (List[int]): Years to use for baseline (for continuous definition)

        Returns:
            DataFrame with treatment cohort assignments
        """
        if mpower_col not in self.data.columns:
            raise ValueError(f"MPOWER column '{mpower_col}' not found in data")

        data_with_cohorts = self.data.copy()

        if treatment_definition == "binary_threshold":
            cohorts = self._create_binary_threshold_cohorts(
                data_with_cohorts, mpower_col, threshold, min_years_high
            )
        elif treatment_definition == "continuous_change":
            cohorts = self._create_continuous_change_cohorts(
                data_with_cohorts, mpower_col, baseline_years
            )
        elif treatment_definition == "component_based":
            if not component_cols:
                raise ValueError(
                    "component_cols required for component_based treatment definition"
                )
            cohorts = self._create_component_based_cohorts(
                data_with_cohorts, component_cols, threshold
            )
        else:
            raise ValueError(f"Unknown treatment definition: {treatment_definition}")

        data_with_cohorts["treatment_cohort"] = cohorts
        self._treatment_cohorts = data_with_cohorts

        return data_with_cohorts

    def _create_binary_threshold_cohorts(
        self, data: DataFrame, mpower_col: str, threshold: float, min_years_high: int
    ) -> pd.Series:
        """Create cohorts based on binary threshold for MPOWER scores."""
        cohorts = pd.Series(0, index=data.index)  # 0 = never treated

        # For each country, find when they first achieve sustained high MPOWER
        for country in data[self.country_col].unique():
            country_data = data[data[self.country_col] == country].sort_values(
                self.year_col
            )

            if country_data.empty:
                continue

            # Identify periods with high MPOWER score
            high_periods = country_data[mpower_col] >= threshold

            if not high_periods.any():
                continue  # Never achieved high score

            # Find first sustained period of high scores
            country_years = country_data[self.year_col].values
            country_indices = country_data.index

            consecutive_count = 0
            treatment_year = None

            for _i, (year, is_high, _idx) in enumerate(
                zip(country_years, high_periods, country_indices, strict=False)
            ):
                if is_high:
                    consecutive_count += 1
                    if consecutive_count >= min_years_high and treatment_year is None:
                        treatment_year = year - (min_years_high - 1)
                        break
                else:
                    consecutive_count = 0

            # Assign treatment cohort
            if treatment_year is not None:
                country_mask = data[self.country_col] == country
                cohorts.loc[country_mask] = treatment_year

        return cohorts

    def _create_continuous_change_cohorts(
        self, data: DataFrame, mpower_col: str, baseline_years: list[int] | None
    ) -> pd.Series:
        """Create cohorts based on continuous changes in MPOWER scores."""
        if baseline_years is None:
            # Use first 3 years as baseline
            all_years = sorted(data[self.year_col].unique())
            baseline_years = all_years[:3]

        cohorts = pd.Series(0, index=data.index)

        for country in data[self.country_col].unique():
            country_data = data[data[self.country_col] == country].sort_values(
                self.year_col
            )

            # Calculate baseline MPOWER score
            baseline_data = country_data[
                country_data[self.year_col].isin(baseline_years)
            ]
            if baseline_data.empty:
                continue

            baseline_score = baseline_data[mpower_col].mean()

            # Find year of largest improvement
            country_data = country_data.copy()
            country_data["mpower_change"] = country_data[mpower_col] - baseline_score

            # Find first year with substantial improvement (>20% increase)
            improvement_threshold = 0.2 * baseline_score
            substantial_improvement = (
                country_data["mpower_change"] >= improvement_threshold
            )

            if substantial_improvement.any():
                treatment_year = country_data[substantial_improvement][
                    self.year_col
                ].iloc[0]
                country_mask = data[self.country_col] == country
                cohorts.loc[country_mask] = treatment_year

        return cohorts

    def _create_component_based_cohorts(
        self, data: DataFrame, component_cols: list[str], threshold: float
    ) -> pd.Series:
        """Create cohorts based on individual MPOWER components."""
        missing_components = [col for col in component_cols if col not in data.columns]
        if missing_components:
            raise ValueError(f"Component columns not found: {missing_components}")

        cohorts = pd.Series(0, index=data.index)

        # For each country, find when they first implement multiple components strongly
        for country in data[self.country_col].unique():
            country_data = data[data[self.country_col] == country].sort_values(
                self.year_col
            )

            # Count strong implementations per year
            strong_implementations = (country_data[component_cols] >= threshold).sum(
                axis=1
            )

            # Find first year with majority of components implemented strongly
            majority_threshold = len(component_cols) / 2
            strong_policy_years = strong_implementations >= majority_threshold

            if strong_policy_years.any():
                treatment_year = country_data[strong_policy_years][self.year_col].iloc[
                    0
                ]
                country_mask = data[self.country_col] == country
                cohorts.loc[country_mask] = treatment_year

        return cohorts

    def balance_panel(
        self,
        data: DataFrame | None = None,
        method: Literal[
            "drop_unbalanced", "fill_missing", "interpolate"
        ] = "drop_unbalanced",
        min_years: int = 5,
    ) -> DataFrame:
        """Balance the panel data for causal inference analysis.

        Args:
            data (DataFrame, optional): Data to balance (uses stored data if None)
            method (str): Method for balancing
                - 'drop_unbalanced': Drop countries without full time series
                - 'fill_missing': Fill missing values with appropriate methods
                - 'interpolate': Interpolate missing values
            min_years (int): Minimum years required per country

        Returns:
            DataFrame with balanced panel
        """
        if data is None:
            data = (
                self._treatment_cohorts
                if self._treatment_cohorts is not None
                else self.data
            )

        data = data.copy()

        if method == "drop_unbalanced":
            balanced_data = self._drop_unbalanced_countries(data, min_years)
        elif method == "fill_missing":
            balanced_data = self._fill_missing_values(data)
        elif method == "interpolate":
            balanced_data = self._interpolate_missing_values(data)
        else:
            raise ValueError(f"Unknown balancing method: {method}")

        self._panel_data = balanced_data
        return balanced_data

    def _drop_unbalanced_countries(self, data: DataFrame, min_years: int) -> DataFrame:
        """Drop countries that don't have sufficient time series data."""
        # Count observations per country
        country_counts = data.groupby(self.country_col).size()

        # Keep countries with sufficient data
        valid_countries = country_counts[country_counts >= min_years].index

        filtered_data = data[data[self.country_col].isin(valid_countries)].copy()

        dropped_countries = len(data[self.country_col].unique()) - len(valid_countries)
        if dropped_countries > 0:
            warnings.warn(
                f"Dropped {dropped_countries} countries with insufficient data",
                stacklevel=2,
            )

        return filtered_data

    def _fill_missing_values(self, data: DataFrame) -> DataFrame:
        """Fill missing values using appropriate methods."""
        filled_data = data.copy()

        # Identify numeric columns for filling
        numeric_cols = filled_data.select_dtypes(include=[np.number]).columns

        # Forward fill within countries, then backward fill
        for country in filled_data[self.country_col].unique():
            country_mask = filled_data[self.country_col] == country
            filled_data.loc[country_mask, numeric_cols] = (
                filled_data.loc[country_mask, numeric_cols]
                .fillna(method="ffill")
                .fillna(method="bfill")
            )

        return filled_data

    def _interpolate_missing_values(self, data: DataFrame) -> DataFrame:
        """Interpolate missing values within countries."""
        interpolated_data = data.copy()

        # Identify numeric columns
        numeric_cols = interpolated_data.select_dtypes(include=[np.number]).columns

        # Interpolate within countries
        for country in interpolated_data[self.country_col].unique():
            country_mask = interpolated_data[self.country_col] == country
            country_data = interpolated_data.loc[country_mask].sort_values(
                self.year_col
            )

            # Linear interpolation
            interpolated_values = country_data[numeric_cols].interpolate(
                method="linear"
            )
            interpolated_data.loc[country_mask, numeric_cols] = interpolated_values

        return interpolated_data

    def prepare_for_analysis(
        self,
        outcome_cols: list[str],
        control_cols: list[str] | None = None,
        log_transform: list[str] | None = None,
        standardize: list[str] | None = None,
        create_lags: dict[str, int] | None = None,
    ) -> DataFrame:
        """Prepare final dataset for causal inference analysis.

        Args:
            outcome_cols (List[str]): Outcome variable columns
            control_cols (List[str], optional): Control variable columns
            log_transform (List[str], optional): Variables to log-transform
            standardize (List[str], optional): Variables to standardize
            create_lags (Dict[str, int], optional): Variables and lag lengths to create

        Returns:
            DataFrame ready for causal inference analysis
        """
        # Use balanced panel data if available
        if self._panel_data is not None:
            analysis_data = self._panel_data.copy()
        elif self._treatment_cohorts is not None:
            analysis_data = self._treatment_cohorts.copy()
        else:
            analysis_data = self.data.copy()

        # Validate required columns
        all_required_cols = outcome_cols.copy()
        if control_cols:
            all_required_cols.extend(control_cols)

        missing_cols = [
            col for col in all_required_cols if col not in analysis_data.columns
        ]
        if missing_cols:
            raise ValueError(f"Required columns not found: {missing_cols}")

        # Log transformation
        if log_transform:
            for col in log_transform:
                if col in analysis_data.columns:
                    # Add small constant to handle zeros
                    analysis_data[f"{col}_log"] = np.log(analysis_data[col] + 1)

        # Standardization
        if standardize:
            for col in standardize:
                if col in analysis_data.columns:
                    mean_val = analysis_data[col].mean()
                    std_val = analysis_data[col].std()
                    analysis_data[f"{col}_std"] = (
                        analysis_data[col] - mean_val
                    ) / std_val

        # Create lagged variables
        if create_lags:
            analysis_data = analysis_data.sort_values([self.country_col, self.year_col])

            for var, max_lag in create_lags.items():
                if var in analysis_data.columns:
                    for lag in range(1, max_lag + 1):
                        analysis_data[f"{var}_lag{lag}"] = analysis_data.groupby(
                            self.country_col
                        )[var].shift(lag)

        # Create additional useful variables
        analysis_data = self._create_additional_variables(analysis_data)

        return analysis_data

    def _create_additional_variables(self, data: DataFrame) -> DataFrame:
        """Create additional variables useful for causal inference."""
        enhanced_data = data.copy()

        # Treatment indicator (1 if ever treated, 0 if never treated)
        if "treatment_cohort" in enhanced_data.columns:
            enhanced_data["ever_treated"] = (
                enhanced_data["treatment_cohort"] > 0
            ).astype(int)

            # Post-treatment indicator
            enhanced_data["post_treatment"] = 0
            for country in enhanced_data[self.country_col].unique():
                country_mask = enhanced_data[self.country_col] == country
                country_data = enhanced_data[country_mask]

                if country_data["treatment_cohort"].iloc[0] > 0:
                    treatment_year = country_data["treatment_cohort"].iloc[0]
                    post_mask = country_mask & (
                        enhanced_data[self.year_col] >= treatment_year
                    )
                    enhanced_data.loc[post_mask, "post_treatment"] = 1

            # Years since treatment
            enhanced_data["years_since_treatment"] = 0
            treated_mask = enhanced_data["ever_treated"] == 1
            enhanced_data.loc[treated_mask, "years_since_treatment"] = (
                enhanced_data.loc[treated_mask, self.year_col]
                - enhanced_data.loc[treated_mask, "treatment_cohort"]
            )

        return enhanced_data

    def generate_summary_report(self) -> dict[str, Any]:
        """Generate a comprehensive summary report of the data preparation.

        Returns:
            Dict with summary statistics and information
        """
        report = {}

        # Basic data structure
        report["basic_structure"] = {
            "n_countries": self.data[self.country_col].nunique(),
            "n_years": self.data[self.year_col].nunique(),
            "year_range": [
                self.data[self.year_col].min(),
                self.data[self.year_col].max(),
            ],
            "n_observations": len(self.data),
            "countries": self.data[self.country_col].unique().tolist(),
        }

        # Treatment cohort analysis
        if self._treatment_cohorts is not None:
            cohort_summary = (
                self._treatment_cohorts.groupby("treatment_cohort")
                .agg({self.country_col: "nunique", self.year_col: "count"})
                .rename(
                    columns={
                        self.country_col: "n_countries",
                        self.year_col: "n_observations",
                    }
                )
            )

            report["treatment_cohorts"] = {
                "cohort_distribution": cohort_summary.to_dict(),
                "n_treated_countries": (
                    self._treatment_cohorts["treatment_cohort"] > 0
                ).sum(),
                "n_never_treated_countries": (
                    self._treatment_cohorts["treatment_cohort"] == 0
                ).sum(),
                "treatment_years": sorted(
                    self._treatment_cohorts[
                        self._treatment_cohorts["treatment_cohort"] > 0
                    ]["treatment_cohort"]
                    .unique()
                    .tolist()
                ),
            }

        # Panel balance analysis
        if self._panel_data is not None:
            panel_balance = self._panel_data.groupby(self.country_col).size()

            report["panel_balance"] = {
                "is_balanced": panel_balance.nunique() == 1,
                "min_observations_per_country": panel_balance.min(),
                "max_observations_per_country": panel_balance.max(),
                "mean_observations_per_country": panel_balance.mean(),
                "countries_with_full_data": (
                    panel_balance == panel_balance.max()
                ).sum(),
            }

        # Missing data analysis
        missing_analysis = {}
        for col in self.data.columns:
            if self.data[col].dtype in ["float64", "int64"]:
                missing_pct = self.data[col].isnull().mean() * 100
                if missing_pct > 0:
                    missing_analysis[col] = missing_pct

        report["missing_data"] = missing_analysis

        return report

    def export_prepared_data(
        self,
        filepath: str,
        data_type: Literal["raw", "cohorts", "balanced", "analysis"] = "analysis",
        format: str = "csv",
    ) -> None:
        """Export prepared data to file.

        Args:
            filepath (str): Output file path
            data_type (str): Type of data to export
            format (str): Export format ('csv', 'parquet', 'excel')
        """
        # Select data to export
        if data_type == "raw":
            export_data = self.data
        elif data_type == "cohorts":
            if self._treatment_cohorts is None:
                raise ValueError("Treatment cohorts not created yet")
            export_data = self._treatment_cohorts
        elif data_type == "balanced":
            if self._panel_data is None:
                raise ValueError("Panel data not balanced yet")
            export_data = self._panel_data
        elif data_type == "analysis":
            if self._panel_data is not None:
                export_data = self._panel_data
            elif self._treatment_cohorts is not None:
                export_data = self._treatment_cohorts
            else:
                export_data = self.data
        else:
            raise ValueError(f"Unknown data type: {data_type}")

        # Export in specified format
        if format == "csv":
            export_data.to_csv(filepath, index=False)
        elif format == "parquet":
            export_data.to_parquet(filepath, index=False)
        elif format == "excel":
            export_data.to_excel(filepath, index=False)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    @staticmethod
    def validate_mpower_data(
        data: DataFrame, required_cols: list[str] | None = None
    ) -> dict[str, Any]:
        """Validate MPOWER data structure and content.

        Args:
            data (DataFrame): MPOWER data to validate
            required_cols (List[str], optional): Required columns to check for

        Returns:
            Dict with validation results
        """
        validation = {"valid": True, "issues": []}

        # Check basic structure
        if data.empty:
            validation["valid"] = False
            validation["issues"].append("Data is empty")
            return validation

        # Check required columns
        if required_cols:
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                validation["valid"] = False
                validation["issues"].append(f"Missing required columns: {missing_cols}")

        # Check for reasonable year range
        if "year" in data.columns:
            years = data["year"].unique()
            if len(years) < 3:
                validation["issues"].append("Very short time series (< 3 years)")
            if years.min() < 2000 or years.max() > 2030:
                validation["issues"].append("Unusual year range detected")

        # Check for reasonable MPOWER scores
        mpower_cols = [col for col in data.columns if "mpower" in col.lower()]
        for col in mpower_cols:
            if data[col].min() < 0 or data[col].max() > 100:
                validation["issues"].append(f"Unusual MPOWER scores in {col}")

        # Check for missing data patterns
        missing_pct = data.isnull().mean()
        high_missing_cols = missing_pct[missing_pct > 0.5].index.tolist()
        if high_missing_cols:
            validation["issues"].append(
                f"High missing data (>50%) in: {high_missing_cols}"
            )

        return validation
