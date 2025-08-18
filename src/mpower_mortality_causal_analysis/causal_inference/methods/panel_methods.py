"""Panel Data Fixed Effects Methods Implementation.

This module provides wrappers for panel data methods using both pyfixest
and linearmodels packages, optimized for MPOWER policy analysis.
"""

import warnings

from typing import Any, Literal

import pandas as pd

from pandas import DataFrame

# Try to import panel data packages
try:
    import pyfixest as pf

    PYFIXEST_AVAILABLE = True
except ImportError:
    PYFIXEST_AVAILABLE = False
    warnings.warn(
        "pyfixest not available. Some functionality will be limited.", stacklevel=2
    )

try:
    from linearmodels import FirstDifferenceOLS, PanelOLS, PooledOLS, RandomEffects
    from linearmodels.panel.data import PanelData

    LINEARMODELS_AVAILABLE = True
except ImportError:
    LINEARMODELS_AVAILABLE = False
    warnings.warn(
        "linearmodels not available. Some functionality will be limited.", stacklevel=2
    )

from ..utils.base import CausalInferenceBase


class PanelFixedEffects(CausalInferenceBase):
    """Panel Data Fixed Effects Estimator.

    Provides a unified interface for various panel data estimators including:
    - Two-way fixed effects (TWFE)
    - Entity fixed effects
    - Time fixed effects
    - Random effects
    - First differences
    - Pooled OLS

    Supports both pyfixest (R fixest-style) and linearmodels backends.

    Parameters:
        data (DataFrame): Panel data
        unit_col (str): Column name for unit identifier
        time_col (str): Column name for time identifier
        backend (str): Backend to use ('pyfixest', 'linearmodels', 'auto')

    Example:
        >>> panel = PanelFixedEffects(data=panel_data, unit_col='country', time_col='year')
        >>> results = panel.fit('mortality_rate ~ mpower_total + gdp_log | country + year')
        >>> print(panel.summary())
    """

    def __init__(
        self,
        data: DataFrame,
        unit_col: str,
        time_col: str,
        backend: Literal["pyfixest", "linearmodels", "auto"] = "auto",
    ):
        """Initialize Panel Fixed Effects estimator."""
        super().__init__(data)

        self.unit_col = unit_col
        self.time_col = time_col
        self.backend = self._determine_backend(backend)

        # Validate required columns
        required_cols = [unit_col, time_col]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Store fitted model and results
        self._fitted_model = None
        self._results = None
        self._formula = None

        # Validate panel structure
        panel_info = self.validate_panel_structure(unit_col, time_col)
        self._panel_info = panel_info

    def _determine_backend(self, backend: str) -> str:
        """Determine which backend to use based on availability and preference."""
        if backend == "auto":
            if PYFIXEST_AVAILABLE:
                return "pyfixest"
            if LINEARMODELS_AVAILABLE:
                return "linearmodels"
            raise ImportError(
                "Neither pyfixest nor linearmodels is available. "
                "Please install one of them: pip install pyfixest OR pip install linearmodels"
            )
        if backend == "pyfixest":
            if not PYFIXEST_AVAILABLE:
                raise ImportError(
                    "pyfixest not available. Install with: pip install pyfixest"
                )
            return "pyfixest"
        if backend == "linearmodels":
            if not LINEARMODELS_AVAILABLE:
                raise ImportError(
                    "linearmodels not available. Install with: pip install linearmodels"
                )
            return "linearmodels"
        raise ValueError(f"Unknown backend: {backend}")

    def fit(
        self,
        formula: str | None = None,
        outcome: str | None = None,
        covariates: list[str] | None = None,
        fixed_effects: list[str] | None = None,
        vcov: str | None = None,
        method: Literal["twfe", "entity", "time", "pooled", "random", "fd"] = "twfe",
        **kwargs,
    ) -> "PanelFixedEffects":
        """Fit panel data model.

        Args:
            formula (str, optional): Formula in fixest style (e.g., 'y ~ x1 + x2 | fe1 + fe2')
            outcome (str, optional): Outcome variable (alternative to formula)
            covariates (List[str], optional): Covariate variables (alternative to formula)
            fixed_effects (List[str], optional): Fixed effect variables (alternative to formula)
            vcov (str, optional): Variance-covariance estimation method
            method (str): Panel method to use
            **kwargs: Additional arguments passed to the underlying estimator

        Returns:
            PanelFixedEffects: Fitted estimator instance
        """
        # Determine the specification
        if formula:
            self._formula = formula
        else:
            if not outcome:
                raise ValueError("Either formula or outcome must be specified")
            self._formula = self._build_formula(
                outcome, covariates, fixed_effects, method
            )

        # Set default clustering if not specified
        if vcov is None:
            if method in ["twfe", "entity", "time"]:
                vcov = f"CL1({self.unit_col})"  # Cluster by entity
            else:
                vcov = "robust"

        # Fit using the specified backend
        if self.backend == "pyfixest":
            self._fitted_model = self._fit_pyfixest(self._formula, vcov, **kwargs)
        elif self.backend == "linearmodels":
            self._fitted_model = self._fit_linearmodels(
                outcome or self._extract_outcome_from_formula(self._formula),
                covariates,
                method,
                vcov,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        return self

    def _build_formula(
        self,
        outcome: str,
        covariates: list[str] | None,
        fixed_effects: list[str] | None,
        method: str,
    ) -> str:
        """Build formula string from components."""
        # Build the basic formula
        if covariates:
            formula = f"{outcome} ~ {' + '.join(covariates)}"
        else:
            formula = f"{outcome} ~ 1"

        # Add fixed effects based on method
        if method == "twfe":
            fe_vars = [self.unit_col, self.time_col]
        elif method == "entity":
            fe_vars = [self.unit_col]
        elif method == "time":
            fe_vars = [self.time_col]
        else:
            fe_vars = []

        # Add custom fixed effects
        if fixed_effects:
            fe_vars.extend(fixed_effects)

        if fe_vars:
            formula += f" | {' + '.join(fe_vars)}"

        return formula

    def _extract_outcome_from_formula(self, formula: str) -> str:
        """Extract outcome variable name from formula."""
        return formula.split("~")[0].strip()

    def _fit_pyfixest(self, formula: str, vcov: str, **kwargs) -> Any:
        """Fit using pyfixest backend."""
        try:
            model = pf.feols(formula, data=self.data, vcov=vcov, **kwargs)

            self._results = {
                "model": model,
                "backend": "pyfixest",
                "formula": formula,
                "vcov": vcov,
                "coefficients": model.coef(),
                "std_errors": model.se(),
                "pvalues": model.pvalue(),
                "nobs": model.nobs,
                "r2": model.r2,
                "r2_adj": model.r2_adj,
            }

            return model

        except Exception as e:
            raise RuntimeError(f"Failed to fit with pyfixest: {e}")

    def _fit_linearmodels(
        self,
        outcome: str,
        covariates: list[str] | None,
        method: str,
        vcov: str,
        **kwargs,
    ) -> Any:
        """Fit using linearmodels backend."""
        try:
            # Prepare data for linearmodels
            data_copy = self.data.copy()

            # Set MultiIndex for panel data
            data_copy = data_copy.set_index([self.unit_col, self.time_col])

            # Prepare dependent variable
            dependent = data_copy[outcome]

            # Prepare exogenous variables
            exog = data_copy[covariates] if covariates else None

            # Choose model based on method
            if method == "pooled":
                model_class = PooledOLS
                model_kwargs = {}
            elif method == "random":
                model_class = RandomEffects
                model_kwargs = {}
            elif method == "fd":
                model_class = FirstDifferenceOLS
                model_kwargs = {}
            else:  # twfe, entity, time
                model_class = PanelOLS
                model_kwargs = {
                    "entity_effects": method in ["twfe", "entity"],
                    "time_effects": method in ["twfe", "time"],
                }

            # Initialize model
            if exog is not None:
                model = model_class(dependent=dependent, exog=exog, **model_kwargs)
            else:
                model = model_class(dependent=dependent, **model_kwargs)

            # Determine clustering
            if "CL1" in vcov:
                cluster_entity = True
                cov_type = "clustered"
            elif vcov == "robust":
                cluster_entity = False
                cov_type = "robust"
            else:
                cluster_entity = False
                cov_type = "unadjusted"

            # Fit model
            fitted_model = model.fit(
                cov_type=cov_type, cluster_entity=cluster_entity, **kwargs
            )

            self._results = {
                "model": fitted_model,
                "backend": "linearmodels",
                "method": method,
                "vcov": vcov,
                "coefficients": fitted_model.params,
                "std_errors": fitted_model.std_errors,
                "pvalues": fitted_model.pvalues,
                "nobs": fitted_model.nobs,
                "r2": fitted_model.rsquared,
                "r2_adj": fitted_model.rsquared_adj
                if hasattr(fitted_model, "rsquared_adj")
                else None,
            }

            return fitted_model

        except Exception as e:
            raise RuntimeError(f"Failed to fit with linearmodels: {e}")

    def summary(self) -> str:
        """Return a summary of the fitted model."""
        if not self._fitted_model:
            return "Model not fitted yet. Call fit() first."

        if self.backend == "pyfixest" and hasattr(self._fitted_model, "summary"):
            return str(self._fitted_model.summary())
        if self.backend == "linearmodels" and hasattr(self._fitted_model, "summary"):
            return str(self._fitted_model.summary)
        # Fallback summary
        results = self._results
        return f"""
Panel Fixed Effects Results ({self.backend})
==========================================
Formula: {results.get("formula", "N/A")}
Method: {results.get("method", "N/A")}
Observations: {results.get("nobs", "N/A")}
R-squared: {results.get("r2", "N/A"):.4f}
Adj. R-squared: {results.get("r2_adj", "N/A"):.4f}

Panel Structure:
  Units: {self._panel_info["n_units"]}
  Time periods: {self._panel_info["n_periods"]}
  Balanced: {self._panel_info["is_balanced"]}
"""

    def get_coefficients(self) -> pd.DataFrame:
        """Get coefficient table with standard errors and p-values.

        Returns:
            DataFrame with coefficient results
        """
        if not self._fitted_model:
            raise ValueError("Model must be fitted first")

        results = self._results

        coef_df = pd.DataFrame(
            {
                "coefficient": results["coefficients"],
                "std_error": results["std_errors"],
                "p_value": results["pvalues"],
            }
        )

        # Add confidence intervals
        coef_df["ci_lower"] = coef_df["coefficient"] - 1.96 * coef_df["std_error"]
        coef_df["ci_upper"] = coef_df["coefficient"] + 1.96 * coef_df["std_error"]

        # Add significance stars
        coef_df["significance"] = coef_df["p_value"].apply(self._get_significance_stars)

        return coef_df

    @staticmethod
    def _get_significance_stars(p_value: float) -> str:
        """Convert p-value to significance stars."""
        if p_value < 0.001:
            return "***"
        if p_value < 0.01:
            return "**"
        if p_value < 0.05:
            return "*"
        if p_value < 0.1:
            return "."
        return ""

    def predict(self, data: DataFrame | None = None) -> pd.Series:
        """Generate predictions from fitted model.

        Args:
            data (DataFrame, optional): Data for prediction (uses training data if None)

        Returns:
            Series with predictions
        """
        if not self._fitted_model:
            raise ValueError("Model must be fitted first")

        if data is None:
            data = self.data

        try:
            if self.backend == "pyfixest" and hasattr(self._fitted_model, "predict"):
                return self._fitted_model.predict(data)
            if self.backend == "linearmodels" and hasattr(
                self._fitted_model, "predict"
            ):
                # For linearmodels, need to format data properly
                data.set_index([self.unit_col, self.time_col])
                return self._fitted_model.predict()
            warnings.warn(
                "Prediction not implemented for this backend/model combination",
                stacklevel=2,
            )
            return pd.Series(index=data.index, dtype=float)
        except Exception as e:
            warnings.warn(f"Prediction failed: {e}", stacklevel=2)
            return pd.Series(index=data.index, dtype=float)

    def residuals(self) -> pd.Series:
        """Get model residuals.

        Returns:
            Series with residuals
        """
        if not self._fitted_model:
            raise ValueError("Model must be fitted first")

        try:
            if self.backend == "pyfixest" and hasattr(self._fitted_model, "resid"):
                return self._fitted_model.resid()
            if self.backend == "linearmodels" and hasattr(self._fitted_model, "resids"):
                return self._fitted_model.resids
            warnings.warn(
                "Residuals not available for this backend/model combination",
                stacklevel=2,
            )
            return pd.Series(dtype=float)
        except Exception as e:
            warnings.warn(f"Failed to get residuals: {e}", stacklevel=2)
            return pd.Series(dtype=float)

    def fit_multiple_outcomes(
        self,
        outcomes: list[str],
        covariates: list[str] | None = None,
        fixed_effects: list[str] | None = None,
        method: str = "twfe",
        vcov: str | None = None,
        **kwargs,
    ) -> dict[str, "PanelFixedEffects"]:
        """Fit models for multiple outcomes simultaneously.

        Args:
            outcomes (List[str]): List of outcome variables
            covariates (List[str], optional): Covariate variables
            fixed_effects (List[str], optional): Fixed effect variables
            method (str): Panel method to use
            vcov (str, optional): Variance-covariance estimation
            **kwargs: Additional arguments

        Returns:
            Dict mapping outcome names to fitted PanelFixedEffects instances
        """
        results = {}

        for outcome in outcomes:
            # Create a new instance for each outcome
            panel_model = PanelFixedEffects(
                data=self.data,
                unit_col=self.unit_col,
                time_col=self.time_col,
                backend=self.backend,
            )

            # Fit the model
            panel_model.fit(
                outcome=outcome,
                covariates=covariates,
                fixed_effects=fixed_effects,
                method=method,
                vcov=vcov,
                **kwargs,
            )

            results[outcome] = panel_model

        return results

    def export_results(self, filepath: str, format: str = "csv") -> None:
        """Export model results to file.

        Args:
            filepath (str): Output file path
            format (str): Export format ('csv', 'json', 'excel')
        """
        if not self._fitted_model:
            raise ValueError("Model must be fitted first")

        # Get coefficient table
        coef_table = self.get_coefficients()

        # Add model metadata
        metadata = {
            "backend": self.backend,
            "formula": self._formula,
            "n_observations": self._results.get("nobs"),
            "r_squared": self._results.get("r2"),
            "adj_r_squared": self._results.get("r2_adj"),
            "n_units": self._panel_info["n_units"],
            "n_periods": self._panel_info["n_periods"],
            "is_balanced": self._panel_info["is_balanced"],
        }

        if format == "csv":
            coef_table.to_csv(filepath)
        elif format == "excel":
            with pd.ExcelWriter(filepath) as writer:
                coef_table.to_excel(writer, sheet_name="Coefficients")
                pd.DataFrame([metadata]).to_excel(
                    writer, sheet_name="Metadata", index=False
                )
        elif format == "json":
            results_dict = {"coefficients": coef_table.to_dict(), "metadata": metadata}
            import json

            with open(filepath, "w") as f:
                json.dump(results_dict, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")
