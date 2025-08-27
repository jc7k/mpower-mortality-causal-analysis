"""Spatial econometric models for spillover analysis.

This module implements spatial panel data models including Spatial Lag (SAR),
Spatial Error (SEM), and Spatial Durbin (SDM) models for estimating spillover effects.
"""

import logging

import numpy as np
import pandas as pd

from scipy import optimize
from scipy.stats import chi2, norm

logger = logging.getLogger(__name__)


class SpatialPanelModel:
    """Spatial panel data models for spillover estimation.

    Implements maximum likelihood estimation for spatial econometric models
    that capture spillover effects through spatial dependence.

    Args:
        data: Panel data with columns for unit, time, outcome, and covariates
        W: Spatial weight matrix (n_units x n_units)
        unit_col: Column name for unit identifier
        time_col: Column name for time identifier
    """

    def __init__(
        self,
        data: pd.DataFrame,
        W: np.ndarray,
        unit_col: str = "country",
        time_col: str = "year",
    ):
        """Initialize spatial panel model.

        Args:
            data: Panel data DataFrame
            W: Spatial weight matrix
            unit_col: Name of unit identifier column
            time_col: Name of time identifier column
        """
        self.data = data.copy()
        self.W = W
        self.unit_col = unit_col
        self.time_col = time_col

        # Set up panel structure
        self.units = sorted(data[unit_col].unique())
        self.n_units = len(self.units)
        self.times = sorted(data[time_col].unique())
        self.n_times = len(self.times)
        self.n_obs = len(data)

        # Validate weight matrix dimensions
        if W.shape != (self.n_units, self.n_units):
            raise ValueError(
                f"Weight matrix shape {W.shape} doesn't match number of units {self.n_units}"
            )

        # Precompute useful matrices
        self.I = np.eye(self.n_units)
        self.eigenvalues = None  # Will compute when needed

        # Results storage
        self.results = {}

    def spatial_lag_model(
        self, outcome: str, covariates: list[str], fixed_effects: bool = True
    ) -> dict:
        """Estimate Spatial Lag Model (SAR).

        Model: Y = ρWY + Xβ + ε

        The spatial lag term ρWY captures the effect of neighbors' outcomes
        on own outcome (endogenous interaction effect).

        Args:
            outcome: Name of outcome variable
            covariates: List of covariate names
            fixed_effects: Include unit fixed effects

        Returns:
            Dictionary with estimation results
        """
        logger.info(f"Estimating Spatial Lag Model for {outcome}")

        # Prepare data matrices
        y, X = self._prepare_data(outcome, covariates, fixed_effects)

        # Initial OLS estimates for starting values
        beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]

        # Spatial lag of y
        Wy = self._spatial_lag(y)

        # Maximum likelihood estimation
        def log_likelihood(params):
            rho = params[0]
            beta = params[1:]

            # Check rho bounds
            if abs(rho) >= 0.999:
                return 1e10

            # Residuals
            e = y - rho * Wy - X @ beta
            sigma2 = (e.T @ e) / self.n_obs

            # Log-likelihood
            A = self.I - rho * self.W
            log_det = np.linalg.slogdet(A)[1] * self.n_times
            ll = -0.5 * self.n_obs * (np.log(2 * np.pi) + np.log(sigma2))
            ll += log_det - 0.5 * (e.T @ e) / sigma2

            return -ll  # Minimize negative log-likelihood

        # Optimize
        init_params = np.concatenate([[0.1], beta_ols])
        result = optimize.minimize(
            log_likelihood,
            init_params,
            method="L-BFGS-B",
            bounds=[(-0.999, 0.999)] + [(None, None)] * len(beta_ols),
        )

        # Extract results
        rho = result.x[0]
        beta = result.x[1:]

        # Calculate standard errors
        se = self._calculate_standard_errors_sar(y, X, rho, beta)

        # Direct, indirect, and total effects
        effects = self._calculate_spatial_effects(beta, rho, "lag")

        # Spatial autocorrelation tests
        residuals = y - rho * Wy - X @ beta
        moran_i = self._morans_i(residuals)

        return {
            "model": "Spatial Lag (SAR)",
            "outcome": outcome,
            "rho": rho,
            "rho_se": se[0] if len(se) > 0 else np.nan,
            "rho_pvalue": 2 * (1 - norm.cdf(abs(rho / se[0])))
            if len(se) > 0
            else np.nan,
            "beta": beta,
            "beta_se": se[1:] if len(se) > 1 else np.full(len(beta), np.nan),
            "covariates": covariates,
            "direct_effects": effects["direct"],
            "indirect_effects": effects["indirect"],
            "total_effects": effects["total"],
            "log_likelihood": -result.fun,
            "aic": 2 * (len(init_params) + result.fun),
            "bic": np.log(self.n_obs) * len(init_params) + 2 * result.fun,
            "moran_i": moran_i,
            "n_obs": self.n_obs,
            "converged": result.success,
        }

    def spatial_error_model(
        self, outcome: str, covariates: list[str], fixed_effects: bool = True
    ) -> dict:
        """Estimate Spatial Error Model (SEM).

        Model: Y = Xβ + u, where u = λWu + ε

        The spatial error term λWu captures spatial correlation in unobservables.

        Args:
            outcome: Name of outcome variable
            covariates: List of covariate names
            fixed_effects: Include unit fixed effects

        Returns:
            Dictionary with estimation results
        """
        logger.info(f"Estimating Spatial Error Model for {outcome}")

        # Prepare data matrices
        y, X = self._prepare_data(outcome, covariates, fixed_effects)

        # Initial OLS estimates
        beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]

        # Maximum likelihood estimation
        def log_likelihood(params):
            lambda_param = params[0]
            beta = params[1:]

            # Check lambda bounds
            if abs(lambda_param) >= 0.999:
                return 1e10

            # Transformed residuals
            A = self.I - lambda_param * self.W
            e = y - X @ beta
            u = A @ e
            sigma2 = (u.T @ u) / self.n_obs

            # Log-likelihood
            log_det = np.linalg.slogdet(A)[1] * self.n_times
            ll = -0.5 * self.n_obs * (np.log(2 * np.pi) + np.log(sigma2))
            ll += log_det - 0.5 * (u.T @ u) / sigma2

            return -ll

        # Optimize
        init_params = np.concatenate([[0.1], beta_ols])
        result = optimize.minimize(
            log_likelihood,
            init_params,
            method="L-BFGS-B",
            bounds=[(-0.999, 0.999)] + [(None, None)] * len(beta_ols),
        )

        # Extract results
        lambda_param = result.x[0]
        beta = result.x[1:]

        # Calculate standard errors
        se = self._calculate_standard_errors_sem(y, X, lambda_param, beta)

        # Test for remaining spatial autocorrelation
        residuals = y - X @ beta
        moran_i = self._morans_i(residuals)

        return {
            "model": "Spatial Error (SEM)",
            "outcome": outcome,
            "lambda": lambda_param,
            "lambda_se": se[0] if len(se) > 0 else np.nan,
            "lambda_pvalue": 2 * (1 - norm.cdf(abs(lambda_param / se[0])))
            if len(se) > 0
            else np.nan,
            "beta": beta,
            "beta_se": se[1:] if len(se) > 1 else np.full(len(beta), np.nan),
            "covariates": covariates,
            "log_likelihood": -result.fun,
            "aic": 2 * (len(init_params) + result.fun),
            "bic": np.log(self.n_obs) * len(init_params) + 2 * result.fun,
            "moran_i": moran_i,
            "n_obs": self.n_obs,
            "converged": result.success,
        }

    def spatial_durbin_model(
        self, outcome: str, covariates: list[str], fixed_effects: bool = True
    ) -> dict:
        """Estimate Spatial Durbin Model (SDM).

        Model: Y = ρWY + Xβ + WXθ + ε

        The SDM includes both spatial lag of the outcome (ρWY) and spatial
        lags of covariates (WXθ), capturing both types of spatial dependence.

        Args:
            outcome: Name of outcome variable
            covariates: List of covariate names
            fixed_effects: Include unit fixed effects

        Returns:
            Dictionary with estimation results
        """
        logger.info(f"Estimating Spatial Durbin Model for {outcome}")

        # Prepare data matrices
        y, X = self._prepare_data(outcome, covariates, fixed_effects)

        # Create spatial lag of X
        WX = self._spatial_lag_matrix(X)
        X_full = np.hstack([X, WX])

        # Initial estimates
        beta_init = np.linalg.lstsq(X_full, y, rcond=None)[0]

        # Spatial lag of y
        Wy = self._spatial_lag(y)

        # Maximum likelihood estimation
        def log_likelihood(params):
            rho = params[0]
            beta_theta = params[1:]

            # Check rho bounds
            if abs(rho) >= 0.999:
                return 1e10

            # Residuals
            e = y - rho * Wy - X_full @ beta_theta
            sigma2 = (e.T @ e) / self.n_obs

            # Log-likelihood
            A = self.I - rho * self.W
            log_det = np.linalg.slogdet(A)[1] * self.n_times
            ll = -0.5 * self.n_obs * (np.log(2 * np.pi) + np.log(sigma2))
            ll += log_det - 0.5 * (e.T @ e) / sigma2

            return -ll

        # Optimize
        init_params = np.concatenate([[0.1], beta_init])
        result = optimize.minimize(
            log_likelihood,
            init_params,
            method="L-BFGS-B",
            bounds=[(-0.999, 0.999)] + [(None, None)] * len(beta_init),
        )

        # Extract results
        rho = result.x[0]
        beta_theta = result.x[1:]
        beta = beta_theta[: len(covariates)]
        theta = beta_theta[len(covariates) :]

        # Calculate standard errors
        se = self._calculate_standard_errors_sdm(y, X_full, rho, beta_theta)

        # Direct, indirect, and total effects (more complex for SDM)
        effects = self._calculate_spatial_effects_sdm(beta, theta, rho)

        # Test for remaining spatial autocorrelation
        residuals = y - rho * Wy - X_full @ beta_theta
        moran_i = self._morans_i(residuals)

        # Wald test for SDM vs SAR (H0: theta = 0)
        if len(theta) > 0 and len(se) > len(covariates) + 1:
            theta_se = se[len(covariates) + 1 :]
            wald_stat = np.sum((theta / theta_se) ** 2)
            wald_pvalue = 1 - chi2.cdf(wald_stat, len(theta))
        else:
            wald_stat = np.nan
            wald_pvalue = np.nan

        return {
            "model": "Spatial Durbin (SDM)",
            "outcome": outcome,
            "rho": rho,
            "rho_se": se[0] if len(se) > 0 else np.nan,
            "rho_pvalue": 2 * (1 - norm.cdf(abs(rho / se[0])))
            if len(se) > 0
            else np.nan,
            "beta": beta,
            "beta_se": se[1 : len(covariates) + 1]
            if len(se) > len(covariates)
            else np.full(len(beta), np.nan),
            "theta": theta,
            "theta_se": se[len(covariates) + 1 :]
            if len(se) > len(covariates) + 1
            else np.full(len(theta), np.nan),
            "covariates": covariates,
            "direct_effects": effects["direct"],
            "indirect_effects": effects["indirect"],
            "total_effects": effects["total"],
            "log_likelihood": -result.fun,
            "aic": 2 * (len(init_params) + result.fun),
            "bic": np.log(self.n_obs) * len(init_params) + 2 * result.fun,
            "moran_i": moran_i,
            "wald_test": wald_stat,
            "wald_pvalue": wald_pvalue,
            "n_obs": self.n_obs,
            "converged": result.success,
        }

    def _prepare_data(
        self, outcome: str, covariates: list[str], fixed_effects: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prepare data matrices for estimation.

        Args:
            outcome: Name of outcome variable
            covariates: List of covariate names
            fixed_effects: Include unit fixed effects

        Returns:
            Tuple of (y vector, X matrix)
        """
        # Sort data by unit and time for proper ordering
        df = self.data.sort_values([self.unit_col, self.time_col]).copy()

        # Extract outcome
        y = df[outcome].values

        # Extract covariates
        X = df[covariates].values

        # Add fixed effects if requested
        if fixed_effects:
            # Unit fixed effects (demean by unit)
            unit_means = df.groupby(self.unit_col)[outcome].transform("mean")
            y = y - unit_means.values

            for i, col in enumerate(covariates):
                col_means = df.groupby(self.unit_col)[col].transform("mean")
                X[:, i] = X[:, i] - col_means.values

        # Add constant if not using fixed effects
        if not fixed_effects:
            X = np.column_stack([np.ones(len(y)), X])

        return y, X

    def _spatial_lag(self, y: np.ndarray) -> np.ndarray:
        """Calculate spatial lag of a vector.

        Args:
            y: Vector to lag

        Returns:
            Spatial lag Wy
        """
        # Reshape to panel format if needed
        if len(y) == self.n_obs:
            y_panel = y.reshape(self.n_times, self.n_units).T
            Wy_panel = self.W @ y_panel
            return Wy_panel.T.flatten()
        return self.W @ y

    def _spatial_lag_matrix(self, X: np.ndarray) -> np.ndarray:
        """Calculate spatial lag of a matrix.

        Args:
            X: Matrix to lag

        Returns:
            Spatial lag WX
        """
        n_vars = X.shape[1]
        WX = np.zeros_like(X)

        for j in range(n_vars):
            WX[:, j] = self._spatial_lag(X[:, j])

        return WX

    def _calculate_spatial_effects(
        self, beta: np.ndarray, rho: float, model_type: str
    ) -> dict:
        """Calculate direct, indirect, and total effects for spatial models.

        Args:
            beta: Coefficient estimates
            rho: Spatial lag parameter
            model_type: Type of model ("lag" or "error")

        Returns:
            Dictionary with direct, indirect, and total effects
        """
        if model_type == "error":
            # SEM has no spillover effects in covariates
            return {"direct": beta, "indirect": np.zeros_like(beta), "total": beta}

        # For SAR model: (I - ρW)^(-1) * β
        A_inv = np.linalg.inv(self.I - rho * self.W)

        # Direct effects: average diagonal elements
        direct = np.mean(np.diag(A_inv)) * beta

        # Total effects: average row sum
        total = np.mean(A_inv.sum(axis=1)) * beta

        # Indirect effects: total - direct
        indirect = total - direct

        return {"direct": direct, "indirect": indirect, "total": total}

    def _calculate_spatial_effects_sdm(
        self, beta: np.ndarray, theta: np.ndarray, rho: float
    ) -> dict:
        """Calculate effects for Spatial Durbin Model.

        Args:
            beta: Direct coefficient estimates
            theta: Spatial lag coefficient estimates
            rho: Spatial lag parameter

        Returns:
            Dictionary with direct, indirect, and total effects
        """
        # For SDM: (I - ρW)^(-1) * (βI + θW)
        A_inv = np.linalg.inv(self.I - rho * self.W)

        effects = {"direct": [], "indirect": [], "total": []}

        for k in range(len(beta)):
            # Effect matrix for variable k
            S_k = A_inv @ (beta[k] * self.I + theta[k] * self.W)

            # Direct: average diagonal
            direct = np.mean(np.diag(S_k))

            # Total: average row sum
            total = np.mean(S_k.sum(axis=1))

            # Indirect: total - direct
            indirect = total - direct

            effects["direct"].append(direct)
            effects["indirect"].append(indirect)
            effects["total"].append(total)

        return {
            "direct": np.array(effects["direct"]),
            "indirect": np.array(effects["indirect"]),
            "total": np.array(effects["total"]),
        }

    def _calculate_standard_errors_sar(
        self, y: np.ndarray, X: np.ndarray, rho: float, beta: np.ndarray
    ) -> np.ndarray:
        """Calculate standard errors for SAR model.

        Uses the information matrix approach.

        Args:
            y: Outcome vector
            X: Covariate matrix
            rho: Spatial lag parameter
            beta: Coefficient estimates

        Returns:
            Standard errors for [rho, beta]
        """
        try:
            # Residuals
            Wy = self._spatial_lag(y)
            e = y - rho * Wy - X @ beta
            sigma2 = (e.T @ e) / self.n_obs

            # Information matrix
            A = self.I - rho * self.W
            A_inv = np.linalg.inv(A)

            # Build information matrix blocks
            n_params = 1 + len(beta)
            info = np.zeros((n_params, n_params))

            # d2L/drho2
            tr_term = np.trace(A_inv @ self.W @ A_inv @ self.W)
            info[0, 0] = tr_term * self.n_times + (Wy.T @ Wy) / sigma2

            # d2L/drho*dbeta
            info[0, 1:] = (Wy.T @ X) / sigma2
            info[1:, 0] = info[0, 1:]

            # d2L/dbeta2
            info[1:, 1:] = (X.T @ X) / sigma2

            # Variance-covariance matrix
            vcov = np.linalg.inv(info)

            # Standard errors
            se = np.sqrt(np.diag(vcov))

            return se

        except Exception as e:
            logger.warning(f"Could not calculate standard errors: {e}")
            return np.full(1 + len(beta), np.nan)

    def _calculate_standard_errors_sem(
        self, y: np.ndarray, X: np.ndarray, lambda_param: float, beta: np.ndarray
    ) -> np.ndarray:
        """Calculate standard errors for SEM model.

        Args:
            y: Outcome vector
            X: Covariate matrix
            lambda_param: Spatial error parameter
            beta: Coefficient estimates

        Returns:
            Standard errors for [lambda, beta]
        """
        try:
            # Similar to SAR but for error model
            A = self.I - lambda_param * self.W
            e = y - X @ beta
            u = A @ e
            sigma2 = (u.T @ u) / self.n_obs

            # Information matrix
            A_inv = np.linalg.inv(A)
            n_params = 1 + len(beta)
            info = np.zeros((n_params, n_params))

            # d2L/dlambda2
            tr_term = np.trace(A_inv @ self.W @ A_inv @ self.W)
            We = self._spatial_lag(e)
            info[0, 0] = tr_term * self.n_times + (We.T @ We) / sigma2

            # d2L/dlambda*dbeta
            info[0, 1:] = -(We.T @ X) / sigma2
            info[1:, 0] = info[0, 1:]

            # d2L/dbeta2
            info[1:, 1:] = (X.T @ A.T @ A @ X) / sigma2

            # Variance-covariance matrix
            vcov = np.linalg.inv(info)

            # Standard errors
            se = np.sqrt(np.diag(vcov))

            return se

        except Exception as e:
            logger.warning(f"Could not calculate standard errors: {e}")
            return np.full(1 + len(beta), np.nan)

    def _calculate_standard_errors_sdm(
        self, y: np.ndarray, X_full: np.ndarray, rho: float, beta_theta: np.ndarray
    ) -> np.ndarray:
        """Calculate standard errors for SDM model.

        Args:
            y: Outcome vector
            X_full: Combined covariate matrix [X, WX]
            rho: Spatial lag parameter
            beta_theta: Combined coefficient estimates

        Returns:
            Standard errors for [rho, beta, theta]
        """
        # Similar to SAR but with augmented X matrix
        return self._calculate_standard_errors_sar(y, X_full, rho, beta_theta)

    def _morans_i(self, residuals: np.ndarray) -> dict:
        """Calculate Moran's I statistic for spatial autocorrelation.

        Args:
            residuals: Residual vector

        Returns:
            Dictionary with Moran's I statistic and p-value
        """
        # Demean residuals
        e = residuals - residuals.mean()

        # Calculate Moran's I
        numerator = e.T @ self._spatial_lag(e)
        denominator = e.T @ e
        n = len(e) if len(e) <= self.n_units else self.n_units
        S0 = self.W.sum()

        moran_i = (n / S0) * (numerator / denominator)

        # Expected value and variance under null hypothesis
        E_I = -1 / (n - 1)

        # Approximate variance (simplified)
        _b2 = (e**2).sum() / n
        _b4 = (e**4).sum() / n
        S1 = 0.5 * ((self.W + self.W.T) ** 2).sum()
        S2 = (self.W.sum(axis=1) ** 2).sum()

        Var_I = (
            n * ((n**2 - 3 * n + 3) * S1 - n * S2 + 3 * S0**2)
            - (n**2 - n) * S1
            + 2 * n * S2
            - 6 * S0**2
        ) / ((n - 1) * (n - 2) * (n - 3) * S0**2)

        # Z-score and p-value
        z_score = (moran_i - E_I) / np.sqrt(Var_I) if Var_I > 0 else 0
        p_value = 2 * (1 - norm.cdf(abs(z_score)))

        return {
            "statistic": moran_i,
            "expected": E_I,
            "variance": Var_I,
            "z_score": z_score,
            "p_value": p_value,
        }

    def lagrange_multiplier_tests(
        self, outcome: str, covariates: list[str], fixed_effects: bool = True
    ) -> dict:
        """Perform Lagrange Multiplier tests for spatial dependence.

        Tests for both spatial lag and spatial error dependence.

        Args:
            outcome: Name of outcome variable
            covariates: List of covariate names
            fixed_effects: Include unit fixed effects

        Returns:
            Dictionary with LM test results
        """
        # Estimate OLS model
        y, X = self._prepare_data(outcome, covariates, fixed_effects)
        beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
        e = y - X @ beta_ols
        sigma2 = (e.T @ e) / self.n_obs

        # Spatial lag of residuals
        We = self._spatial_lag(e)

        # LM test for spatial lag
        T = np.trace((self.W + self.W.T) @ self.W)
        LM_lag = ((e.T @ self._spatial_lag(y)) / sigma2) ** 2 / (self.n_obs * T)
        p_lag = 1 - chi2.cdf(LM_lag, 1)

        # LM test for spatial error
        LM_error = ((e.T @ We) / sigma2) ** 2 / T
        p_error = 1 - chi2.cdf(LM_error, 1)

        # Robust LM tests (robust to presence of the other form)
        # Robust LM lag (robust to error)
        LM_lag_robust = ((e.T @ self._spatial_lag(y) - e.T @ We) / sigma2) ** 2 / (
            self.n_obs * T
        )
        p_lag_robust = 1 - chi2.cdf(LM_lag_robust, 1)

        # Robust LM error (robust to lag)
        LM_error_robust = (
            (e.T @ We - T * e.T @ self._spatial_lag(y) / self.n_obs) / sigma2
        ) ** 2 / (T * (1 - T / self.n_obs))
        p_error_robust = 1 - chi2.cdf(LM_error_robust, 1)

        return {
            "LM_lag": {"statistic": LM_lag, "p_value": p_lag},
            "LM_error": {"statistic": LM_error, "p_value": p_error},
            "LM_lag_robust": {"statistic": LM_lag_robust, "p_value": p_lag_robust},
            "LM_error_robust": {
                "statistic": LM_error_robust,
                "p_value": p_error_robust,
            },
        }
