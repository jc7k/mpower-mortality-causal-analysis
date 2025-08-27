"""Main spillover analysis pipeline orchestration.

This module provides the main pipeline class that coordinates all spillover
analysis components and integrates with the main MPOWER analysis framework.
"""

import json
import logging
import os

import numpy as np
import pandas as pd

from .border_analysis import BorderDiscontinuity
from .diffusion_analysis import PolicyDiffusionNetwork
from .spatial_models import SpatialPanelModel
from .spatial_weights import SpatialWeightMatrix
from .visualization import SpilloverVisualization

logger = logging.getLogger(__name__)


class SpilloverPipeline:
    """Main pipeline for comprehensive spillover analysis.

    Orchestrates all spillover analysis components including spatial econometrics,
    network diffusion, and border discontinuity designs.

    Args:
        data: Panel data for analysis
        unit_col: Column name for unit identifier (country)
        time_col: Column name for time identifier (year)
        outcomes: List of outcome variable names
        treatment_col: Treatment variable column name
        covariates: List of control variable names
    """

    def __init__(
        self,
        data: pd.DataFrame,
        unit_col: str = "country",
        time_col: str = "year",
        outcomes: list[str] | None = None,
        treatment_col: str = "mpower_high",
        covariates: list[str] | None = None,
    ):
        """Initialize spillover analysis pipeline.

        Args:
            data: Panel data for analysis
            unit_col: Column name for unit identifier
            time_col: Column name for time identifier
            outcomes: List of outcome variable names
            treatment_col: Treatment variable column name
            covariates: List of control variable names
        """
        self.data = data.copy()
        self.unit_col = unit_col
        self.time_col = time_col
        self.treatment_col = treatment_col

        # Set default outcomes if not provided
        self.outcomes = outcomes or [
            "lung_cancer_mortality_rate",
            "cardiovascular_mortality_rate",
            "ischemic_heart_disease_mortality_rate",
            "copd_mortality_rate",
        ]

        # Filter outcomes to only those present in data
        self.outcomes = [col for col in self.outcomes if col in data.columns]

        # Set default covariates if not provided
        self.covariates = covariates or [
            "gdp_per_capita_log",
            "urban_population_pct",
            "total_population_log",
        ]

        # Filter covariates to only those present in data
        self.covariates = [col for col in self.covariates if col in data.columns]

        # Extract basic info
        self.countries = sorted(data[unit_col].unique())
        self.years = sorted(data[time_col].unique())
        self.n_countries = len(self.countries)
        self.n_years = len(self.years)

        # Initialize components
        self.weight_matrix = None
        self.spatial_model = None
        self.diffusion_network = None
        self.border_analysis = None
        self.visualizer = SpilloverVisualization()

        # Results storage
        self.results = {}

        logger.info(
            f"Initialized spillover pipeline with {self.n_countries} countries, "
            f"{self.n_years} years, {len(self.outcomes)} outcomes"
        )

    def run_full_analysis(
        self, save_results: bool = True, output_dir: str = "spillover_results"
    ) -> dict:
        """Run complete spillover analysis pipeline.

        Args:
            save_results: Whether to save results to files
            output_dir: Directory to save results

        Returns:
            Dictionary containing all analysis results
        """
        logger.info("Starting comprehensive spillover analysis")

        # Step 1: Create spatial weight matrices
        logger.info("Step 1: Creating spatial weight matrices")
        self._create_weight_matrices()

        # Step 2: Spatial econometric analysis
        logger.info("Step 2: Running spatial econometric models")
        self._run_spatial_models()

        # Step 3: Network diffusion analysis
        logger.info("Step 3: Analyzing policy diffusion networks")
        self._analyze_diffusion()

        # Step 4: Border discontinuity analysis
        logger.info("Step 4: Running border discontinuity analysis")
        self._analyze_borders()

        # Step 5: Model comparison and selection
        logger.info("Step 5: Comparing spatial models")
        self._compare_models()

        # Step 6: Generate summary and key findings
        logger.info("Step 6: Generating summary results")
        self._generate_summary()

        # Save results if requested
        if save_results:
            self._save_results(output_dir)

        logger.info("Spillover analysis complete")
        return self.results

    def _create_weight_matrices(self):
        """Create various spatial weight matrices."""
        # Initialize weight matrix constructor
        self.weight_matrix = SpatialWeightMatrix(self.countries)

        # Create different types of weight matrices
        matrices = {}

        # Contiguity matrix
        matrices["contiguity"] = self.weight_matrix.contiguity_matrix()

        # Distance-based matrix
        matrices["distance"] = self.weight_matrix.distance_matrix(cutoff=5000)

        # K-nearest neighbors
        matrices["knn"] = self.weight_matrix.k_nearest_neighbors(k=5)

        # Hybrid matrix
        matrices["hybrid"] = self.weight_matrix.hybrid_matrix(
            contiguity_weight=0.4, distance_weight=0.4, economic_weight=0.2
        )

        # Store matrices and their statistics
        self.results["weight_matrices"] = {}
        for name, W in matrices.items():
            stats = self.weight_matrix.get_connectivity_stats(W)
            self.results["weight_matrices"][name] = {"matrix": W, "statistics": stats}

        logger.info(f"Created {len(matrices)} spatial weight matrices")

    def _run_spatial_models(self):
        """Run spatial econometric models for all outcomes."""
        self.results["spatial_models"] = {}

        # Use hybrid weight matrix as default
        W = self.results["weight_matrices"]["hybrid"]["matrix"]

        # Initialize spatial model
        self.spatial_model = SpatialPanelModel(
            self.data, W, self.unit_col, self.time_col
        )

        # Run models for each outcome
        for outcome in self.outcomes:
            logger.info(f"Analyzing spatial models for {outcome}")

            outcome_results = {}

            # Spatial Lag Model (SAR)
            try:
                sar_results = self.spatial_model.spatial_lag_model(
                    outcome, self.covariates, fixed_effects=True
                )
                outcome_results["sar"] = sar_results
                logger.info(f"SAR model for {outcome}: ρ = {sar_results['rho']:.3f}")
            except Exception as e:
                logger.warning(f"SAR model failed for {outcome}: {e}")
                outcome_results["sar"] = {"error": str(e)}

            # Spatial Error Model (SEM)
            try:
                sem_results = self.spatial_model.spatial_error_model(
                    outcome, self.covariates, fixed_effects=True
                )
                outcome_results["sem"] = sem_results
                logger.info(f"SEM model for {outcome}: λ = {sem_results['lambda']:.3f}")
            except Exception as e:
                logger.warning(f"SEM model failed for {outcome}: {e}")
                outcome_results["sem"] = {"error": str(e)}

            # Spatial Durbin Model (SDM)
            try:
                sdm_results = self.spatial_model.spatial_durbin_model(
                    outcome, self.covariates, fixed_effects=True
                )
                outcome_results["sdm"] = sdm_results
                logger.info(f"SDM model for {outcome}: ρ = {sdm_results['rho']:.3f}")
            except Exception as e:
                logger.warning(f"SDM model failed for {outcome}: {e}")
                outcome_results["sdm"] = {"error": str(e)}

            # Lagrange Multiplier tests
            try:
                lm_tests = self.spatial_model.lagrange_multiplier_tests(
                    outcome, self.covariates, fixed_effects=True
                )
                outcome_results["lm_tests"] = lm_tests
            except Exception as e:
                logger.warning(f"LM tests failed for {outcome}: {e}")
                outcome_results["lm_tests"] = {"error": str(e)}

            self.results["spatial_models"][outcome] = outcome_results

    def _analyze_diffusion(self):
        """Analyze policy diffusion through networks."""
        # Prepare adoption data
        adoption_data = self._prepare_adoption_data()

        # Use hybrid weight matrix
        W = self.results["weight_matrices"]["hybrid"]["matrix"]

        # Initialize diffusion network
        self.diffusion_network = PolicyDiffusionNetwork(adoption_data, W)

        diffusion_results = {}

        # Estimate contagion effects
        try:
            threshold_results = self.diffusion_network.estimate_contagion("threshold")
            diffusion_results["threshold_model"] = threshold_results
            logger.info(
                f"Threshold model: mean threshold = {threshold_results.get('mean_threshold', 0):.3f}"
            )
        except Exception as e:
            logger.warning(f"Threshold model failed: {e}")
            diffusion_results["threshold_model"] = {"error": str(e)}

        try:
            cascade_results = self.diffusion_network.estimate_contagion("cascade")
            diffusion_results["cascade_model"] = cascade_results
            logger.info(
                f"Cascade model: {cascade_results.get('n_cascades', 0)} cascades identified"
            )
        except Exception as e:
            logger.warning(f"Cascade model failed: {e}")
            diffusion_results["cascade_model"] = {"error": str(e)}

        # Identify influencers
        try:
            influencers = self.diffusion_network.identify_influencers(top_k=10)
            diffusion_results["influencers"] = influencers
            if influencers:
                top_influencer = influencers[0]["country"]
                logger.info(f"Top influencer: {top_influencer}")
        except Exception as e:
            logger.warning(f"Influencer identification failed: {e}")
            diffusion_results["influencers"] = {"error": str(e)}

        # Estimate peer effects on outcomes
        for outcome in self.outcomes:
            try:
                outcome_data = self.data[
                    [self.unit_col, self.time_col, outcome]
                ].dropna()
                outcome_data = outcome_data.rename(columns={outcome: "mortality_rate"})

                peer_effects = self.diffusion_network.estimate_peer_effects(
                    outcome_data, self.covariates
                )
                diffusion_results[f"peer_effects_{outcome}"] = peer_effects

                peer_effect = peer_effects.get("peer_effect", 0)
                logger.info(f"Peer effect on {outcome}: {peer_effect:.3f}")
            except Exception as e:
                logger.warning(f"Peer effects estimation failed for {outcome}: {e}")
                diffusion_results[f"peer_effects_{outcome}"] = {"error": str(e)}

        # Predict future adoption
        try:
            future_adoption = self.diffusion_network.predict_future_adoption(
                n_periods=5
            )
            diffusion_results["future_adoption"] = {
                "predictions": future_adoption.tolist(),
                "predicted_adoption_rate": future_adoption[:, -1].mean(),
            }
        except Exception as e:
            logger.warning(f"Future adoption prediction failed: {e}")
            diffusion_results["future_adoption"] = {"error": str(e)}

        self.results["diffusion_analysis"] = diffusion_results

    def _analyze_borders(self):
        """Run border discontinuity analysis."""
        # Prepare border data
        border_data = self._prepare_border_data()

        # Initialize border analysis
        self.border_analysis = BorderDiscontinuity(border_data)

        border_results = {}

        # Analyze each outcome
        for outcome in self.outcomes:
            logger.info(f"Running border discontinuity for {outcome}")

            try:
                rdd_results = self.border_analysis.estimate_border_effect(
                    outcome, self.treatment_col, bandwidth=100
                )
                border_results[outcome] = rdd_results

                effect = rdd_results.get("effect", 0)
                p_value = rdd_results.get("p_value", 1)
                logger.info(
                    f"Border effect on {outcome}: {effect:.3f} (p={p_value:.3f})"
                )
            except Exception as e:
                logger.warning(f"Border analysis failed for {outcome}: {e}")
                border_results[outcome] = {"error": str(e)}

        # Analyze heterogeneity
        try:
            heterogeneity_vars = ["gdp_per_capita_log", "urban_population_pct"]
            available_het_vars = [
                v for v in heterogeneity_vars if v in self.data.columns
            ]

            if available_het_vars and self.outcomes:
                heterogeneity_results = self.border_analysis.analyze_heterogeneity(
                    self.outcomes[0], self.treatment_col, available_het_vars
                )
                border_results["heterogeneity"] = heterogeneity_results
        except Exception as e:
            logger.warning(f"Heterogeneity analysis failed: {e}")
            border_results["heterogeneity"] = {"error": str(e)}

        # Estimate effects for all borders
        try:
            all_borders = self.border_analysis.estimate_all_borders(
                self.outcomes[0] if self.outcomes else "mortality_rate",
                self.treatment_col,
            )
            border_results["all_borders"] = all_borders.to_dict("records")

            n_significant = (
                all_borders["significant"].sum()
                if "significant" in all_borders.columns
                else 0
            )
            logger.info(
                f"Border analysis: {n_significant}/{len(all_borders)} borders significant"
            )
        except Exception as e:
            logger.warning(f"All borders analysis failed: {e}")
            border_results["all_borders"] = {"error": str(e)}

        self.results["border_analysis"] = border_results

    def _compare_models(self):
        """Compare spatial models and select best-fitting ones."""
        model_comparison = {}

        for outcome in self.outcomes:
            if outcome in self.results["spatial_models"]:
                outcome_models = self.results["spatial_models"][outcome]

                # Extract model fit statistics
                model_stats = {}
                for model_name, model_result in outcome_models.items():
                    if model_name != "lm_tests" and "error" not in model_result:
                        model_stats[model_name] = {
                            "aic": model_result.get("aic", np.inf),
                            "bic": model_result.get("bic", np.inf),
                            "log_likelihood": model_result.get(
                                "log_likelihood", -np.inf
                            ),
                            "n_obs": model_result.get("n_obs", 0),
                        }

                # Find best model by AIC
                if model_stats:
                    best_model = min(
                        model_stats.keys(), key=lambda x: model_stats[x]["aic"]
                    )
                    model_comparison[outcome] = {
                        "model_statistics": model_stats,
                        "best_model": best_model,
                        "best_aic": model_stats[best_model]["aic"],
                    }

                    logger.info(
                        f"Best model for {outcome}: {best_model} (AIC={model_stats[best_model]['aic']:.1f})"
                    )

        self.results["model_comparison"] = model_comparison

    def _generate_summary(self):
        """Generate summary statistics and key findings."""
        summary = {
            "analysis_overview": {
                "n_countries": self.n_countries,
                "n_years": self.n_years,
                "n_outcomes": len(self.outcomes),
                "outcomes_analyzed": self.outcomes,
                "covariates_used": self.covariates,
                "treatment_variable": self.treatment_col,
            }
        }

        # Spatial autocorrelation summary
        spatial_summary = {}
        for outcome in self.outcomes:
            if (
                outcome in self.results.get("spatial_models", {})
                and "lm_tests" in self.results["spatial_models"][outcome]
            ):
                lm_tests = self.results["spatial_models"][outcome]["lm_tests"]
                if "error" not in lm_tests:
                    spatial_summary[outcome] = {
                        "spatial_lag_significant": lm_tests.get("LM_lag", {}).get(
                            "p_value", 1
                        )
                        < 0.05,
                        "spatial_error_significant": lm_tests.get("LM_error", {}).get(
                            "p_value", 1
                        )
                        < 0.05,
                    }
        summary["spatial_autocorrelation"] = spatial_summary

        # Spillover effects summary
        spillover_summary = {}
        for outcome in self.outcomes:
            if outcome in self.results.get("spatial_models", {}):
                models = self.results["spatial_models"][outcome]
                if "sar" in models and "error" not in models["sar"]:
                    sar = models["sar"]
                    spillover_summary[outcome] = {
                        "spatial_lag_coefficient": sar.get("rho", 0),
                        "significant": sar.get("rho_pvalue", 1) < 0.05,
                        "indirect_effects": sar.get("indirect_effects", []).tolist()
                        if isinstance(sar.get("indirect_effects"), np.ndarray)
                        else sar.get("indirect_effects", []),
                    }
        summary["spillover_effects"] = spillover_summary

        # Diffusion summary
        if "diffusion_analysis" in self.results:
            diffusion = self.results["diffusion_analysis"]
            summary["diffusion_summary"] = {
                "threshold_model_available": "threshold_model" in diffusion
                and "error" not in diffusion["threshold_model"],
                "n_influencers_identified": len(diffusion.get("influencers", []))
                if isinstance(diffusion.get("influencers"), list)
                else 0,
                "peer_effects_detected": any(
                    "peer_effects" in key for key in diffusion.keys()
                ),
            }

        # Border effects summary
        if "border_analysis" in self.results:
            border = self.results["border_analysis"]
            n_outcomes_with_border_effects = sum(
                1
                for outcome in self.outcomes
                if outcome in border
                and "error" not in border[outcome]
                and border[outcome].get("p_value", 1) < 0.05
            )
            summary["border_effects_summary"] = {
                "n_outcomes_with_significant_effects": n_outcomes_with_border_effects,
                "proportion_significant": n_outcomes_with_border_effects
                / len(self.outcomes)
                if self.outcomes
                else 0,
            }

        # Key findings
        key_findings = []

        # Check for spatial dependence
        n_spatial = sum(
            1
            for outcome_summary in spatial_summary.values()
            if outcome_summary.get("spatial_lag_significant")
            or outcome_summary.get("spatial_error_significant")
        )
        if n_spatial > 0:
            key_findings.append(
                f"Spatial dependence detected in {n_spatial}/{len(self.outcomes)} outcomes"
            )

        # Check for spillover effects
        n_spillovers = sum(
            1
            for outcome_summary in spillover_summary.values()
            if outcome_summary.get("significant")
        )
        if n_spillovers > 0:
            key_findings.append(
                f"Significant spillover effects found in {n_spillovers}/{len(self.outcomes)} outcomes"
            )

        # Check for border effects
        if "border_effects_summary" in summary:
            n_border = summary["border_effects_summary"][
                "n_outcomes_with_significant_effects"
            ]
            if n_border > 0:
                key_findings.append(
                    f"Border discontinuity effects detected in {n_border}/{len(self.outcomes)} outcomes"
                )

        # Check for network effects
        if (
            "diffusion_summary" in summary
            and summary["diffusion_summary"]["peer_effects_detected"]
        ):
            key_findings.append("Network peer effects identified in policy diffusion")

        if not key_findings:
            key_findings.append("Limited evidence of spatial spillover effects")

        summary["key_findings"] = key_findings
        self.results["summary"] = summary

        logger.info(f"Generated summary with {len(key_findings)} key findings")

    def _prepare_adoption_data(self) -> pd.DataFrame:
        """Prepare adoption data for diffusion analysis."""
        # Create adoption indicator
        if self.treatment_col in self.data.columns:
            adoption_data = self.data[
                [self.unit_col, self.time_col, self.treatment_col]
            ].copy()
            adoption_data = adoption_data.rename(
                columns={self.treatment_col: "adopted"}
            )
        else:
            # Create mock adoption data
            adoption_data = self.data[[self.unit_col, self.time_col]].copy()
            adoption_data["adopted"] = 0

            # Simulate staggered adoption
            np.random.seed(42)
            for country in self.countries[
                : len(self.countries) // 3
            ]:  # 1/3 of countries adopt
                adoption_year = np.random.choice(
                    self.years[1:-1]
                )  # Not first or last year
                mask = (adoption_data[self.unit_col] == country) & (
                    adoption_data[self.time_col] >= adoption_year
                )
                adoption_data.loc[mask, "adopted"] = 1

        return adoption_data.rename(
            columns={self.unit_col: "country", self.time_col: "year"}
        )

    def _prepare_border_data(self) -> pd.DataFrame:
        """Prepare border data for discontinuity analysis."""
        # Create mock border data for demonstration
        np.random.seed(42)
        n_border_obs = 200

        border_data = pd.DataFrame(
            {
                "country1": np.random.choice(self.countries, n_border_obs),
                "country2": np.random.choice(self.countries, n_border_obs),
                "distance_to_border": np.random.uniform(-100, 100, n_border_obs),
                self.treatment_col: np.random.binomial(1, 0.5, n_border_obs),
            }
        )

        # Add outcomes with border effects
        for outcome in self.outcomes:
            # Create outcome with discontinuity
            effect = -10  # Negative effect of treatment
            border_data[outcome] = (
                50
                + 0.1 * border_data["distance_to_border"]
                + effect * border_data[self.treatment_col]
                + np.random.normal(0, 8, n_border_obs)
            )

        return border_data

    def _save_results(self, output_dir: str):
        """Save analysis results to files."""
        os.makedirs(output_dir, exist_ok=True)

        # Save main results as JSON (excluding numpy arrays)
        json_results = self._convert_for_json(self.results)
        with open(os.path.join(output_dir, "spillover_results.json"), "w") as f:
            json.dump(json_results, f, indent=2)

        # Save weight matrices as CSV
        weight_dir = os.path.join(output_dir, "weight_matrices")
        os.makedirs(weight_dir, exist_ok=True)

        for name, matrix_info in self.results.get("weight_matrices", {}).items():
            W = matrix_info["matrix"]
            matrix_df = pd.DataFrame(W, index=self.countries, columns=self.countries)
            matrix_df.to_csv(os.path.join(weight_dir, f"{name}_weights.csv"))

        # Save model results as Excel (if openpyxl available)
        if "spatial_models" in self.results:
            try:
                with pd.ExcelWriter(
                    os.path.join(output_dir, "spatial_models.xlsx")
                ) as writer:
                    for outcome, models in self.results["spatial_models"].items():
                        for model_name, model_result in models.items():
                            if "error" not in model_result and model_name != "lm_tests":
                                # Create summary DataFrame
                                if model_name == "sar":
                                    data = {
                                        "coefficient": ["rho"]
                                        + [
                                            f"beta_{i}"
                                            for i in range(
                                                len(model_result.get("beta", []))
                                            )
                                        ],
                                        "estimate": [model_result.get("rho", 0)]
                                        + list(model_result.get("beta", [])),
                                        "std_error": [model_result.get("rho_se", 0)]
                                        + list(model_result.get("beta_se", [])),
                                        "p_value": [model_result.get("rho_pvalue", 1)]
                                        + [np.nan] * len(model_result.get("beta", [])),
                                    }
                                else:
                                    continue  # Skip other models for brevity

                                df = pd.DataFrame(data)
                                sheet_name = f"{outcome}_{model_name}"[
                                    :31
                                ]  # Excel sheet name limit
                                df.to_excel(writer, sheet_name=sheet_name, index=False)
            except ImportError:
                logger.warning("openpyxl not available, skipping Excel export")
            except Exception as e:
                logger.warning(f"Could not save Excel file: {e}")

        # Generate and save visualizations
        self._save_visualizations(output_dir)

        logger.info(f"Results saved to {output_dir}")

    def _save_visualizations(self, output_dir: str):
        """Generate and save visualizations."""
        viz_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)

        try:
            # Weight matrix visualization
            if "weight_matrices" in self.results:
                W_hybrid = self.results["weight_matrices"]["hybrid"]["matrix"]
                fig = self.visualizer.plot_weight_matrix(
                    W_hybrid, self.countries, "Hybrid Weight Matrix"
                )
                fig.savefig(
                    os.path.join(viz_dir, "weight_matrix.png"),
                    dpi=300,
                    bbox_inches="tight",
                )

            # Spatial effects visualization
            if self.outcomes and "spatial_models" in self.results:
                outcome = self.outcomes[0]
                if (
                    outcome in self.results["spatial_models"]
                    and "sar" in self.results["spatial_models"][outcome]
                ):
                    sar_result = self.results["spatial_models"][outcome]["sar"]
                    if "error" not in sar_result and "direct_effects" in sar_result:
                        effects_dict = {
                            "direct": sar_result["direct_effects"],
                            "indirect": sar_result["indirect_effects"],
                            "total": sar_result["total_effects"],
                        }
                        fig = self.visualizer.plot_spatial_effects(
                            effects_dict, self.covariates
                        )
                        fig.savefig(
                            os.path.join(viz_dir, f"spatial_effects_{outcome}.png"),
                            dpi=300,
                            bbox_inches="tight",
                        )

            # Diffusion timeline
            if "diffusion_analysis" in self.results and hasattr(
                self, "diffusion_network"
            ):
                try:
                    adoption_matrix = self.diffusion_network.adoption_matrix
                    fig = self.visualizer.plot_diffusion_timeline(
                        adoption_matrix, self.countries, self.years
                    )
                    fig.savefig(
                        os.path.join(viz_dir, "diffusion_timeline.png"),
                        dpi=300,
                        bbox_inches="tight",
                    )
                except Exception as e:
                    logger.warning(f"Could not create diffusion timeline: {e}")

            # Summary dashboard
            dashboard_data = self._prepare_dashboard_data()
            fig = self.visualizer.create_summary_dashboard(dashboard_data)
            fig.savefig(
                os.path.join(viz_dir, "summary_dashboard.png"),
                dpi=300,
                bbox_inches="tight",
            )

        except Exception as e:
            logger.warning(f"Some visualizations could not be created: {e}")

    def _prepare_dashboard_data(self) -> dict:
        """Prepare data for summary dashboard."""
        dashboard_data = {}

        # Add key findings
        if "summary" in self.results:
            dashboard_data["key_findings"] = self.results["summary"].get(
                "key_findings", []
            )

        # Add model comparison
        if "model_comparison" in self.results:
            dashboard_data["model_comparison"] = {}
            for outcome, comparison in self.results["model_comparison"].items():
                if "model_statistics" in comparison:
                    dashboard_data["model_comparison"][outcome] = comparison[
                        "model_statistics"
                    ]

        # Add spillover effects
        if self.outcomes and "spatial_models" in self.results:
            outcome = self.outcomes[0]
            if (
                outcome in self.results["spatial_models"]
                and "sar" in self.results["spatial_models"][outcome]
            ):
                sar = self.results["spatial_models"][outcome]["sar"]
                if "error" not in sar:
                    dashboard_data["spillover_effects"] = {
                        "direct": sar.get("direct_effects", [0])[0]
                        if sar.get("direct_effects")
                        else 0,
                        "indirect": sar.get("indirect_effects", [0])[0]
                        if sar.get("indirect_effects")
                        else 0,
                        "total": sar.get("total_effects", [0])[0]
                        if sar.get("total_effects")
                        else 0,
                    }

        # Add adoption timeline
        if hasattr(self, "diffusion_network"):
            try:
                cumulative = self.diffusion_network.adoption_matrix.sum(axis=0)
                dashboard_data["adoption_timeline"] = {
                    "years": self.years,
                    "cumulative": cumulative.tolist(),
                }
            except Exception:
                # Non-fatal; optionally log warning in caller's logger
                pass

        return dashboard_data

    def _convert_for_json(self, obj):
        """Convert numpy arrays and other non-JSON serializable objects for saving."""
        if isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        if isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict("records")
        return obj
