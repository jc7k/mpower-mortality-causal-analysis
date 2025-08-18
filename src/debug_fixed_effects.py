#!/usr/bin/env python3
"""Debug script to identify issues with fixed effects data preparation."""

import sys

sys.path.insert(0, "src")

import numpy as np
import pandas as pd


def debug_fixed_effects():
    """Debug the fixed effects preparation in event study."""
    # Import after setting up the path
    from mpower_mortality_causal_analysis.analysis import MPOWERAnalysisPipeline
    from mpower_mortality_causal_analysis.causal_inference.utils.event_study import (
        EventStudyAnalysis,
    )

    # Initialize pipeline
    pipeline = MPOWERAnalysisPipeline(
        data_path="/home/user/projects/mpower-mortality-causal-analysis/data/processed/analysis_ready_data.csv",
        outcomes=["mort_lung_cancer_asr"],
        treatment_col="ever_treated",
        treatment_year_col="treatment_cohort",
        unit_col="country_name",
        time_col="year",
        control_vars=["gdp_pc_constant_log", "urban_pop_pct"],
    )

    # Create event study
    event_study = EventStudyAnalysis(
        data=pipeline.data,
        unit_col=pipeline.unit_col,
        time_col=pipeline.time_col,
        treatment_col=pipeline.treatment_year_col,
        never_treated_value=0,
    )

    # Create event time dummies
    data_with_dummies = event_study.create_event_time_dummies(max_lag=3, max_lead=3)

    print("=" * 60)
    print("DEBUGGING FIXED EFFECTS DATA PREPARATION")
    print("=" * 60)

    # Simulate the exact steps in _estimate_fixed_effects
    outcome = "mort_lung_cancer_asr"
    rhs_vars = [
        "gdp_pc_constant_log",
        "urban_pop_pct",
        "event_time_lead_3",
        "event_time_lead_2",
        "event_time_0",
        "event_time_lag_1",
        "event_time_lag_2",
        "event_time_lag_3",
    ]
    cluster_var = "country_name"

    print(f"Outcome: {outcome}")
    print(f"RHS vars: {rhs_vars}")
    print(f"Cluster var: {cluster_var}")

    # Check the data before processing
    print(f"\nData shape: {data_with_dummies.shape}")
    print(f"Unique countries: {data_with_dummies[event_study.unit_col].nunique()}")
    print(f"Unique years: {data_with_dummies[event_study.time_col].nunique()}")

    # Create unit dummies
    print("\nCreating unit dummies...")
    unit_dummies = pd.get_dummies(
        data_with_dummies[event_study.unit_col], prefix="unit", drop_first=True
    )
    print(f"Unit dummies shape: {unit_dummies.shape}")
    print(f"Unit dummies dtypes: {unit_dummies.dtypes.unique()}")

    # Create time dummies
    print("\nCreating time dummies...")
    time_dummies = pd.get_dummies(
        data_with_dummies[event_study.time_col], prefix="time", drop_first=True
    )
    print(f"Time dummies shape: {time_dummies.shape}")
    print(f"Time dummies dtypes: {time_dummies.dtypes.unique()}")

    # Check individual RHS variables
    print("\nChecking RHS variables:")
    for var in rhs_vars:
        if var in data_with_dummies.columns:
            dtype = data_with_dummies[var].dtype
            has_nan = data_with_dummies[var].isna().any()
            unique_vals = data_with_dummies[var].nunique()
            print(f"  {var}: {dtype}, NaN: {has_nan}, unique: {unique_vals}")

            # Check if can convert to numeric
            try:
                numeric_vals = pd.to_numeric(data_with_dummies[var], errors="coerce")
                converted_nans = (
                    numeric_vals.isna().sum() - data_with_dummies[var].isna().sum()
                )
                print(f"    -> Numeric conversion: OK, new NaNs: {converted_nans}")
            except Exception as e:
                print(f"    -> Numeric conversion: FAILED - {e}")
        else:
            print(f"  {var}: NOT FOUND!")

    # Combine all variables (exactly as in the code)
    print("\nCombining variables...")
    try:
        # First combine RHS vars
        rhs_data = data_with_dummies[rhs_vars]
        print(f"RHS data shape: {rhs_data.shape}")
        print(f"RHS data dtypes:\n{rhs_data.dtypes}")

        # Combine with unit dummies
        print("\nCombining with unit dummies...")
        combined1 = pd.concat([rhs_data, unit_dummies], axis=1)
        print(f"After unit dummies: {combined1.shape}")
        print(f"After unit dummies dtypes: {combined1.dtypes.value_counts()}")

        # Combine with time dummies
        print("\nCombining with time dummies...")
        X = pd.concat([combined1, time_dummies], axis=1)
        print(f"Final X shape: {X.shape}")
        print(f"Final X dtypes: {X.dtypes.value_counts()}")

        # Add constant
        import statsmodels.api as sm

        print("\nAdding constant...")
        X = sm.add_constant(X)
        print(f"X with constant shape: {X.shape}")
        print(f"X with constant dtypes: {X.dtypes.value_counts()}")

        # Get outcome
        y = data_with_dummies[outcome]
        print("\nY variable:")
        print(f"Y shape: {y.shape}")
        print(f"Y dtype: {y.dtype}")
        print(f"Y has NaN: {y.isna().any()}")

        # Check for any remaining issues
        print("\nFinal data check:")
        print(f"X has any object columns: {(X.dtypes == 'object').any()}")
        print(f"X has any NaN: {X.isna().any().any()}")
        print(f"Y has any NaN: {y.isna().any()}")

        # Try manual conversion to numpy
        print("\nTesting numpy conversion:")
        try:
            X_np = np.asarray(X)
            print(f"X numpy conversion: OK, shape {X_np.shape}, dtype {X_np.dtype}")
        except Exception as e:
            print(f"X numpy conversion: FAILED - {e}")
            # Check which columns are problematic
            for col in X.columns:
                try:
                    col_np = np.asarray(X[col])
                    print(f"  {col}: OK")
                except Exception as ce:
                    print(f"  {col}: FAILED - {ce}")
                    print(f"    Sample values: {X[col].head().tolist()}")

        try:
            y_np = np.asarray(y)
            print(f"Y numpy conversion: OK, shape {y_np.shape}, dtype {y_np.dtype}")
        except Exception as e:
            print(f"Y numpy conversion: FAILED - {e}")

    except Exception as e:
        print(f"Error during combination: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    debug_fixed_effects()
