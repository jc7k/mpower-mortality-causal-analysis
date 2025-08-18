#!/usr/bin/env python3
"""Debug script to identify the data type issue in event study analysis."""

import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np

def debug_event_study_data():
    """Debug the data types in event study analysis."""
    
    # Import after setting up the path
    from mpower_mortality_causal_analysis.analysis import MPOWERAnalysisPipeline
    
    # Initialize pipeline
    pipeline = MPOWERAnalysisPipeline(
        data_path="/home/user/projects/mpower-mortality-causal-analysis/data/processed/analysis_ready_data.csv",
        outcomes=['mort_lung_cancer_asr'], 
        treatment_col="ever_treated",
        treatment_year_col="treatment_cohort", 
        unit_col="country_name",
        time_col="year",
        control_vars=['gdp_pc_constant_log', 'urban_pop_pct']
    )
    
    print(f"Data types in pipeline data:")
    print(pipeline.data.dtypes)
    
    print(f"\nControl variables: {pipeline.control_vars}")
    for var in pipeline.control_vars:
        if var in pipeline.data.columns:
            print(f"{var}: {pipeline.data[var].dtype}, has NaN: {pipeline.data[var].isna().any()}")
            print(f"  Sample values: {pipeline.data[var].dropna().head().tolist()}")
        else:
            print(f"{var}: NOT FOUND!")
    
    print(f"\nOutcome variable: mort_lung_cancer_asr")
    outcome = 'mort_lung_cancer_asr'
    print(f"{outcome}: {pipeline.data[outcome].dtype}, has NaN: {pipeline.data[outcome].isna().any()}")
    print(f"  Sample values: {pipeline.data[outcome].dropna().head().tolist()}")
    
    print(f"\nUnit variable: {pipeline.unit_col}")
    print(f"{pipeline.unit_col}: {pipeline.data[pipeline.unit_col].dtype}")
    print(f"  Sample values: {pipeline.data[pipeline.unit_col].head().tolist()}")
    
    print(f"\nTime variable: {pipeline.time_col}")
    print(f"{pipeline.time_col}: {pipeline.data[pipeline.time_col].dtype}")
    print(f"  Sample values: {pipeline.data[pipeline.time_col].head().tolist()}")
    
    # Test creating event study data
    print(f"\n" + "="*50)
    print("TESTING EVENT STUDY DATA CREATION")
    print("="*50)
    
    from mpower_mortality_causal_analysis.causal_inference.utils.event_study import EventStudyAnalysis
    
    # Create event study with correct cohort column
    event_study = EventStudyAnalysis(
        data=pipeline.data,
        unit_col=pipeline.unit_col,
        time_col=pipeline.time_col,
        treatment_col=pipeline.treatment_year_col,  # Use treatment_year_col as cohort
        never_treated_value=0,
    )
    
    print(f"Event study data shape: {event_study.data.shape}")
    print(f"Event time column created: {'event_time' in event_study.data.columns}")
    print(f"Event time data type: {event_study.data['event_time'].dtype}")
    print(f"Event time unique values: {sorted(event_study.data['event_time'].unique())[:10]}")
    
    # Create event time dummies
    data_with_dummies = event_study.create_event_time_dummies(max_lag=3, max_lead=3)
    print(f"\nData with dummies shape: {data_with_dummies.shape}")
    
    # Check dummy columns
    dummy_cols = [col for col in data_with_dummies.columns if 'event_time' in col]
    print(f"Event time dummy columns: {dummy_cols}")
    
    # Check data types of variables that will be used in regression
    test_vars = pipeline.control_vars + dummy_cols
    print(f"\nData types for regression variables:")
    for var in test_vars:
        if var in data_with_dummies.columns:
            dtype = data_with_dummies[var].dtype
            has_nan = data_with_dummies[var].isna().any()
            print(f"  {var}: {dtype}, has NaN: {has_nan}")
            
            # Check if numeric
            try:
                pd.to_numeric(data_with_dummies[var])
                print(f"    -> Can convert to numeric: YES")
            except:
                print(f"    -> Can convert to numeric: NO")
                print(f"    -> Sample values: {data_with_dummies[var].unique()[:5]}")
        else:
            print(f"  {var}: NOT FOUND!")
    
    return data_with_dummies, pipeline, event_study

if __name__ == "__main__":
    data_with_dummies, pipeline, event_study = debug_event_study_data()