#!/usr/bin/env python3
"""Test script to run MPOWER analysis pipeline with debugging."""

import sys
sys.path.insert(0, 'src')

import pandas as pd
from pathlib import Path

def test_column_mapping():
    """Test the data loading and column mapping."""
    data_path = Path("/home/user/projects/mpower-mortality-causal-analysis/data/processed/analysis_ready_data.csv")
    
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        return False
        
    # Load data
    print("Loading data...")
    data = pd.read_csv(data_path)
    
    print(f"Data shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    
    # Check for expected columns
    expected_cols = {
        'country_name': 'country',  
        'treatment_cohort': 'first_high_year',
        'ever_treated': 'mpower_high_binary',
        'mort_lung_cancer_asr': 'lung_cancer_mortality_rate',
        'gdp_pc_constant_log': 'gdp_per_capita_log'
    }
    
    missing_cols = []
    for actual_col, expected_col in expected_cols.items():
        if actual_col not in data.columns:
            missing_cols.append(actual_col)
            
    if missing_cols:
        print(f"Missing columns: {missing_cols}")
        return False
        
    print("Column mapping successful!")
    return True

def test_pipeline_init():
    """Test pipeline initialization with correct column mapping."""
    try:
        from mpower_mortality_causal_analysis.analysis import MPOWERAnalysisPipeline
        
        # Use correct column mappings
        pipeline = MPOWERAnalysisPipeline(
            data_path="/home/user/projects/mpower-mortality-causal-analysis/data/processed/analysis_ready_data.csv",
            outcomes=['mort_lung_cancer_asr', 'mort_cvd_asr'], 
            treatment_col="ever_treated",
            treatment_year_col="treatment_cohort",
            unit_col="country_name",
            time_col="year",
            control_vars=['gdp_pc_constant_log', 'urban_pop_pct']
        )
        
        print("Pipeline initialized successfully!")
        print(f"Data shape: {pipeline.data.shape}")
        return pipeline
        
    except Exception as e:
        print(f"Pipeline initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("=== Testing Column Mapping ===")
    if test_column_mapping():
        print("\n=== Testing Pipeline Initialization ===")
        pipeline = test_pipeline_init()
        
        if pipeline:
            print("\n=== Running Descriptive Analysis ===")
            try:
                results = pipeline.run_descriptive_analysis()
                print("Descriptive analysis completed successfully!")
            except Exception as e:
                print(f"Descriptive analysis failed: {e}")
                import traceback
                traceback.print_exc()