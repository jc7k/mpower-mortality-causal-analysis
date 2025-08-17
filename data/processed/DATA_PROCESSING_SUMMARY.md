# MPOWER Causal Analysis - Data Processing Summary

**Date**: 2025-01-17
**Script**: `scripts/data_processing.py`
**Analyst**: Claude Code Assistant

## Overview

This document summarizes the data processing pipeline that created the analysis-ready dataset for studying the causal impact of WHO MPOWER tobacco control policies on mortality outcomes using staggered difference-in-differences methods.

## Data Sources

### 1. WHO MPOWER Policy Data (`data/raw/mpower_gho/`)
- **Source**: WHO Global Health Observatory (GHO)
- **Coverage**: 195 countries, 2007-2022 (biennial surveys)
- **Variables**: Six tobacco control policy implementation scores
  - M: Monitor tobacco use and prevention policies (0-4 scale)
  - P: Protect people from tobacco smoke (0-4 scale)
  - O: Offer help to quit tobacco use (0-4 scale)
  - W: Warn about dangers of tobacco (0-4 scale)
  - E: Enforce bans on tobacco advertising/promotion (0-4 scale)
  - R: Raise taxes on tobacco (0-5 scale)
- **Total Score**: 0-29 points (sum of all components)

### 2. IHME Global Burden of Disease Data (`data/raw/gbd/`)
- **Source**: Institute for Health Metrics and Evaluation (IHME)
- **Coverage**: 204 countries, 2008-2019 (annual)
- **Variables**: Age-standardized mortality rates (deaths per 100,000)
  - Lung cancer mortality (`mort_lung_cancer_asr`)
  - Cardiovascular disease mortality (`mort_cvd_asr`)
  - Ischemic heart disease mortality (`mort_ihd_asr`)
  - COPD mortality (`mort_copd_asr`)
- **Unit**: Both sexes, age-standardized rates

### 3. World Bank World Development Indicators (`data/raw/worldbank/`)
- **Source**: World Bank WDI
- **Coverage**: 265 countries, 2008-2019 (annual)
- **Variables**: Economic and demographic controls
  - GDP per capita, constant 2015 USD (`gdp_pc_constant`)
  - Urban population percentage (`urban_pop_pct`)
  - Total population (`population_total`)
  - Education expenditure % of GDP (`edu_exp_pct_gdp`)

## Data Processing Steps

### Step 1: Individual Dataset Cleaning

**MPOWER Data Transformation:**
- Filtered to numeric policy scores only
- Pivoted from long format (indicator-level) to wide format (country-year panel)
- Created total MPOWER score as primary treatment intensity measure
- Standardized country names and added temporary ISO3 codes

**GBD Data Transformation:**
- Filtered to tobacco-related mortality causes
- Selected age-standardized rates for both sexes combined
- Focused on four key outcomes: lung cancer, CVD, IHD, COPD mortality
- Reshaped from cause-specific rows to outcome columns

**World Bank Data Transformation:**
- Combined manual and additional WDI downloads
- Melted from wide format (years as columns) to long format
- Selected key economic/demographic control variables
- Created log-transformed GDP per capita variable

### Step 2: Country Name Standardization

**Challenge**: Different naming conventions across datasets
- WHO: "United States of America"
- IHME: "United States"
- World Bank: "United States"

**Solution**: Created mapping dictionary for major discrepancies and standardized to common country names for merging.

### Step 3: Dataset Integration

**Merge Strategy**: Left join starting with MPOWER data (treatment)
1. MPOWER ← GBD (mortality outcomes)
2. Combined ← World Bank (control variables)

**Time Period**: Restricted to 2008-2018 overlap period
- MPOWER: Biennial surveys (2008, 2010, 2012, 2014, 2016, 2018)
- GBD: Annual data available for all years
- World Bank: Annual data available for all years

### Step 4: Treatment Cohort Creation

**Treatment Definition**: Countries achieving sustained high MPOWER implementation
- **Threshold**: Total MPOWER score ≥ 25 (out of 29 possible)
- **Sustainability**: Must maintain high score for ≥ 2 consecutive survey periods
- **Cohort Assignment**: Year of first achieving sustained high implementation

**Cohort Distribution**:
- Never treated (control): 151 countries (906 observations)
- 2009 cohort: 8 countries (48 observations)
- 2011 cohort: 6 countries (36 observations)
- 2013 cohort: 9 countries (54 observations)
- 2015 cohort: 8 countries (48 observations)
- 2017 cohort: 13 countries (78 observations)

### Step 5: Panel Data Preparation

**Variable Creation**:
- `treatment_cohort`: Year of first high MPOWER adoption (0 = never treated)
- `ever_treated`: Binary indicator for ever achieving high MPOWER
- `post_treatment`: Binary indicator for post-treatment periods
- `years_since_treatment`: Time since treatment adoption
- `gdp_log`: Log-transformed GDP per capita
- `mpower_total_lag1-3`: Lagged MPOWER scores for dynamics

**Data Quality**:
- Dropped countries with < 5 years of data to ensure sufficient time series
- Final sample: 195 countries, 1,170 country-year observations
- Missing data: <3% for mortality outcomes, ~14% for some controls

## Final Dataset Characteristics

### Panel Structure
- **Unit of Analysis**: Country-year
- **Time Period**: 2008-2018 (6 time periods)
- **Countries**: 195 countries
- **Observations**: 1,170 country-year observations
- **Treatment Variation**: 44 countries with staggered MPOWER adoption

### Outcome Variables (Mean ± SD)
- Lung cancer mortality: 11.00 ± 8.52 deaths per 100,000
- CVD mortality: 37.78 ± 29.84 deaths per 100,000
- IHD mortality: 23.97 ± 20.58 deaths per 100,000
- COPD mortality: 11.29 ± 12.94 deaths per 100,000

### Treatment Variables (Mean ± SD)
- MPOWER total score: 19.74 ± 4.30 (out of 29)
- Monitor score: 2.88 ± 1.05 (out of 4)
- Protect score: 3.12 ± 1.19 (out of 4)
- Offer score: 2.89 ± 1.06 (out of 4)
- Warn score: 3.73 ± 0.74 (out of 4)
- Enforce score: 3.73 ± 0.72 (out of 4)
- Raise taxes score: 3.41 ± 1.06 (out of 5)

### Control Variables
- GDP per capita: Wide variation across countries
- Urban population: 14% missing, managed through multiple imputation
- Population size: Available for most observations

## Data Quality Assessment

### Strengths
1. **High-quality sources**: WHO, IHME, and World Bank are gold-standard data providers
2. **Treatment variation**: Meaningful staggered adoption across time and countries
3. **Outcome relevance**: Age-standardized mortality rates are standard epidemiological measures
4. **Temporal alignment**: Good overlap period (2008-2018) across all sources
5. **Sample size**: Sufficient observations for credible causal inference

### Limitations
1. **Biennial MPOWER data**: Reduces precision of treatment timing
2. **Missing controls**: ~14% missing for some World Bank variables
3. **Country name matching**: Some potential mismatches in merge process
4. **Treatment definition**: Binary threshold may miss intensity effects

### Suitability for Causal Inference
- ✅ **Panel structure**: Balanced time series for most countries
- ✅ **Treatment variation**: Clear staggered adoption pattern
- ✅ **Pre-treatment data**: Multiple periods before treatment for most cohorts
- ✅ **Control group**: Large set of never-treated countries
- ✅ **Outcome measurement**: Standardized, objective mortality measures
- ✅ **Confounders**: Key economic and demographic controls included

## Files Generated

1. **`data/processed/integrated_panel.csv`**: Raw merged data from all sources
2. **`data/processed/analysis_ready_data.csv`**: Final dataset with treatment cohorts and transformed variables
3. **`scripts/data_processing.py`**: Complete data processing pipeline
4. **`data/processed/DATA_PROCESSING_SUMMARY.md`**: This documentation

## Next Steps

The analysis-ready dataset is suitable for implementing:

1. **Callaway & Sant'Anna (2021) staggered DiD**: Primary causal identification strategy
2. **Two-way fixed effects models**: Robustness check with traditional approach
3. **Synthetic control methods**: Case study analysis for specific countries
4. **Event study analysis**: Dynamic treatment effects and pre-trends testing

**Recommended Analysis Sequence**:
1. Descriptive statistics and visualization
2. Pre-trends testing and parallel trends assumption
3. Main staggered DiD results with multiple aggregations
4. Robustness checks with alternative specifications
5. Heterogeneity analysis by region, income, baseline conditions

---

*This dataset was prepared following best practices for causal inference in panel data and is ready for academic-quality analysis of tobacco control policy effectiveness.*
