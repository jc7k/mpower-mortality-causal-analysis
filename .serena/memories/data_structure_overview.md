# Data Structure & Research Design Overview

## Data Sources Present in `data/raw/`

### 1. WHO MPOWER Data (`mpower_gho/`)
- **File**: `mpower_gho_data.csv`
- **Content**: WHO tobacco control policy scores
- **Variables**: M, P, O, W, E, R scores (0-4 or 0-5 scale)
- **Time Period**: 2008-2019 (biennial reports)
- **Coverage**: ~180 countries

### 2. IHME Global Burden of Disease (`gbd/`)
- **File**: `IHME-GBD_2008-2019_DATA-AllCountries-Tobacco.csv`
- **Content**: Health outcomes data
- **Variables**:
  - Adult smoking prevalence (age-standardized)
  - Lung cancer mortality (age-standardized, per 100,000)
  - Cardiovascular disease mortality (age-standardized, per 100,000)
- **Time Period**: 2008-2019
- **Coverage**: National level, both sexes

### 3. World Bank Data (`worldbank/`)
- **Files**: Multiple CSV files with WDI indicators
- **Content**: Control variables
- **Variables**:
  - GDP per capita (constant 2010 USD)
  - Urban population percentage
  - Education index
  - Healthcare access & quality index
  - Other socioeconomic indicators

## Research Design Framework

### Treatment Variables (MPOWER Components)
- **M**: Monitor tobacco use (0-4 scale)
- **P**: Protect from smoke (0-4 scale)
- **O**: Offer help to quit (0-4 scale)
- **W**: Warn about dangers (0-4 scale)
- **E**: Enforce ad bans (0-4 scale)
- **R**: Raise taxes (0-5 scale)
- **Total**: Composite score (0-37 scale)

### Outcome Variables
1. **Smoking Prevalence**: Age-standardized daily smoking, ages 15+
2. **Lung Cancer Mortality**: Age-standardized mortality per 100,000
3. **CVD Mortality**: Age-standardized cardiovascular mortality per 100,000

### Panel Structure
- **Unit**: Countries (ISO3 codes)
- **Time**: 2008-2019 (12 years)
- **Structure**: Unbalanced panel (~180 countries × 12 years)
- **Treatment**: Staggered adoption of "high" MPOWER scores (≥25/37)

## Analysis Methods to Implement

### Primary Methods
1. **Two-Way Fixed Effects**: Country + year fixed effects with clustered SEs
2. **Callaway & Sant'Anna DiD**: Staggered treatment timing
3. **Event Study Analysis**: Dynamic treatment effects with lags 0-5 years
4. **Synthetic Control**: Robustness checks for major implementers

### Identification Strategy
- **Treatment Assignment**: Countries crossing MPOWER implementation thresholds
- **Controls**: Time-invariant country factors (fixed effects)
- **Confounders**: Time-varying economic, demographic, health system variables
- **Timing**: Exploit staggered policy adoption across countries/years

## Data Processing Pipeline (To Be Implemented)

### Stage 1: Cleaning
- Standardize country codes (ISO3)
- Handle missing values (linear interpolation for MPOWER)
- Validate variable ranges
- Remove outliers/implausible values

### Stage 2: Integration
- Merge datasets on country-year
- Create treatment indicators
- Generate lag structures
- Balance panel decisions

### Stage 3: Validation
- Check coverage (target: 150+ countries)
- Verify temporal consistency
- Test for systematic missingness
- Generate descriptive statistics

## Expected Sample Characteristics
- **Countries**: 150+ with complete data
- **Time Period**: 12 years (2008-2019)
- **Treatment Variation**: Countries implementing policies at different times
- **Control Group**: Countries with low/stable MPOWER scores
- **Exclusions**: Small island states, conflict-affected countries, insufficient data
