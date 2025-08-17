# Data Sources and APIs for MPOWER Analysis

## WHO MPOWER Tobacco Control Data

### Data Source
- **Source**: WHO Report on the Global Tobacco Epidemic (2008-2019 editions)
- **Coverage**: Biennial reports with policy scores for ~180 countries
- **Manual Download**: Data already obtained and stored in `data/raw/`

### MPOWER Components and Scoring

#### Policy Areas (0-37 total score)
1. **M - Monitor** (0-4): Tobacco use surveillance systems
2. **P - Protect** (0-4): Smoke-free policies
3. **O - Offer** (0-4): Cessation support services
4. **W - Warn** (0-4): Health warning labels
5. **E - Enforce** (0-4): Advertising, promotion, sponsorship bans
6. **R - Raise** (0-5): Tobacco taxation

#### Data Access Methods
- **Primary**: Manual download from WHO reports (already completed)
- **API Alternative**: WHO GHO (Global Health Observatory)
  - URL: https://www.who.int/data/gho/data/themes/topics/topic-details/GHO/gho-tobacco-conntrol-mpower-progress-towards-selected-tobacco-control-policies-for-demand-reduction
  - Methods: GHO OData API, Athena API, bulk ZIP download

### Data Processing Requirements
- **Interpolation**: Linear interpolation for missing years between reports
- **Panel Construction**: Country-year format with ISO3 codes
- **Validation**: Scores must be within valid ranges (M,P,O,W,E: 0-4; R: 0-5; Total: 0-37)

## IHME Global Burden of Disease (GBD) Data

### Data Source
- **Source**: Institute for Health Metrics and Evaluation (IHME) GBD 2019
- **Access Portal**: https://vizhub.healthdata.org/gbd-results/
- **Manual Download**: Mortality data already obtained

### Key Variables for Analysis

#### Mortality Outcomes
1. **Lung Cancer Mortality**
   - ICD-10: C33-C34
   - Measure: Age-standardized death rate per 100,000
   - Both sexes, all ages

2. **Cardiovascular Disease Mortality**
   - ICD-10: I00-I99
   - Measure: Age-standardized death rate per 100,000
   - Both sexes, all ages

3. **Smoking Prevalence**
   - Daily smoking prevalence, age-standardized
   - Ages 15+ years, both sexes

### Data Access Method
- **Required**: Account creation at healthdata.org
- **Download Format**: CSV files
- **Coverage**: 204 countries/territories, 1990-2021
- **Uncertainty Intervals**: Available but not required for main analysis

### GBD Data Processing
```python
# Expected data structure from GBD downloads
gbd_columns = [
    'location_name',      # Country name
    'location_id',        # IHME location ID
    'year',              # Year
    'cause_name',        # Cause of death
    'measure_name',      # Death rate, prevalence, etc.
    'metric_name',       # Rate, percent, count
    'val',               # Point estimate
    'upper',             # Upper uncertainty interval
    'lower'              # Lower uncertainty interval
]
```

## World Bank World Development Indicators (WDI)

### Data Source
- **Source**: World Bank Open Data
- **Manual Download**: Control variables already obtained
- **API Alternative**: wbgapi Python package (not needed - data exists)

### Key Control Variables

#### Economic Indicators
1. **GDP per capita (constant 2010 USD)**
   - Indicator: NY.GDP.PCAP.KD
   - Transformation: Log for regression analysis

2. **Urban Population (% of total)**
   - Indicator: SP.URB.TOTL.IN.ZS
   - Used as level

#### Development Indicators
3. **Education Index** (from UNDP HDI)
   - Alternative: Mean years of schooling
   - Indicator: BAR.SCHL.15UP

4. **Healthcare Access & Quality Index**
   - From GBD 2019 covariates
   - Alternative WDI: Health expenditure % GDP

#### Additional Controls
5. **Alcohol Consumption** (liters per capita)
   - WHO Global Health Observatory
   - Transformation: Log

6. **Air Pollution (PM2.5)**
   - Annual mean exposure
   - From GBD environmental risk factors

### Data Integration Requirements

#### Country Code Harmonization
```python
# Required mapping between data sources
country_mapping = {
    'iso3': 'ISO 3-letter codes',           # Primary merge key
    'country_name_wb': 'World Bank names',   # WDI source
    'location_name_gbd': 'GBD location names', # IHME source
    'country_name_who': 'WHO country names'  # MPOWER source
}
```

#### Panel Structure
```python
# Target panel data structure
panel_columns = [
    # Identifiers
    'iso3', 'country_name', 'year',

    # MPOWER policy variables
    'mpower_total', 'm_score', 'p_score', 'o_score',
    'w_score', 'e_score', 'r_score',

    # Health outcomes (GBD)
    'prev_smoking_adult', 'mort_lung_cancer_asr', 'mort_cvd_asr',

    # Control variables (WDI + others)
    'gdp_pc_constant_2010', 'urban_pop_pct', 'edu_index',
    'haq_index', 'alcohol_consumption', 'air_pollution_pm25'
]
```

## Data Quality and Validation

### Coverage Requirements
- **Minimum**: 150+ countries with 5+ years of data
- **Time Period**: 2008-2019 (12 years)
- **Balanced Panel**: Not required (staggered DiD handles unbalanced)

### Quality Checks
```python
# Essential validation tests
def validate_panel_data(df):
    """
    Validate merged panel dataset
    """
    # MPOWER score bounds
    assert df['mpower_total'].between(0, 37).all()
    assert df[['m_score', 'p_score', 'o_score', 'w_score', 'e_score']].between(0, 4).all()
    assert df['r_score'].between(0, 5).all()

    # No duplicate country-years
    assert not df.groupby(['iso3', 'year']).size().gt(1).any()

    # Reasonable outcome ranges
    assert df['prev_smoking_adult'].between(0, 100).all()
    assert df['mort_lung_cancer_asr'].gt(0).all()  # Mortality rates > 0

    # Required coverage
    countries_per_year = df.groupby('year')['iso3'].nunique()
    assert countries_per_year.min() >= 100  # At least 100 countries per year
```

### Missing Data Handling
1. **MPOWER Scores**: Linear interpolation between survey years
2. **Mortality Data**: No imputation (use available data)
3. **Control Variables**: Forward/backward fill within country, max 2 years

## Alternative Data Sources (Backup)

### MPOWER Alternative Sources
- **Tobacco Atlas**: Independent policy tracking
- **FCTC Implementation Database**: WHO Framework Convention reporting
- **TobaccoFree Kids**: Policy tracking by advocacy organization

### Mortality Alternative Sources
- **WHO Mortality Database**: Country-reported vital statistics
- **UN Population Division**: Demographic estimates
- **Country-specific**: National statistical offices

### Control Variables Alternatives
- **UNDP Human Development Database**: HDI components
- **OECD Statistics**: For high-income countries
- **IMF World Economic Outlook**: Economic indicators
