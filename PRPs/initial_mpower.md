# INITIAL\_mpower.md – MPOWER Tobacco Control Policy Impact Research

**FEATURE:**\
Rigorous, reproducible causal inference analysis of the WHO MPOWER tobacco control policies (2008–2019) to evaluate their impact on smoking prevalence, lung cancer mortality, and cardiovascular disease mortality worldwide.

---

**EXAMPLES:**

- **For Policymakers:** Learn which MPOWER measures save lives fastest (1–5 years) and yield the highest return on investment.
- **For Researchers:** Access the first application of Callaway & Sant’Anna (2021) staggered DiD methods to MPOWER evaluation, with fully reproducible code.
- **For Citizens & Advocates:** See which government actions most effectively reduce tobacco-related deaths globally.
---

## 1. Research Questions & Impact

### Primary Research Question
How have the adoption and strengthening of WHO's MPOWER tobacco control measures between 2008-2019 affected national trends in:
- Adult tobacco use prevalence
- Lung cancer mortality rates
- Cardiovascular disease mortality rates

### Key Contributions to Literature
1. **Methodological Advance**: First application of Callaway & Sant'Anna (2021) staggered DiD methods to MPOWER evaluation
2. **Novel Outcomes**: Direct empirical link between MPOWER and observed mortality (vs. simulation-based estimates)
3. **Comprehensive Scope**: Systematic comparison of all six MPOWER components with lag structures
4. **Causal Identification**: Addresses selection bias and time-varying confounders that limit existing studies

---

## 2. Hypotheses

**H1**: Stronger MPOWER implementation is associated with statistically significant declines in national adult smoking prevalence.

**H2**: Stronger MPOWER implementation is associated with reduced lung cancer mortality rates, with effects observable after policy adoption lags.

**H3**: Stronger MPOWER implementation is associated with reduced cardiovascular disease mortality rates, with effects observable after policy adoption lags.

**H4** (exploratory): Individual MPOWER components have heterogeneous effects, with taxation (R) and smoke-free policies (P) showing the largest mortality impacts.

---

## 3. Data Sources & Variable Definitions

### 3.1 MPOWER Policy Variables
**Source**: WHO Report on the Global Tobacco Epidemic (2008-2019 editions)

| Variable | Description | Scale | Source Years |
|----------|-------------|-------|--------------|
| `mpower_total` | Composite score | 0-37 | 2008, 2010, 2012, 2014, 2016, 2018, 2019 |
| `m_score` | Monitor tobacco use | 0-4 | Biennial |
| `p_score` | Protect from smoke | 0-4 | Biennial |
| `o_score` | Offer help to quit | 0-4 | Biennial |
| `w_score` | Warn about dangers | 0-4 | Biennial |
| `e_score` | Enforce ad bans | 0-4 | Biennial |
| `r_score` | Raise taxes | 0-5 | Biennial |

**Note**: Linear interpolation for missing years; last observation carried forward for 2019-forward

### 3.2 Health Outcome Variables
**Source**: IHME Global Burden of Disease (GBD) 2019

| Variable | Definition | Unit | ICD-10 |
|----------|-----------|------|--------|
| `prev_smoking_adult` | Age-standardized daily smoking prevalence, ages 15+ | % | - |
| `mort_lung_cancer_asr` | Age-standardized lung cancer mortality | per 100,000 | C33-C34 |
| `mort_cvd_asr` | Age-standardized CVD mortality | per 100,000 | I00-I99 |

### 3.3 Control Variables

| Variable | Description | Source | Transformation |
|----------|-------------|--------|----------------|
| `gdp_pc_constant_2010` | GDP per capita (constant 2010 USD) | World Bank WDI | Log |
| `urban_pop_pct` | Urban population (% of total) | World Bank WDI | None |
| `edu_index` | Education index component of HDI | UNDP | None |
| `haq_index` | Healthcare Access & Quality Index | GBD 2019 | None |
| `alcohol_consumption` | Liters per capita | WHO GHO | Log |
| `air_pollution_pm25` | Annual mean PM2.5 exposure | GBD 2019 | None |

### 3.4 Sample Construction
- **Primary merge key**: ISO3 country code + year
- **Panel structure**: Unbalanced panel, ~180 countries × 12 years
- **Exclusions**:
  - Countries with <5 years of data
  - Small island states <100,000 population
  - Countries with ongoing conflicts affecting data quality

---

## 4. Methodology

### 4.1 Main Specification - Two-Way Fixed Effects

```
Y_it = β₀ + β₁MPOWER_it + β₂X_it + μᵢ + λₜ + εᵢₜ
```

Where:
- Y_it = outcome (prevalence/mortality) for country i in year t
- MPOWER_it = total or component score (with lags 0-5 years)
- X_it = time-varying covariates
- μᵢ = country fixed effects (controls for time-invariant factors)
- λₜ = year fixed effects (controls for global trends)
- εᵢₜ = idiosyncratic error

**Standard errors**: Clustered at country level to account for serial correlation

### 4.2 Difference-in-Differences for Staggered Adoption

For countries crossing MPOWER implementation thresholds at different times:

```
Y_it = α + τ̂(g,t)D_it + μᵢ + λₜ + εᵢₜ
```

Using Callaway & Sant'Anna (2021) estimator for:
- Treatment: Achieving "high" MPOWER score (≥25/37)
- Accounts for: Heterogeneous treatment effects, negative weighting issues
- Aggregation: Simple weighted average of group-time ATTs

### 4.3 Lag Structure Analysis

Estimate distributed lag model:
```
Y_it = β₀ + Σ(k=0 to 5) βₖMPOWER_i,t-k + X_it'γ + μᵢ + λₜ + εᵢₜ
```

To identify:
- Immediate effects (k=0)
- Short-term effects (k=1-2)
- Medium-term effects (k=3-5)

### 4.4 Heterogeneity Analysis

Interact MPOWER with:
- Income level (World Bank classification)
- Baseline smoking prevalence (above/below median)
- Regional indicators
- Gender-specific models

### 4.5 Robustness Checks

1. **Alternative specifications**:
   - Random effects models
   - First differences
   - Arellano-Bond GMM for dynamic panels

2. **Sample sensitivity**:
   - Exclude high-income countries
   - Exclude countries with major tobacco production
   - Balanced panel only

3. **Synthetic control** (for top implementers):
   - Construct synthetic controls for early/strong adopters
   - Compare actual vs. synthetic trajectories

4. **Placebo tests**:
   - Outcomes: Traffic accidents, suicide rates
   - Timing: Artificial policy adoption dates

5. **Alternative MPOWER scoring**:
   - Population-weighted scores
   - Principal component analysis
   - Binary indicators for "best practice"

---

---

**DOCUMENTATION:**
- User web search to located the following documentation and save relevant parts in @PRPs/ai_docs for future reference
- WHO MPOWER Reports
- IHME Global Burden of Disease (GBD)
- World Bank World Development Indicators (WDI)
- UNDP Human Development Data
- Callaway & Sant’Anna (2021) Staggered DiD
- Abadie et al. (2010) Synthetic Control

---

**Single-Container (Flow Overview):**

```
[Research Team]
     |
     v
[Single Container Environment]
     |------------------------------|
     |                              |
[Data Cleaning + Processing]   [Analysis + Modeling]
     |                              |
     v                              v
[Raw Data (manual download)]   [Causal Inference Results]
                                 |
                                 v
                          [Reports + Dashboards]
```

---

**TECH STACK:**

- **Core Language:** Python only
- **Environment & Dependency Management:** uv
- **Key Packages:** pandas, numpy, statsmodels, linearmodels, pyfixest, differences, matplotlib, seaborn, plotly, duckdb, pytest
- **Data Infrastructure:** DuckDB (local), parquet/feather formats for processed data
- **Workflow & QA:** pytest for testing, pre-commit hooks for style/quality
- **IDE Setup:** VS Code with Python, Jupyter, and GitHub Copilot
- **AI Assistance:** Claude Code optional for data cleaning, code generation, and documentation support

---

**uv PROJECT CONFIGURATION (pyproject.toml):**

```toml
[project]
name = "mpower-analysis"
version = "0.1.0"
description = "Causal inference analysis of WHO MPOWER tobacco control policies"
requires-python = ">=3.11"
dependencies = [
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "statsmodels>=0.14.0",
    "linearmodels>=5.0",
    "pyfixest>=0.18.0",
    "differences>=0.3.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "plotly>=5.14.0",
    "duckdb>=0.9.0",
    "pytest>=7.3.0"
]

[project.optional-dependencies]
dev = [
    "black>=23.0.0",
    "ruff>=0.1.0",
    "pre-commit>=3.3.0"
]
```

---

**MINIMAL DOCKERFILE TEMPLATE:**

```dockerfile
# Dockerfile – Container for replication
FROM python:3.11-slim
WORKDIR /app

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Copy dependencies and install
COPY requirements.txt .
RUN uv pip install -r requirements.txt

COPY . .

CMD ["bash"]
```

---

**MINIMAL FASTAPI EXAMPLE (Replication API):**

```python
# backend/main.py
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "ok"}
```

---

**DIRECTORY STRUCTURE:**

```
project/
  data/
    raw/           # WHO, GBD, WDI manually downloaded data
    ├── gbd
    │   ├── IHME-GBD_2008-2019_DATA-AllCountries-Tobacco.csv
    │   └── citation.txt
    ├── mpower_gho
    │   └── mpower_gho_data.csv
    └── worldbank
        ├── combined_downloads_analysis.json
        ├── download_analysis_report.json
        ├── worldbank_wdi_additional_20250815.csv
        ├── worldbank_wdi_additional_metadata_20250815.csv
        ├── worldbank_wdi_manual_20250814.csv
        └── worldbank_wdi_metadata_20250814.csv
    processed/     # Final panel dataset
  src/
    01_data_cleaning.py
    02_analysis.py
    03_visualization.py
    04_robustness.py
  output/
    tables/
    figures/
    reports/
  docs/
    codebook.md
    methods.md
```

---

**SETUP SCRIPT (setup.sh):**

```bash
#!/bin/bash
# setup.sh - Environment setup for MPOWER analysis

# Install uv if not already installed
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Create virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt || uv pip install .

# Create project structure
mkdir -p data/{raw,processed} code output/{tables,figures,reports} docs

# Initialize git with pre-commit hooks
if [ ! -d .git ]; then
  git init
fi
pre-commit install

echo "Setup complete! Activate environment with: source .venv/bin/activate"
```

---

**PYTEST EXAMPLE TEST (tests/test\_data\_quality.py):**

```python
import pytest
import pandas as pd

# Example test to validate MPOWER scores are within valid bounds
def test_mpower_scores():
    df = pd.read_csv('data/processed/panel_data.csv')
    assert df['mpower_total'].between(0, 37).all()
    assert not df['iso3'].duplicated().any()
```

---

**OTHER CONSIDERATIONS:**

- **Risks:** Missing MPOWER data, mortality misclassification, convergence issues → mitigated with interpolation, multiple cause-of-death codes, alternative optimizers.
- **QA:** Validation tests for data bounds, balanced panel checks, multiple SE corrections.
- **Success Metrics:** >150 countries coverage, R² >0.7, reproducibility, policy citations.
- **Future Considerations:** Extend to other NCD policies, add causal ML methods, interactive dashboards.
