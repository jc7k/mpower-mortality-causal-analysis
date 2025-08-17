# Technology Stack & Architecture

## Primary Language
**Python** - The project is primarily Python-based for data analysis and econometric modeling

## Core Dependencies (from research plan)
### Data Processing & Analysis
- `pandas>=2.0.0` - Data manipulation and analysis
- `numpy>=1.24.0` - Numerical computing
- `statsmodels>=0.14.0` - Statistical modeling
- `linearmodels>=5.0` - Panel data models
- `pyfixest>=0.18.0` - High-dimensional fixed effects
- `differences>=0.3.0` - Callaway-Sant'Anna DiD implementation
- `scikit-learn>=1.3.0` - Machine learning tools

### Visualization & Reporting
- `matplotlib>=3.7.0` - Basic plotting
- `seaborn>=0.12.0` - Statistical visualization
- `plotly>=5.14.0` - Interactive plots
- `weasyprint>=60.0` - PDF generation from HTML/CSS

### Data Sources & APIs
- `wbgapi>=1.0.0` - World Bank API access
- `requests>=2.31.0` - HTTP requests
- `beautifulsoup4>=4.12.0` - PDF/HTML parsing
- `tabula-py>=2.5.0` - PDF table extraction

### Development Tools
- `pytest>=7.3.0` - Testing framework
- `black>=23.0.0` - Code formatting
- `ruff>=0.1.0` - Fast linting
- `jupyter>=1.0.0` - Interactive notebooks

## Package Management
- **uv** - Fast Python package and project management (preferred over pip/conda/poetry)
- Virtual environment: `.venv` (not currently set up)

## Alternative/Complementary Stack (R)
Available for methods not in Python:
- `plm` - Panel data models
- `did` - Callaway-Sant'Anna implementation
- `Synth` - Synthetic control methods
- `fixest` - Fast fixed effects
- `reticulate` - Python-R bridge

## Data Infrastructure
- **Storage**: Local files (raw CSV/PDF, processed Parquet)
- **Database**: Optional DuckDB for analytical queries
- **Formats**: CSV/Excel (raw) → Parquet (processed) → Feather (analysis)

## Development Environment
- **IDE**: VS Code with Python, Jupyter, R extensions
- **Version Control**: Git with Git LFS for large files
- **Claude Code**: AI-assisted development for data cleaning, analysis, documentation

## No Current Dependency Management
- No `requirements.txt`, `pyproject.toml`, or virtual environment detected
- Dependencies listed in research plan but not yet implemented
