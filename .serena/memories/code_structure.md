# Project Structure & Code Organization

## Current Directory Structure
```
project/
├── .claude/                    # Claude Code configuration (access restricted)
├── .git/                      # Git repository
├── .serena/                   # Serena MCP memory storage
├── CLAUDE.md                  # Project conventions and AI rules
├── README.md                  # Context Engineering template documentation
├── mpower-research-plan.md    # Detailed research methodology
├── PRPs/                      # Product Requirements Prompts
│   ├── ai_docs/              # Research documentation for AI context
│   │   ├── causal_inference_methods.md
│   │   ├── data_sources_apis.md
│   │   └── python_packages.md
│   ├── templates/            # PRP base templates
│   ├── initial_mpower.md     # Initial feature specification
│   └── mpower_causal_analysis.md  # Main implementation PRP
├── data/                     # Data directory
│   └── raw/                  # Raw data files (WHO, IHME, World Bank)
│       ├── gbd/              # IHME Global Burden of Disease data
│       ├── mpower_gho/       # WHO MPOWER data
│       └── worldbank/        # World Bank indicators
├── examples/                 # Code examples (currently empty except .gitkeep)
└── use-cases/               # Use case documentation
```

## Planned Code Structure (from PRP)
```
src/                          # Main source code (500-line limit per file)
├── __init__.py
├── data/                     # Data processing modules
│   ├── __init__.py
│   ├── cleaning.py          # Raw data cleaning and validation
│   ├── integration.py       # Multi-source data merging
│   └── validation.py        # Data quality checks
├── analysis/                # Econometric analysis modules
│   ├── __init__.py
│   ├── did_analysis.py      # Callaway & Sant'Anna implementation
│   ├── fixed_effects.py    # Two-way FE models
│   ├── synthetic_control.py # Synthetic control methods
│   └── event_studies.py    # Event study analysis
├── visualization/           # Plotting and reporting
│   ├── __init__.py
│   ├── plots.py            # Statistical plots
│   └── tables.py           # Regression tables
└── utils/                  # Utility functions
    ├── __init__.py
    ├── data_utils.py       # Data manipulation helpers
    └── estimation_utils.py # Statistical utilities

tests/                       # Test suite mirroring src/ structure
├── test_data/
├── test_analysis/
├── test_visualization/
└── conftest.py

output/                      # Generated results
├── tables/                 # Regression tables
├── figures/                # Plots and visualizations
└── reports/                # Generated reports
```

## File Organization Principles (from CLAUDE.md)
- **Maximum 500 lines per file** - Refactor by splitting into modules if approaching limit
- **Clear module separation** - Group by feature/responsibility
- **Consistent imports** - Prefer relative imports within packages
- **Agent pattern** (if applicable):
  - `agent.py` - Main agent definition
  - `tools.py` - Tool functions
  - `prompts.py` - System prompts

## Key Configuration Files
- **CLAUDE.md** - AI assistant rules and project conventions
- **PRPs/** - Product Requirements Prompts for AI-guided development
- Data exists in `data/raw/` but no processing pipeline yet implemented
