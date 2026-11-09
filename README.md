# NYC Urban Growth & Road Safety

An interactive analytics dashboard examining relationships between affordable housing production and recorded serious traffic injuries across New York City ZIP codes.

The project integrates NYC housing and SIRTA data at ZIP-code/year level and presents eight research analyses through a responsive Dash website. Results are exploratory and do not establish causation.

## Live Dashboard

[Open the NYC Urban Growth & Road Safety website](https://nyc-urban-growth-road-safety.onrender.com)

## Key Features

- Eight interactive Plotly visualizations covering RQ1-RQ8.
- Borough, year, construction type, and ZIP-code filters.
- Text search such as `Brooklyn 2024 new construction`.
- Dynamic KPI cards, summaries, correlations, medians, and housing-production groups.
- Recalculated balanced ZIP panels for temporal and year-to-year analyses.
- Clear explanations when a selected scope has insufficient observations.
- Responsive desktop and mobile design.

## Technology Stack

- Python Dash
- Plotly
- Pandas and NumPy
- SciPy
- Custom CSS
- Gunicorn and Render

## Data and Methodology

The primary research period is 2023-2025. The integrated dataset also contains 2022 for data-engineering and coverage context.

Processed datasets:

- `data/final_integrated_dataset.csv`
- `data/analysis_dataset.csv`

The application preserves the notebook methodology:

- Incomplete housing measurements are excluded where required.
- Geographic ambiguity is handled according to each research question.
- Missing or unmatched observations are not converted to zero.
- New Construction and Preservation use their respective unit measurements.
- Balanced panels, median thresholds, tertiles, and statistics are recalculated after filtering.

Original sources:

- [Affordable Housing Production by Building](https://data.cityofnewyork.us/d/hg8x-zxpr)
- [Serious Injury Response, Tracking & Analysis (SIRTA)](https://data.cityofnewyork.us/d/xeqp-qz8h)

## Run Locally

```bash
python -m venv .venv
```

Activate the environment:

```powershell
# Windows
.venv\Scripts\Activate.ps1
```

```bash
# macOS/Linux
source .venv/bin/activate
```

Install dependencies and start the application:

```bash
python -m pip install -r requirements.txt
python app.py
```

Open [http://127.0.0.1:8050](http://127.0.0.1:8050).

## Deployment

The application exposes `server = app.server` and includes `render.yaml` and `Procfile` production configurations.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Ahmed-Elsafty36/NYC-Urban-Growth-Road-Safety)

Production URL: [https://nyc-urban-growth-road-safety.onrender.com](https://nyc-urban-growth-road-safety.onrender.com)

## Limitations

- SIRTA has incomplete geographic coverage and does not represent every NYC collision.
- Results use raw counts without population or traffic-exposure adjustment.
- ZIP-year aggregation can hide street-level variation.
- Correlation and geographic co-occurrence do not establish causation.

## Individual Contribution

This project was completed entirely and independently by:

- **Name:** Ahmed Maher Elsafty
- **Student ID:** 16005023
- **Tutorial:** 5

All data engineering, cleaning, integration, analysis, visualization, website development, testing, documentation, and deployment work was completed individually.
