# NYC Urban Growth & Road Safety

Interactive analytics website for the GIU Data Engineering and Visualization project. The application integrates affordable-housing production with serious traffic injury records at NYC ZIP-code/year grain and implements the notebook's eight research questions with Dash, Plotly, Pandas, and custom CSS.

The work is exploratory. Associations shown by the dashboard do not establish that housing activity causes traffic injuries or fatalities.

## What the website includes

- Responsive navigation, report controls, KPI cards, and research storytelling.
- Guided filters for borough, primary research year, housing measure, and ZIP code.
- A transparent text-search mode supporting queries such as `Brooklyn 2024 new construction`.
- A central **Generate Report** action; controls do not update the report until it is selected.
- Eight interactive Plotly views corresponding to RQ1–RQ8.
- Hover details for ZIP code, borough, year, housing units, severe injuries, and fatalities.
- Dynamic correlations, balanced panels, median thresholds, housing-production tertiles, and construction-specific subsets.
- Explicit empty states when a selected scope cannot support the notebook method.

## Research questions

| ID | Question | Dashboard method |
|---|---|---|
| RQ1 | Are areas with greater affordable housing production associated with more severe traffic injuries? | ZIP-year scatter; Pearson and Spearman statistics |
| RQ2 | How does the relationship differ across boroughs? | Borough facets; Spearman statistic by borough |
| RQ3 | How do housing production and severe injuries change over time? | Balanced-ZIP indexed series |
| RQ4 | Which ZIP codes have relatively high housing activity and severe injuries? | Latest-selected-year median quadrants |
| RQ5 | Is housing production associated with recorded traffic fatalities? | Log-housing scatter; Pearson and Spearman statistics |
| RQ6 | Do housing-production levels have different injury distributions? | Recalculated Low/Medium/High tertiles and box plots |
| RQ7 | Does the association differ by construction type? | Positive-unit New Construction and Preservation subsets |
| RQ8 | Are within-ZIP annual changes associated? | Balanced-panel, consecutive-year change scatter |

## Analytical safeguards

- RQ analysis is limited to the notebook's primary 2023–2025 period. The integrated data also contains 2022 for source-coverage context.
- Rows where `Housing_Data_Complete` is false are excluded from housing analysis.
- RQ2–RQ8 exclude `Geographic_Ambiguity`; RQ1 retains and visibly identifies ambiguous rows to preserve the notebook's sensitivity design.
- Missing values are never filled with zero.
- Selecting **New Construction** uses `New_Construction_Units`; selecting **Preservation** uses `Preservation_Units`.
- RQ3 and RQ8 rebuild their balanced ZIP panels after every geographic and time filter.
- RQ4 recalculates both medians from the current eligible observations.
- RQ6 recalculates rank-first tertiles after filtering.
- RQ8 only uses consecutive years; a gap such as 2023 to 2025 is not mislabeled as an annual change.
- Static findings in the story section are explicitly identified as the unfiltered notebook baseline. Chart summaries are recalculated for the generated scope.

## Data

Processed inputs:

- `data/final_integrated_dataset.csv` — 367 matched ZIP-year rows covering 2022–2025.
- `data/analysis_dataset.csv` — 366 rows with complete housing-unit measurements.

Original NYC Open Data sources:

- [Affordable Housing Production by Building](https://data.cityofnewyork.us/d/hg8x-zxpr)
- [Serious Injury Response, Tracking & Analysis (SIRTA)](https://data.cityofnewyork.us/d/xeqp-qz8h)

The source CSV files and `notebook/NYC_Data_Engineering.ipynb` are read-only project evidence and are not modified by the application.

## Project structure

```text
.
├── app.py
├── assets/
│   └── style.css
├── data/
│   ├── analysis_dataset.csv
│   └── final_integrated_dataset.csv
├── notebook/
│   └── NYC_Data_Engineering.ipynb
├── Procfile
├── render.yaml
├── requirements.txt
└── README.md
```

## Run locally

Python 3.11 is recommended.

```bash
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS/Linux
source .venv/bin/activate
```

Install and run:

```bash
python -m pip install -r requirements.txt
python app.py
```

Open [http://127.0.0.1:8050](http://127.0.0.1:8050).

## Production and Render

The Dash Flask server is exposed as `server = app.server`. Both `render.yaml` and `Procfile` use Gunicorn:

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Ahmed-Elsafty36/NYC-Urban-Growth-Road-Safety)

```bash
gunicorn app:server --bind 0.0.0.0:$PORT
```

For Render Blueprint deployment, connect the repository and choose **New Blueprint Instance**. Render will read `render.yaml`, install `requirements.txt`, and start the web service. The health check uses Dash's root route (`/`).

## Validation

Recommended local checks:

```bash
python -m compileall app.py
python -c "import app; print(app.server.test_client().get('/').status_code)"
```

The application also validates required columns, ZIP-year key uniqueness, and alignment between `analysis_dataset.csv` and the complete-data subset at startup.

## Important limitations

- SIRTA does not represent every NYC collision and contains incomplete geographic coverage.
- The integrated sample includes only ZIP-year combinations represented in both source datasets.
- Injury and fatality values are raw counts, not exposure-adjusted rates.
- Counted affordable units include new construction and preservation and do not necessarily represent net additions to housing stock.
- ZIP-year aggregation can conceal street-level variation and within-year ordering.
- Repeated ZIP codes make conventional correlation p-values exploratory.
- Borough and construction-type differences were not formally tested against one another.
- Correlation is not causation.

## Course context

- University: German International University (GIU)
- Faculty: Informatics and Computer Science
- Course: Data Engineering and Visualization
- Semester: Winter 2026

