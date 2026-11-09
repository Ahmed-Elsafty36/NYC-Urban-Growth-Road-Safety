"""Interactive Dash application for the NYC Urban Growth & Road Safety study.

The application intentionally mirrors the eight research-question methods in
``notebook/NYC_Data_Engineering.ipynb``.  All chart statistics are recalculated
from the selected scope; the notebook's static findings are only shown as a
clearly labelled baseline in the research-story section.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html
from scipy.stats import pearsonr, spearmanr


BASE_DIR = Path(__file__).resolve().parent
FINAL_DATA_PATH = BASE_DIR / "data" / "final_integrated_dataset.csv"
ANALYSIS_DATA_PATH = BASE_DIR / "data" / "analysis_dataset.csv"

PRIMARY_YEARS = [2023, 2024, 2025]
NUMERIC_COLUMNS = [
    "Year",
    "Housing_Records",
    "Unique_Housing_Projects",
    "Affordable_Units",
    "New_Construction_Units",
    "Preservation_Units",
    "Missing_Unit_Records",
    "Crash_Records",
    "Severe_Injuries",
    "Fatalities",
    "Non_Severe_Injuries",
]
REQUIRED_COLUMNS = {
    "ZIP_Code",
    "Year",
    "Affordable_Units",
    "New_Construction_Units",
    "Preservation_Units",
    "Severe_Injuries",
    "Fatalities",
    "Housing_Data_Complete",
    "Borough",
    "Geographic_Ambiguity",
}
HOUSING_METRICS = {
    "All affordable housing": ("Affordable_Units", "Affordable housing units"),
    "New Construction": ("New_Construction_Units", "New Construction units"),
    "Preservation": ("Preservation_Units", "Preservation units"),
}
YEAR_COLORS = {"2023": "#2367d1", "2024": "#d29b24", "2025": "#d66a3a"}
CATEGORY_COLORS = {
    "High Housing / High Injuries": "#d95f59",
    "High Housing / Low Injuries": "#d4a72c",
    "Low Housing / High Injuries": "#6f64c0",
    "Low Housing / Low Injuries": "#78909c",
}


def _as_bool(series: pd.Series) -> pd.Series:
    """Parse exported boolean strings without treating non-empty text as True."""

    return series.astype("string").str.strip().str.lower().map({"true": True, "false": False})


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate the two processed datasets without changing source files."""

    final = pd.read_csv(FINAL_DATA_PATH, dtype={"ZIP_Code": "string"})
    analysis = pd.read_csv(ANALYSIS_DATA_PATH, dtype={"ZIP_Code": "string"})

    missing = REQUIRED_COLUMNS.difference(final.columns)
    if missing:
        raise ValueError(f"Integrated dataset is missing required columns: {sorted(missing)}")

    for frame in (final, analysis):
        frame["ZIP_Code"] = frame["ZIP_Code"].str.zfill(5)
        for column in NUMERIC_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["Housing_Data_Complete"] = _as_bool(frame["Housing_Data_Complete"])
        frame["Geographic_Ambiguity"] = _as_bool(frame["Geographic_Ambiguity"])

    if final.duplicated(["ZIP_Code", "Year"]).any():
        raise ValueError("Integrated data contains duplicate ZIP-year keys.")
    expected_analysis = final.loc[final["Housing_Data_Complete"].eq(True)]
    if set(zip(analysis["ZIP_Code"], analysis["Year"])) != set(
        zip(expected_analysis["ZIP_Code"], expected_analysis["Year"])
    ):
        raise ValueError("analysis_dataset.csv does not match the complete-data subset.")

    return final, analysis


FINAL_DF, ANALYSIS_DF = load_data()
BOROUGHS = sorted(FINAL_DF["Borough"].dropna().unique().tolist())
ZIP_CODES = sorted(FINAL_DF["ZIP_Code"].dropna().unique().tolist())


def format_integer(value: Any) -> str:
    if value is None or pd.isna(value):
        return "Not available"
    return f"{int(round(float(value))):,}"


def format_coefficient(value: Any) -> str:
    if value is None or pd.isna(value):
        return "not available"
    return f"{float(value):.3f}"


def format_p_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return "not available"
    value = float(value)
    return "< 0.001" if value < 0.001 else f"{value:.3f}"


def safe_sum(series: pd.Series) -> float:
    return series.sum(min_count=1)


def correlation_pair(data: pd.DataFrame, x: str, y: str) -> dict[str, float]:
    valid = data[[x, y]].dropna()
    if len(valid) < 3 or valid[x].nunique() < 2 or valid[y].nunique() < 2:
        return {"pearson": np.nan, "pearson_p": np.nan, "spearman": np.nan, "spearman_p": np.nan}
    pearson = pearsonr(valid[x], valid[y])
    spearman = spearmanr(valid[x], valid[y])
    return {
        "pearson": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def spearman_only(data: pd.DataFrame, x: str, y: str, minimum: int = 5) -> tuple[float, float]:
    valid = data[[x, y]].dropna()
    if len(valid) < minimum or valid[x].nunique() < 2 or valid[y].nunique() < 2:
        return np.nan, np.nan
    result = spearmanr(valid[x], valid[y])
    return float(result.statistic), float(result.pvalue)


def style_figure(figure: go.Figure, *, height: int = 430, legend_title: str | None = None) -> go.Figure:
    figure.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=56, r=28, t=42, b=52),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font=dict(family="Inter, Segoe UI, sans-serif", color="#26364a", size=12),
        hoverlabel=dict(bgcolor="#10243e", font_color="#ffffff", font_size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0),
        legend_title_text=legend_title,
        transition_duration=250,
    )
    figure.update_xaxes(showgrid=True, gridcolor="#e9eef5", zerolinecolor="#aab7c8")
    figure.update_yaxes(showgrid=True, gridcolor="#e9eef5", zerolinecolor="#aab7c8")
    return figure


def empty_figure(title: str, explanation: str) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(
        text=f"<b>{title}</b><br><span style='font-size:13px'>{explanation}</span>",
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        align="center",
        font=dict(color="#52647a", size=16),
    )
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    return style_figure(figure, height=390)


def summary_box(*items: Any, note: str | None = None) -> html.Div:
    children: list[Any] = [html.Div(item, className="summary-stat") for item in items]
    if note:
        children.append(html.P(note, className="summary-note"))
    return html.Div(children, className="chart-summary")


def parse_search_query(query: str | None) -> dict[str, Any]:
    """Parse the intentionally small, transparent project search vocabulary."""

    text = (query or "").strip().lower()
    result: dict[str, Any] = {"boroughs": [], "years": [], "zips": [], "construction": None}
    if not text:
        return result

    borough_aliases = {
        "bronx": "Bronx",
        "brooklyn": "Brooklyn",
        "manhattan": "Manhattan",
        "queens": "Queens",
        "staten island": "Staten Island",
        "staten": "Staten Island",
    }
    for token, borough in borough_aliases.items():
        if re.search(rf"\b{re.escape(token)}\b", text) and borough not in result["boroughs"]:
            result["boroughs"].append(borough)

    result["years"] = sorted(
        {int(year) for year in re.findall(r"\b20(?:23|24|25)\b", text) if int(year) in PRIMARY_YEARS}
    )
    result["zips"] = sorted({zip_code for zip_code in re.findall(r"\b\d{5}\b", text) if zip_code in ZIP_CODES})

    if "new construction" in text or re.search(r"\bnew build", text):
        result["construction"] = "New Construction"
    elif "preservation" in text or "preserved" in text:
        result["construction"] = "Preservation"
    elif "affordable" in text or "all housing" in text or "total housing" in text:
        result["construction"] = "All affordable housing"
    return result


def resolve_filters(
    boroughs: list[str] | None,
    years: list[int] | None,
    construction: str | None,
    zips: list[str] | None,
    mode: str,
    query: str | None,
) -> dict[str, Any]:
    selected = {
        "boroughs": list(boroughs or BOROUGHS),
        "years": sorted(int(year) for year in (years or PRIMARY_YEARS) if int(year) in PRIMARY_YEARS),
        "construction": construction if construction in HOUSING_METRICS else "All affordable housing",
        "zips": list(zips or []),
    }
    if mode == "search":
        parsed = parse_search_query(query)
        for key in ("boroughs", "years", "zips"):
            if parsed[key]:
                selected[key] = parsed[key]
        if parsed["construction"]:
            selected["construction"] = parsed["construction"]
    if not selected["years"]:
        selected["years"] = PRIMARY_YEARS.copy()
    return selected


def apply_scope(data: pd.DataFrame, filters: dict[str, Any]) -> pd.DataFrame:
    mask = data["Year"].isin(filters["years"])
    if filters["boroughs"]:
        mask &= data["Borough"].isin(filters["boroughs"])
    if filters["zips"]:
        mask &= data["ZIP_Code"].isin(filters["zips"])
    return data.loc[mask].copy()


def build_balanced_panel(data: pd.DataFrame, years: list[int]) -> tuple[pd.DataFrame, list[str]]:
    if len(years) < 2:
        return data.iloc[0:0].copy(), []
    coverage = data.groupby("ZIP_Code")["Year"].nunique()
    balanced_zips = coverage.loc[coverage.eq(len(years))].index.tolist()
    panel = data.loc[data["ZIP_Code"].isin(balanced_zips)].copy()
    return panel, balanced_zips


def rq1_chart(scope: pd.DataFrame, metric: str, metric_label: str) -> tuple[go.Figure, html.Div]:
    data = scope.loc[
        scope["Housing_Data_Complete"].eq(True)
        & scope[metric].ge(0)
        & scope["Severe_Injuries"].ge(0)
    ].copy()
    stats = correlation_pair(data, metric, "Severe_Injuries")
    if data.empty:
        return empty_figure("No eligible RQ1 observations", "Broaden the selected geography or years."), summary_box(
            "0 eligible ZIP-year observations"
        )
    data["Year_Label"] = data["Year"].astype(int).astype(str)
    data["Ambiguity"] = np.where(data["Geographic_Ambiguity"], "Geographically ambiguous", "Unambiguous")
    figure = px.scatter(
        data,
        x=metric,
        y="Severe_Injuries",
        color="Year_Label",
        symbol="Ambiguity",
        color_discrete_map=YEAR_COLORS,
        custom_data=["ZIP_Code", "Year", "Borough", metric, "Severe_Injuries", "Fatalities", "Ambiguity"],
        labels={metric: metric_label, "Severe_Injuries": "Recorded severe injuries", "Year_Label": "Year"},
    )
    figure.update_traces(
        marker=dict(size=9, opacity=0.78, line=dict(width=0.7, color="#ffffff")),
        hovertemplate=(
            "<b>ZIP %{customdata[0]}</b> · %{customdata[2]}<br>"
            "Year: %{customdata[1]}<br>"
            f"{metric_label}: %{{customdata[3]:,.0f}}<br>"
            "Severe injuries: %{customdata[4]:,.0f}<br>"
            "Fatalities: %{customdata[5]:,.0f}<br>"
            "Geography: %{customdata[6]}<extra></extra>"
        ),
    )
    style_figure(figure, legend_title="Year / geography")
    ambiguity_count = int(data["Geographic_Ambiguity"].sum())
    summary = summary_box(
        f"{len(data):,} observations · {data['ZIP_Code'].nunique():,} ZIP codes",
        f"Pearson r = {format_coefficient(stats['pearson'])} (p {format_p_value(stats['pearson_p'])})",
        f"Spearman ρ = {format_coefficient(stats['spearman'])} (p {format_p_value(stats['spearman_p'])})",
        note=(
            f"Notebook method retained: complete housing observations, including {ambiguity_count} geographically "
            "ambiguous record(s) shown with a separate marker. Association is not causation."
        ),
    )
    return figure, summary


def rq2_chart(scope: pd.DataFrame, metric: str, metric_label: str) -> tuple[go.Figure, html.Div]:
    data = scope.loc[
        scope["Housing_Data_Complete"].eq(True)
        & scope["Geographic_Ambiguity"].eq(False)
        & scope[metric].ge(0)
        & scope["Severe_Injuries"].ge(0)
    ].copy()
    if data.empty:
        return empty_figure("No eligible RQ2 observations", "RQ2 excludes incomplete and geographically ambiguous rows."), summary_box(
            "0 eligible ZIP-year observations"
        )
    data["Log_Housing"] = np.log1p(data[metric])
    data["Year_Label"] = data["Year"].astype(int).astype(str)
    figure = px.scatter(
        data,
        x="Log_Housing",
        y="Severe_Injuries",
        color="Year_Label",
        facet_col="Borough",
        facet_col_wrap=3,
        color_discrete_map=YEAR_COLORS,
        custom_data=["ZIP_Code", "Year", "Borough", metric, "Severe_Injuries", "Fatalities"],
        labels={"Log_Housing": f"ln(1 + {metric_label})", "Severe_Injuries": "Recorded severe injuries", "Year_Label": "Year"},
    )
    figure.for_each_annotation(lambda annotation: annotation.update(text=annotation.text.split("=")[-1]))
    figure.update_traces(
        marker=dict(size=8, opacity=0.76, line=dict(width=0.5, color="#ffffff")),
        hovertemplate=(
            "<b>ZIP %{customdata[0]}</b> · %{customdata[2]}<br>"
            "Year: %{customdata[1]}<br>"
            f"{metric_label}: %{{customdata[3]:,.0f}}<br>"
            "Severe injuries: %{customdata[4]:,.0f}<br>Fatalities: %{customdata[5]:,.0f}<extra></extra>"
        ),
    )
    style_figure(figure, height=620, legend_title="Year")
    correlations: list[str] = []
    for borough, group in data.groupby("Borough", observed=True):
        coefficient, _ = spearman_only(group, metric, "Severe_Injuries", minimum=5)
        correlations.append(f"{borough}: ρ {format_coefficient(coefficient)} (n={len(group)})")
    return figure, summary_box(
        f"{len(data):,} unambiguous observations",
        " · ".join(correlations),
        note="Borough coefficients are recalculated only when at least five observations and variation in both measures are available.",
    )


def rq3_chart(scope: pd.DataFrame, metric: str, metric_label: str, years: list[int]) -> tuple[go.Figure, html.Div]:
    eligible = scope.loc[
        scope["Housing_Data_Complete"].eq(True) & scope["Geographic_Ambiguity"].eq(False)
    ].copy()
    if len(years) < 2:
        explanation = "Select at least two primary research years to construct a balanced ZIP-code panel."
        return empty_figure("Balanced panel unavailable", explanation), summary_box(explanation)
    panel, balanced_zips = build_balanced_panel(eligible, years)
    if not balanced_zips:
        explanation = "No ZIP code has an eligible observation in every selected year after geographic filtering."
        return empty_figure("Balanced panel unavailable", explanation), summary_box(explanation)
    annual = (
        panel.groupby("Year", as_index=False)
        .agg(Housing_Value=(metric, safe_sum), Severe_Injuries=("Severe_Injuries", safe_sum), ZIP_Codes=("ZIP_Code", "nunique"))
        .sort_values("Year")
    )
    if annual["Housing_Value"].isna().any() or annual["Severe_Injuries"].isna().any():
        explanation = "At least one selected year has only missing measurements; missing values were not converted to zero."
        return empty_figure("Indexed comparison unavailable", explanation), summary_box(explanation)
    baseline = annual.iloc[0]
    if baseline["Housing_Value"] <= 0 or baseline["Severe_Injuries"] <= 0:
        explanation = "The first selected year has a zero baseline, so a 100-based index cannot be calculated honestly."
        return empty_figure("Indexed comparison unavailable", explanation), summary_box(explanation)
    annual["Housing_Index"] = annual["Housing_Value"] / baseline["Housing_Value"] * 100
    annual["Injury_Index"] = annual["Severe_Injuries"] / baseline["Severe_Injuries"] * 100
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=annual["Year"],
            y=annual["Housing_Index"],
            mode="lines+markers",
            name=metric_label,
            line=dict(color="#2367d1", width=3),
            marker=dict(size=9),
            customdata=np.column_stack([annual["Housing_Value"], annual["ZIP_Codes"]]),
            hovertemplate=(
                "<b>%{x}</b><br>Housing index: %{y:.1f}<br>"
                f"{metric_label}: %{{customdata[0]:,.0f}}<br>"
                "Balanced ZIP codes: %{customdata[1]:,.0f}<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=annual["Year"],
            y=annual["Injury_Index"],
            mode="lines+markers",
            name="Severe injuries",
            line=dict(color="#d66a3a", width=3, dash="dash"),
            marker=dict(size=9, symbol="diamond"),
            customdata=np.column_stack([annual["Severe_Injuries"], annual["ZIP_Codes"]]),
            hovertemplate=(
                "<b>%{x}</b><br>Severe injury index: %{y:.1f}<br>"
                "Severe injuries: %{customdata[0]:,.0f}<br>"
                "Balanced ZIP codes: %{customdata[1]:,.0f}<extra></extra>"
            ),
        )
    )
    figure.add_hline(y=100, line_dash="dot", line_color="#8291a4")
    figure.update_xaxes(tickmode="array", tickvals=years, title="Year")
    figure.update_yaxes(title=f"Index ({years[0]} = 100)")
    style_figure(figure, legend_title="Measure")
    latest = annual.iloc[-1]
    return figure, summary_box(
        f"Balanced panel: {len(balanced_zips):,} ZIP codes across {len(years)} years",
        f"Latest housing index: {latest['Housing_Index']:.1f}",
        f"Latest severe injury index: {latest['Injury_Index']:.1f}",
        note=f"Both series use the same ZIP codes in every selected year; {years[0]} is the recalculated baseline.",
    )


def rq4_chart(scope: pd.DataFrame, metric: str, metric_label: str, years: list[int]) -> tuple[go.Figure, html.Div]:
    target_year = max(years)
    data = scope.loc[
        scope["Year"].eq(target_year)
        & scope["Housing_Data_Complete"].eq(True)
        & scope["Geographic_Ambiguity"].eq(False)
        & scope[metric].notna()
        & scope["Severe_Injuries"].notna()
    ].copy()
    if data.empty:
        explanation = f"No complete, unambiguous observations are available for {target_year}."
        return empty_figure("Hotspot classification unavailable", explanation), summary_box(explanation)
    housing_threshold = float(data[metric].median())
    injury_threshold = float(data["Severe_Injuries"].median())
    high_housing = data[metric].ge(housing_threshold)
    high_injuries = data["Severe_Injuries"].ge(injury_threshold)
    data["Category"] = np.select(
        [high_housing & high_injuries, high_housing & ~high_injuries, ~high_housing & high_injuries],
        ["High Housing / High Injuries", "High Housing / Low Injuries", "Low Housing / High Injuries"],
        default="Low Housing / Low Injuries",
    )
    data["Log_Housing"] = np.log1p(data[metric])
    figure = px.scatter(
        data,
        x="Log_Housing",
        y="Severe_Injuries",
        color="Category",
        color_discrete_map=CATEGORY_COLORS,
        custom_data=["ZIP_Code", "Year", "Borough", metric, "Severe_Injuries", "Fatalities", "Category"],
        labels={"Log_Housing": f"ln(1 + {metric_label})", "Severe_Injuries": "Recorded severe injuries"},
    )
    figure.update_traces(
        marker=dict(size=10, opacity=0.82, line=dict(width=0.7, color="#ffffff")),
        hovertemplate=(
            "<b>ZIP %{customdata[0]}</b> · %{customdata[2]}<br>Year: %{customdata[1]}<br>"
            f"{metric_label}: %{{customdata[3]:,.0f}}<br>"
            "Severe injuries: %{customdata[4]:,.0f}<br>Fatalities: %{customdata[5]:,.0f}<br>"
            "Classification: %{customdata[6]}<extra></extra>"
        ),
    )
    figure.add_vline(x=np.log1p(housing_threshold), line_dash="dash", line_color="#66778d")
    figure.add_hline(y=injury_threshold, line_dash="dash", line_color="#66778d")
    style_figure(figure, height=470, legend_title="Median-based classification")
    counts = data["Category"].value_counts()
    high_high = int(counts.get("High Housing / High Injuries", 0))
    return figure, summary_box(
        f"{target_year}: {len(data):,} eligible ZIP codes",
        f"Recalculated medians — housing: {housing_threshold:,.1f}; severe injuries: {injury_threshold:,.1f}",
        f"High Housing / High Injuries: {high_high:,} ZIP codes",
        note="Thresholds are relative to the current eligible sample and are not official risk standards.",
    )


def rq5_chart(scope: pd.DataFrame, metric: str, metric_label: str) -> tuple[go.Figure, html.Div]:
    data = scope.loc[
        scope["Housing_Data_Complete"].eq(True)
        & scope["Geographic_Ambiguity"].eq(False)
        & scope[metric].ge(0)
        & scope["Fatalities"].ge(0)
    ].copy()
    stats = correlation_pair(data, metric, "Fatalities")
    if data.empty:
        return empty_figure("No eligible RQ5 observations", "Broaden the selected geography or years."), summary_box(
            "0 eligible ZIP-year observations"
        )
    data["Log_Housing"] = np.log1p(data[metric])
    data["Year_Label"] = data["Year"].astype(int).astype(str)
    figure = px.scatter(
        data,
        x="Log_Housing",
        y="Fatalities",
        color="Year_Label",
        color_discrete_map=YEAR_COLORS,
        custom_data=["ZIP_Code", "Year", "Borough", metric, "Severe_Injuries", "Fatalities"],
        labels={"Log_Housing": f"ln(1 + {metric_label})", "Fatalities": "Recorded fatalities", "Year_Label": "Year"},
    )
    figure.update_traces(
        marker=dict(size=9, opacity=0.78, line=dict(width=0.6, color="#ffffff")),
        hovertemplate=(
            "<b>ZIP %{customdata[0]}</b> · %{customdata[2]}<br>Year: %{customdata[1]}<br>"
            f"{metric_label}: %{{customdata[3]:,.0f}}<br>"
            "Severe injuries: %{customdata[4]:,.0f}<br>Fatalities: %{customdata[5]:,.0f}<extra></extra>"
        ),
    )
    style_figure(figure, legend_title="Year")
    return figure, summary_box(
        f"{len(data):,} complete, unambiguous observations",
        f"Pearson r = {format_coefficient(stats['pearson'])} (p {format_p_value(stats['pearson_p'])})",
        f"Spearman ρ = {format_coefficient(stats['spearman'])} (p {format_p_value(stats['spearman_p'])})",
        note="Fatalities are recorded counts, not exposure-adjusted risk rates. Correlations are exploratory.",
    )


def rq6_chart(scope: pd.DataFrame, metric: str, metric_label: str) -> tuple[go.Figure, html.Div]:
    data = scope.loc[
        scope["Housing_Data_Complete"].eq(True)
        & scope["Geographic_Ambiguity"].eq(False)
        & scope[metric].notna()
        & scope["Severe_Injuries"].notna()
    ].copy()
    if len(data) < 3:
        explanation = "At least three eligible observations are required to recalculate Low, Medium, and High groups."
        return empty_figure("Production groups unavailable", explanation), summary_box(explanation)
    data["Housing_Group"] = pd.qcut(
        data[metric].rank(method="first"), q=3, labels=["Low", "Medium", "High"]
    ).astype("string")
    figure = px.box(
        data,
        x="Housing_Group",
        y="Severe_Injuries",
        color="Housing_Group",
        category_orders={"Housing_Group": ["Low", "Medium", "High"]},
        color_discrete_map={"Low": "#78909c", "Medium": "#d29b24", "High": "#2367d1"},
        points="all",
        custom_data=["ZIP_Code", "Year", "Borough", metric, "Severe_Injuries", "Fatalities"],
        labels={"Housing_Group": "Housing production group", "Severe_Injuries": "Recorded severe injuries"},
    )
    figure.update_traces(
        jitter=0.32,
        pointpos=0,
        marker=dict(size=5, opacity=0.45),
        hovertemplate=(
            "<b>ZIP %{customdata[0]}</b> · %{customdata[2]}<br>Year: %{customdata[1]}<br>"
            f"{metric_label}: %{{customdata[3]:,.0f}}<br>"
            "Severe injuries: %{customdata[4]:,.0f}<br>Fatalities: %{customdata[5]:,.0f}<extra></extra>"
        ),
    )
    style_figure(figure, legend_title="Recalculated group")
    medians = data.groupby("Housing_Group", observed=True)["Severe_Injuries"].median()
    return figure, summary_box(
        f"{len(data):,} observations split into sample-based tertiles",
        " · ".join(f"{group}: median {medians.get(group, np.nan):.1f}" for group in ["Low", "Medium", "High"]),
        note="Groups are recalculated after filtering using the notebook's rank-first tertile method; overlap is visible in the distributions.",
    )


def rq7_chart(scope: pd.DataFrame, construction: str) -> tuple[go.Figure, html.Div]:
    base = scope.loc[
        scope["Housing_Data_Complete"].eq(True) & scope["Geographic_Ambiguity"].eq(False)
    ].copy()
    selected_types = ["New Construction", "Preservation"] if construction == "All affordable housing" else [construction]
    frames: list[pd.DataFrame] = []
    summaries: list[str] = []
    for label in selected_types:
        column = HOUSING_METRICS[label][0]
        subset = base.loc[base[column].gt(0) & base["Severe_Injuries"].notna()].copy()
        coefficient, p_value = spearman_only(subset, column, "Severe_Injuries", minimum=5)
        summaries.append(
            f"{label}: ρ {format_coefficient(coefficient)} (p {format_p_value(p_value)}; n={len(subset)})"
        )
        if not subset.empty:
            subset["Construction_Type"] = label
            subset["Construction_Units"] = subset[column]
            subset["Log_Units"] = np.log1p(subset[column])
            subset["Year_Label"] = subset["Year"].astype(int).astype(str)
            frames.append(subset)
    if not frames:
        explanation = "No positive-unit observations remain for the selected construction type and scope."
        return empty_figure("Construction comparison unavailable", explanation), summary_box(explanation)
    data = pd.concat(frames, ignore_index=True)
    figure = px.scatter(
        data,
        x="Log_Units",
        y="Severe_Injuries",
        color="Year_Label",
        facet_col="Construction_Type" if len(selected_types) > 1 else None,
        color_discrete_map=YEAR_COLORS,
        custom_data=["ZIP_Code", "Year", "Borough", "Construction_Units", "Severe_Injuries", "Fatalities", "Construction_Type"],
        labels={"Log_Units": "ln(1 + counted units)", "Severe_Injuries": "Recorded severe injuries", "Year_Label": "Year"},
    )
    figure.for_each_annotation(lambda annotation: annotation.update(text=annotation.text.split("=")[-1]))
    figure.update_traces(
        marker=dict(size=8, opacity=0.75, line=dict(width=0.5, color="#ffffff")),
        hovertemplate=(
            "<b>ZIP %{customdata[0]}</b> · %{customdata[2]}<br>Year: %{customdata[1]}<br>"
            "Construction type: %{customdata[6]}<br>Counted units: %{customdata[3]:,.0f}<br>"
            "Severe injuries: %{customdata[4]:,.0f}<br>Fatalities: %{customdata[5]:,.0f}<extra></extra>"
        ),
    )
    style_figure(figure, height=460, legend_title="Year")
    return figure, summary_box(
        " · ".join(summaries),
        note="Only positive units in each construction category are included. When both are shown, subsets may overlap within a ZIP-year.",
    )


def rq8_chart(scope: pd.DataFrame, metric: str, metric_label: str, years: list[int]) -> tuple[go.Figure, html.Div]:
    eligible = scope.loc[
        scope["Housing_Data_Complete"].eq(True) & scope["Geographic_Ambiguity"].eq(False)
    ].copy()
    if len(years) < 2:
        explanation = "Select at least two years to calculate within-ZIP annual changes."
        return empty_figure("Change analysis unavailable", explanation), summary_box(explanation)
    panel, balanced_zips = build_balanced_panel(eligible, years)
    if not balanced_zips:
        explanation = "No ZIP code has eligible observations in every selected year after geographic filtering."
        return empty_figure("Change analysis unavailable", explanation), summary_box(explanation)
    panel = panel.sort_values(["ZIP_Code", "Year"]).copy()
    panel["Previous_Year"] = panel.groupby("ZIP_Code")["Year"].shift()
    panel["Previous_Housing"] = panel.groupby("ZIP_Code")[metric].shift()
    panel["Previous_Injuries"] = panel.groupby("ZIP_Code")["Severe_Injuries"].shift()
    panel["Housing_Change"] = panel[metric] - panel["Previous_Housing"]
    panel["Injury_Change"] = panel["Severe_Injuries"] - panel["Previous_Injuries"]
    changes = panel.loc[
        panel["Previous_Year"].notna()
        & panel["Year"].sub(panel["Previous_Year"]).eq(1)
        & panel["Housing_Change"].notna()
        & panel["Injury_Change"].notna()
    ].copy()
    if len(changes) < 3:
        explanation = "Fewer than three consecutive-year transitions remain. Non-consecutive selections are not relabeled as annual changes."
        return empty_figure("Change analysis unavailable", explanation), summary_box(explanation)
    coefficient, p_value = spearman_only(changes, "Housing_Change", "Injury_Change", minimum=3)
    changes["Transition"] = changes["Previous_Year"].astype(int).astype(str) + " → " + changes["Year"].astype(int).astype(str)
    figure = px.scatter(
        changes,
        x="Housing_Change",
        y="Injury_Change",
        color="Transition",
        custom_data=[
            "ZIP_Code",
            "Borough",
            "Previous_Year",
            "Year",
            "Previous_Housing",
            metric,
            "Housing_Change",
            "Previous_Injuries",
            "Severe_Injuries",
            "Injury_Change",
        ],
        color_discrete_sequence=["#2367d1", "#d66a3a", "#6f64c0"],
        labels={"Housing_Change": f"Annual change in {metric_label}", "Injury_Change": "Annual change in severe injuries"},
    )
    figure.update_traces(
        marker=dict(size=9, opacity=0.78, line=dict(width=0.6, color="#ffffff")),
        hovertemplate=(
            "<b>ZIP %{customdata[0]}</b> · %{customdata[1]}<br>"
            "Transition: %{customdata[2]:.0f} → %{customdata[3]:.0f}<br>"
            f"{metric_label}: %{{customdata[4]:,.0f}} → %{{customdata[5]:,.0f}} (%{{customdata[6]:+,.0f}})<br>"
            "Severe injuries: %{customdata[7]:,.0f} → %{customdata[8]:,.0f} (%{customdata[9]:+,.0f})<extra></extra>"
        ),
    )
    figure.add_vline(x=0, line_dash="dash", line_color="#8291a4")
    figure.add_hline(y=0, line_dash="dash", line_color="#8291a4")
    style_figure(figure, legend_title="Annual transition")
    return figure, summary_box(
        f"Balanced panel: {len(balanced_zips):,} ZIP codes · {len(changes):,} consecutive-year changes",
        f"Spearman ρ = {format_coefficient(coefficient)} (p {format_p_value(p_value)})",
        note="Changes are current year minus the immediately preceding year within the same ZIP code. Association is not causation.",
    )


def kpi_card(label: str, element_id: str, caption: str) -> html.Div:
    return html.Div(
        [html.P(label, className="kpi-label"), html.H3(id=element_id, className="kpi-value"), html.P(caption, className="kpi-caption")],
        className="kpi-card",
    )


def chart_card(rq: str, title: str, subtitle: str, graph_id: str, summary_id: str) -> html.Article:
    return html.Article(
        [
            html.Div(
                [html.Span(rq, className="rq-badge"), html.Div([html.H3(title), html.P(subtitle)])],
                className="chart-heading",
            ),
            dcc.Loading(dcc.Graph(id=graph_id, config={"displaylogo": False, "responsive": True}), type="circle"),
            html.Div(id=summary_id),
        ],
        className="chart-card",
    )


app = Dash(__name__, title="NYC Urban Growth & Road Safety", update_title="Generating report…")
server = app.server


app.layout = html.Div(
    [
        html.Header(
            [
                html.A(
                    [html.Span("NYC", className="brand-mark"), html.Span("Urban Growth & Road Safety")],
                    href="#overview",
                    className="brand",
                ),
                html.Nav(
                    [
                        html.A("Overview", href="#overview"),
                        html.A("Research", href="#research"),
                        html.A("Story", href="#story"),
                        html.A("Methodology", href="#methodology"),
                    ],
                    className="nav-links",
                    **{"aria-label": "Primary navigation"},
                ),
            ],
            className="topbar",
        ),
        html.Main(
            [
                html.Section(
                    [
                        html.Div(
                            [
                                html.P("GIU · Data Engineering & Visualization", className="eyebrow"),
                                html.H1("NYC Urban Growth & Road Safety"),
                                html.P(
                                    "Explore how affordable housing production and recorded serious traffic injuries intersect across NYC ZIP codes. The dashboard preserves the notebook's exploratory, non-causal methodology.",
                                    className="hero-copy",
                                ),
                                html.Div(
                                    [
                                        html.Span("Primary analysis · 2023–2025"),
                                        html.Span("367 matched ZIP-year records · 2022–2025 source coverage"),
                                    ],
                                    className="hero-meta",
                                ),
                            ],
                            className="hero-content",
                        ),
                        html.Div(
                            [
                                html.Span("HOUSING", className="orbit-label orbit-one"),
                                html.Span("ZIP + YEAR", className="orbit-core"),
                                html.Span("SAFETY", className="orbit-label orbit-two"),
                            ],
                            className="hero-visual",
                            **{"aria-hidden": "true"},
                        ),
                    ],
                    id="overview",
                    className="hero section-shell",
                ),
                html.Section(
                    [
                        html.Div(
                            [
                                html.P("Report builder", className="eyebrow dark"),
                                html.H2("Define the analytical scope"),
                                html.P(
                                    "Use guided controls or enter a compact query such as “Brooklyn 2024 new construction.” Charts only change when Generate Report is selected.",
                                    className="section-intro",
                                ),
                            ],
                            className="section-title-row",
                        ),
                        dcc.RadioItems(
                            id="report-mode",
                            options=[
                                {"label": "Guided filters", "value": "filters"},
                                {"label": "Text search", "value": "search"},
                            ],
                            value="filters",
                            inline=True,
                            className="mode-switch",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [html.Label("Borough", htmlFor="borough-filter"), dcc.Dropdown(
                                        id="borough-filter",
                                        options=[{"label": borough, "value": borough} for borough in BOROUGHS],
                                        value=BOROUGHS,
                                        multi=True,
                                        clearable=True,
                                        placeholder="All boroughs",
                                    )],
                                    className="control-group",
                                ),
                                html.Div(
                                    [html.Label("Research year", htmlFor="year-filter"), dcc.Dropdown(
                                        id="year-filter",
                                        options=[{"label": str(year), "value": year} for year in PRIMARY_YEARS],
                                        value=PRIMARY_YEARS,
                                        multi=True,
                                        clearable=True,
                                        placeholder="2023–2025",
                                    )],
                                    className="control-group",
                                ),
                                html.Div(
                                    [html.Label("Housing measure", htmlFor="construction-filter"), dcc.Dropdown(
                                        id="construction-filter",
                                        options=[{"label": label, "value": label} for label in HOUSING_METRICS],
                                        value="All affordable housing",
                                        clearable=False,
                                    )],
                                    className="control-group",
                                ),
                                html.Div(
                                    [html.Label("ZIP code", htmlFor="zip-filter"), dcc.Dropdown(
                                        id="zip-filter",
                                        options=[{"label": zip_code, "value": zip_code} for zip_code in ZIP_CODES],
                                        value=[],
                                        multi=True,
                                        clearable=True,
                                        placeholder="All ZIP codes",
                                    )],
                                    className="control-group",
                                ),
                            ],
                            className="control-grid",
                        ),
                        html.Div(
                            [
                                html.Label("Search the research data", htmlFor="search-query"),
                                dcc.Input(
                                    id="search-query",
                                    type="search",
                                    placeholder="Brooklyn 2024 new construction",
                                    debounce=False,
                                    className="search-input",
                                ),
                            ],
                            className="search-group",
                        ),
                        html.Button([html.Span("Generate Report"), html.Span("→", className="button-arrow")], id="generate-report", n_clicks=0, className="generate-button"),
                        html.Div(id="filter-feedback", className="filter-feedback", **{"aria-live": "polite"}),
                    ],
                    className="filter-panel section-shell",
                ),
                html.Section(
                    [
                        html.Div(
                            [
                                html.P("Current report", className="eyebrow dark"),
                                html.H2(id="report-title"),
                                html.P(id="report-subtitle", className="section-intro"),
                            ],
                            className="section-title-row",
                        ),
                        html.Div(
                            [
                                kpi_card("Eligible observations", "kpi-observations", "Complete + unambiguous ZIP-years"),
                                kpi_card("ZIP codes", "kpi-zips", "Distinct ZIPs in current scope"),
                                kpi_card("Housing units", "kpi-housing", "Selected housing measure"),
                                kpi_card("Severe injuries", "kpi-injuries", "Recorded SIRTA count"),
                                kpi_card("Fatalities", "kpi-fatalities", "Recorded SIRTA count"),
                            ],
                            className="kpi-grid",
                        ),
                    ],
                    className="report-overview section-shell",
                ),
                html.Section(
                    [
                        html.Div(
                            [
                                html.P("Eight research views", className="eyebrow dark"),
                                html.H2("Evidence by research question"),
                                html.P(
                                    "Hover over points for ZIP, year, housing, injury, and fatality details. Each summary reflects the generated scope—not the notebook's static baseline.",
                                    className="section-intro",
                                ),
                            ],
                            className="section-title-row",
                        ),
                        html.Div(
                            [
                                chart_card("RQ1", "Overall housing–injury association", "ZIP-year scatter with Pearson and Spearman statistics", "rq1-graph", "rq1-summary"),
                                chart_card("RQ2", "Borough-level differences", "Faceted relationship and borough-specific Spearman statistics", "rq2-graph", "rq2-summary"),
                                chart_card("RQ3", "Balanced-panel temporal patterns", "Indexed housing and severe injury totals for the same ZIP codes", "rq3-graph", "rq3-summary"),
                                chart_card("RQ4", "Housing and injury hotspots", "Median-based classification in the latest selected year", "rq4-graph", "rq4-summary"),
                                chart_card("RQ5", "Housing and traffic fatalities", "ZIP-year scatter with dynamically recalculated correlations", "rq5-graph", "rq5-summary"),
                                chart_card("RQ6", "Severe injuries by housing intensity", "Box plots using recalculated Low, Medium, and High tertiles", "rq6-graph", "rq6-summary"),
                                chart_card("RQ7", "Construction type differences", "Positive-unit observations for New Construction and Preservation", "rq7-graph", "rq7-summary"),
                                chart_card("RQ8", "Within-ZIP year-to-year changes", "Change-versus-change scatter from a recalculated balanced panel", "rq8-graph", "rq8-summary"),
                            ],
                            className="chart-grid",
                        ),
                    ],
                    id="research",
                    className="research-section section-shell",
                ),
                html.Section(
                    [
                        html.Div(
                            [
                                html.P("Research storytelling", className="eyebrow light"),
                                html.H2("What the project establishes—and what it does not"),
                            ],
                            className="story-heading",
                        ),
                        html.Div(
                            [
                                html.Article(
                                    [html.Span("01"), html.H3("Objective"), html.P(
                                        "Explore geographic and temporal associations between counted affordable housing production and recorded serious traffic injuries after integrating NYC housing and SIRTA data at ZIP-code/year grain."
                                    )],
                                    className="story-card",
                                ),
                                html.Article(
                                    [html.Span("02"), html.H3("Notebook baseline findings"), html.P(
                                        "For the unfiltered 2023–2025 baseline, linear association with severe injuries was near zero while the rank association was weakly positive. Fatalities and annual changes showed little association. Results varied by borough and construction type."
                                    )],
                                    className="story-card",
                                ),
                                html.Article(
                                    [html.Span("03"), html.H3("Limitations"), html.P(
                                        "SIRTA geographic coverage is incomplete; matched rows do not represent every neighborhood or crash. Counts are not adjusted for population or traffic exposure, repeated ZIP codes are not independent, and correlation does not establish causation."
                                    )],
                                    className="story-card",
                                ),
                                html.Article(
                                    [html.Span("04"), html.H3("Potential benefit to NYC"), html.P(
                                        "The integrated view can help identify ZIP codes for deeper, exposure-aware investigation, compare borough context, distinguish construction categories, and highlight where better geographic completeness would improve planning evidence."
                                    )],
                                    className="story-card",
                                ),
                            ],
                            className="story-grid",
                        ),
                    ],
                    id="story",
                    className="story-section",
                ),
                html.Section(
                    [
                        html.Div(
                            [html.P("Methodology & provenance", className="eyebrow dark"), html.H2("Designed to keep the research honest")],
                            className="section-title-row",
                        ),
                        html.Div(
                            [
                                html.Div([html.H3("Scope"), html.P("Primary RQ analysis uses 2023–2025. The integrated source also contains 2022 for data-engineering coverage context.")]),
                                html.Div([html.H3("Eligibility"), html.P("Incomplete housing totals are excluded. RQ2–RQ8 exclude geographic ambiguity; RQ1 retains it and marks it separately, matching the notebook sensitivity design.")]),
                                html.Div([html.H3("Missingness"), html.P("Missing or unmatched data are never converted to zero. Empty and statistically insufficient selections return an explanation instead of a fabricated chart.")]),
                                html.Div([html.H3("Dynamic methods"), html.P("Balanced panels, median thresholds, housing tertiles, and relevant correlations are recalculated from the current eligible observations.")]),
                            ],
                            className="method-grid",
                        ),
                        html.Div(
                            [
                                html.P("Sources"),
                                html.A("NYC Affordable Housing Production by Building", href="https://data.cityofnewyork.us/d/hg8x-zxpr", target="_blank"),
                                html.A("NYC SIRTA", href="https://data.cityofnewyork.us/d/xeqp-qz8h", target="_blank"),
                                html.Span("Processed sources: final_integrated_dataset.csv · analysis_dataset.csv"),
                            ],
                            className="source-strip",
                        ),
                    ],
                    id="methodology",
                    className="method-section section-shell",
                ),
            ]
        ),
        html.Footer([html.Span("NYC Urban Growth & Road Safety"), html.Span("GIU · Data Engineering & Visualization")]),
    ]
)


CALLBACK_OUTPUTS = [
    Output("filter-feedback", "children"),
    Output("report-title", "children"),
    Output("report-subtitle", "children"),
    Output("kpi-observations", "children"),
    Output("kpi-zips", "children"),
    Output("kpi-housing", "children"),
    Output("kpi-injuries", "children"),
    Output("kpi-fatalities", "children"),
]
for rq_number in range(1, 9):
    CALLBACK_OUTPUTS.extend(
        [Output(f"rq{rq_number}-graph", "figure"), Output(f"rq{rq_number}-summary", "children")]
    )


@app.callback(
    CALLBACK_OUTPUTS,
    Input("generate-report", "n_clicks"),
    State("borough-filter", "value"),
    State("year-filter", "value"),
    State("construction-filter", "value"),
    State("zip-filter", "value"),
    State("report-mode", "value"),
    State("search-query", "value"),
)
def generate_report(
    _n_clicks: int,
    boroughs: list[str] | None,
    years: list[int] | None,
    construction: str | None,
    zips: list[str] | None,
    mode: str,
    query: str | None,
) -> tuple[Any, ...]:
    filters = resolve_filters(boroughs, years, construction, zips, mode, query)
    metric, metric_label = HOUSING_METRICS[filters["construction"]]
    scope = apply_scope(FINAL_DF, filters)
    eligible = scope.loc[
        scope["Housing_Data_Complete"].eq(True) & scope["Geographic_Ambiguity"].eq(False)
    ].copy()

    borough_label = ", ".join(filters["boroughs"]) if len(filters["boroughs"]) < len(BOROUGHS) else "All boroughs"
    zip_label = f"{len(filters['zips'])} selected ZIP code(s)" if filters["zips"] else "All ZIP codes"
    year_label = ", ".join(str(year) for year in filters["years"])
    incomplete_count = int(scope["Housing_Data_Complete"].eq(False).sum())
    ambiguity_count = int(scope["Geographic_Ambiguity"].eq(True).sum())
    mode_label = "Text search" if mode == "search" else "Guided filters"
    feedback = [
        html.Strong(f"{mode_label} applied: "),
        html.Span(f"{borough_label} · {year_label} · {filters['construction']} · {zip_label}. "),
        html.Span(
            f"Current scope contains {incomplete_count} incomplete and {ambiguity_count} geographically ambiguous row(s); each RQ applies its documented eligibility rules."
        ),
    ]
    report_title = f"{filters['construction']} · {year_label}"
    report_subtitle = (
        f"{borough_label}; {zip_label}. KPI totals use complete, geographically unambiguous rows. "
        "RQ-specific summaries below may use a different documented eligibility population."
    )

    kpis = [
        format_integer(len(eligible)),
        format_integer(eligible["ZIP_Code"].nunique()),
        format_integer(safe_sum(eligible[metric])),
        format_integer(safe_sum(eligible["Severe_Injuries"])),
        format_integer(safe_sum(eligible["Fatalities"])),
    ]

    charts = [
        rq1_chart(scope, metric, metric_label),
        rq2_chart(scope, metric, metric_label),
        rq3_chart(scope, metric, metric_label, filters["years"]),
        rq4_chart(scope, metric, metric_label, filters["years"]),
        rq5_chart(scope, metric, metric_label),
        rq6_chart(scope, metric, metric_label),
        rq7_chart(scope, filters["construction"]),
        rq8_chart(scope, metric, metric_label, filters["years"]),
    ]
    flattened_charts: list[Any] = []
    for figure, summary in charts:
        flattened_charts.extend([figure, summary])
    return tuple([feedback, report_title, report_subtitle, *kpis, *flattened_charts])


if __name__ == "__main__":
    app.run(debug=True)
