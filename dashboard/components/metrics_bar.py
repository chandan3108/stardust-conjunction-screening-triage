"""
metrics_bar.py — Top KPI Metrics Row
"""

import streamlit as st
import pandas as pd


def render_metrics_bar(
    df: pd.DataFrame,
    threshold: float = 0.52,
):
    """Render minimalist KPI metric cards with clean, non-truncated labels."""
    n_flagged = int((df['ml_score'] >= threshold).sum())
    n_critical = int((df['pc_chan'] > 1e-4).sum())
    min_miss = df['miss_distance_m'].min()
    next_tca = df['tca_hours'].min()

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Active Pairs",
        f"{len(df)}",
        delta="Live Catalog",
        delta_color="off",
    )

    c2.metric(
        "ML Flagged",
        f"{n_flagged}",
        delta="Filtered 98%",
        delta_color="normal",
    )

    c3.metric(
        "Critical Alerts",
        f"{n_critical}",
        delta="Action Required" if n_critical > 0 else "Nominal",
        delta_color="inverse" if n_critical > 0 else "normal",
    )

    c4.metric(
        "Min Miss Dist",
        f"{min_miss:.1f} m",
        delta="Near Miss" if min_miss < 50 else "Safe",
        delta_color="inverse" if min_miss < 50 else "normal",
    )

    c5.metric(
        "Next TCA",
        f"{next_tca:.1f} hrs",
        delta="Horizon 6h",
        delta_color="off",
    )
