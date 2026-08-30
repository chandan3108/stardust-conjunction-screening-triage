"""
funnel_chart.py — Dynamic Pipeline Funnel Visualization

Renders a logarithmic-scaled pipeline reduction chart so all stages
from 450,000 pairs down to 3 critical alerts remain clearly visible
and readable without collapsing into 0-pixel lines.
"""

import numpy as np
import plotly.graph_objects as go
from typing import List, Optional


def render_funnel_chart(
    threshold: float = 0.52,
    total_pairs: int = 450000,
    moid_pairs: int = 52000,
    propagated_pairs: int = 160,
    n_flagged: int = 7,
    n_critical: int = 3,
):
    """
    Render sleek, clearly visible step-down reduction chart.
    """
    stages = [
        "1. Catalog Object Pairs",
        "2. MOID Geometric Filter",
        "3. SGP4 Propagated Window",
        f"4. ML Triage (Tau={threshold:.2f})",
        "5. Critical Alerts (Pc>1e-4)"
    ]

    actual_values = [
        total_pairs,
        moid_pairs,
        propagated_pairs,
        max(1, n_flagged),
        max(1, n_critical),
    ]

    # Formatted labels for display
    display_texts = [
        f"450,000 pairs (100%)",
        f"52,000 pairs (11.6%)",
        f"{propagated_pairs:,} active encounters",
        f"{n_flagged} flagged ({n_flagged/max(1, propagated_pairs)*100:.1f}%)",
        f"{n_critical} critical threats ({n_critical/max(1, n_flagged)*100:.1f}%)"
    ]

    # Log10-scaled widths so 3 alerts and 450,000 pairs are both clearly visible
    log_widths = [np.log10(max(1.5, v)) for v in actual_values]

    colors = ["#1E293B", "#334155", "#0284C7", "#F59E0B", "#EF4444"]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=stages[::-1],
        x=log_widths[::-1],
        orientation='h',
        marker=dict(
            color=colors[::-1],
            line=dict(width=1, color="rgba(255,255,255,0.15)")
        ),
        text=display_texts[::-1],
        textposition='inside',
        insidetextanchor='start',
        textfont=dict(family="JetBrains Mono", size=11, color="#F8FAFC"),
    ))

    fig.update_layout(
        xaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(family="Inter", size=11, color="#CBD5E1"),
        ),
        margin=dict(l=10, r=10, t=10, b=10),
        height=320,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )

    return fig
