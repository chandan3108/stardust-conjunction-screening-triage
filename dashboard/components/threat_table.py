"""
threat_table.py — Interactive CDM Table Component

Renders the conjunction data message table with threat scores,
miss distances, and actionable status indicators.
"""

import streamlit as st
import pandas as pd
from typing import Optional


def render_threat_table(
    df: pd.DataFrame,
    threshold: float = 0.08,
):
    """
    Render the interactive CDM threat table.

    Args:
        df: DataFrame with conjunction screening results
        threshold: ML decision threshold for flagging
    """
    display_df = df.copy()

    # Add action-required column
    display_df['Action Required'] = display_df['ml_score'] >= threshold

    # Sort by threat score descending
    display_df = display_df.sort_values('ml_score', ascending=False)

    st.dataframe(
        display_df,
        column_config={
            "event_id": st.column_config.TextColumn(
                "Event ID", width="medium"
            ),
            "primary": st.column_config.TextColumn(
                "Primary Asset", width="medium"
            ),
            "secondary": st.column_config.TextColumn(
                "Debris Object", width="medium"
            ),
            "tca_hours": st.column_config.NumberColumn(
                "TCA (hours)", format="%.1f"
            ),
            "miss_distance_m": st.column_config.NumberColumn(
                "Miss Dist (m)", format="%.1f"
            ),
            "rel_velocity_kms": st.column_config.NumberColumn(
                "Rel Vel (km/s)", format="%.1f"
            ),
            "pc_chan": st.column_config.NumberColumn(
                "Chan Pc", format="%.2e"
            ),
            "ml_score": st.column_config.ProgressColumn(
                "ML Threat Score",
                format="%.3f",
                min_value=0.0,
                max_value=1.0,
            ),
            "moid_km": st.column_config.NumberColumn(
                "MOID (km)", format="%.2f"
            ),
            "Action Required": st.column_config.CheckboxColumn(
                "Action Required"
            ),
        },
        use_container_width=True,
        hide_index=True,
        height=400,
    )
