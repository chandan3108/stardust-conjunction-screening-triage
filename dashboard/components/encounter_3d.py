"""
encounter_3d.py — 3D Encounter Geometry Visualization

Clean, modern 3D orbital conjunction visualizer for ISRO Mission Control.
Shows Primary Satellite, Debris Target, Encounter Vector,
1-sigma / 3-sigma Covariance Ellipsoids, and Hard-Body Collision Boundary.
"""

import numpy as np
import plotly.graph_objects as go
from typing import Dict, Optional


def render_encounter_3d(
    primary_name: str = "EOS-06",
    secondary_name: str = "FENGYUN-1C-DEB-51923",
    miss_distance_m: float = 18.1,
    sigma_r: float = 45.0,
    sigma_i: float = 180.0,
    sigma_c: float = 60.0,
    show_3sigma: bool = True,
    show_1sigma: bool = True,
    show_hbr: bool = True,
    camera_view: str = "Perspective",
    **kwargs
) -> go.Figure:
    """
    Render sleek 3D encounter geometry with configurable visual layers.

    Args:
        primary_name: Name of primary satellite
        secondary_name: Name of debris object
        miss_distance_m: Miss distance in meters
        sigma_r: Radial 3σ uncertainty (meters)
        sigma_i: In-track 3σ uncertainty (meters)
        sigma_c: Cross-track 3σ uncertainty (meters)
        show_3sigma: Toggle 3-sigma outer cloud
        show_1sigma: Toggle 1-sigma core zone
        show_hbr: Toggle 10m Hard-Body Radius envelope
        camera_view: 'Perspective', 'B-Plane (Frontal)', or 'Overhead (RIC)'

    Returns:
        Plotly Figure object
    """
    # Grid for covariance ellipsoid
    u = np.linspace(0, 2 * np.pi, 28)
    v = np.linspace(0, np.pi, 18)

    # 3-sigma Ellipsoid coordinates (centered at origin)
    x_3s = sigma_r * np.outer(np.cos(u), np.sin(v))
    y_3s = sigma_i * np.outer(np.sin(u), np.sin(v))
    z_3s = sigma_c * np.outer(np.ones(np.size(u)), np.cos(v))

    fig = go.Figure()

    # 1. 3σ Covariance Outer Cloud
    if show_3sigma:
        fig.add_trace(go.Surface(
            x=x_3s, y=y_3s, z=z_3s,
            opacity=0.12,
            colorscale=[[0, '#EF4444'], [1, '#DC2626']],
            showscale=False,
            name="3σ Uncertainty Boundary (99.7%)",
            hoverinfo='name',
        ))

    # 2. 1σ Covariance Core Zone
    if show_1sigma:
        fig.add_trace(go.Surface(
            x=x_3s / 3.0, y=y_3s / 3.0, z=z_3s / 3.0,
            opacity=0.28,
            colorscale=[[0, '#F59E0B'], [1, '#D97706']],
            showscale=False,
            name="1σ High-Risk Core (68.3%)",
            hoverinfo='name',
        ))

    # 3. 10m Hard-Body Radius (HBR) Sphere
    if show_hbr:
        r_hbr = 10.0
        x_hbr = r_hbr * np.outer(np.cos(u), np.sin(v))
        y_hbr = r_hbr * np.outer(np.sin(u), np.sin(v))
        z_hbr = r_hbr * np.outer(np.ones(np.size(u)), np.cos(v))

        fig.add_trace(go.Surface(
            x=x_hbr, y=y_hbr, z=z_hbr,
            opacity=0.45,
            colorscale=[[0, '#38BDF8'], [1, '#0284C7']],
            showscale=False,
            name="10m Hard-Body Physical Envelope",
            hoverinfo='name',
        ))

    # 4. Primary Satellite (Origin: [0, 0, 0])
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers+text',
        marker=dict(size=10, color='#38BDF8', symbol='diamond',
                    line=dict(color='#FFFFFF', width=1.5)),
        text=[f"  <b>{primary_name}</b> (ISRO)"],
        textposition='top right',
        textfont=dict(size=12, color='#F8FAFC', family='Inter'),
        name=f"Primary: {primary_name}",
    ))

    # 5. Primary Orbital Velocity Vector (+In-Track)
    traj_len = sigma_i * 1.25
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[-traj_len, traj_len], z=[0, 0],
        mode='lines',
        line=dict(color='#38BDF8', width=2.5, dash='dot'),
        name="Primary Velocity Vector (+V)",
        hoverinfo='name',
    ))

    # 6. Debris Relative Offset Calculation at TCA
    angle = np.radians(38)
    dx = float(miss_distance_m * np.cos(angle) * 0.42)
    dy = float(miss_distance_m * np.sin(angle) * 0.82)
    dz = float(miss_distance_m * 0.28)

    # Debris Trajectory Vector (relative approach line)
    deb_traj_len = 2.2
    fig.add_trace(go.Scatter3d(
        x=[dx - dx * deb_traj_len, dx + dx * deb_traj_len],
        y=[dy + dy * deb_traj_len, dy - dy * deb_traj_len],
        z=[dz - dz * deb_traj_len, dz + dz * deb_traj_len],
        mode='lines',
        line=dict(color='#EF4444', width=2, dash='dash'),
        name="Debris Relative Flight Path",
        hoverinfo='name',
    ))

    # Debris Position Marker
    fig.add_trace(go.Scatter3d(
        x=[dx], y=[dy], z=[dz],
        mode='markers+text',
        marker=dict(size=8.5, color='#EF4444', symbol='circle',
                    line=dict(color='#FFA4A4', width=1.5)),
        text=[f"  <b>{secondary_name}</b>"],
        textposition='bottom right',
        textfont=dict(size=11, color='#FCA5A5', family='Inter'),
        name=f"Debris: {secondary_name}",
    ))

    # 7. Miss Distance Vector Line
    fig.add_trace(go.Scatter3d(
        x=[0, dx], y=[0, dy], z=[0, dz],
        mode='lines+text',
        line=dict(color='#FCD34D', width=4),
        text=["", f"<b>MISS: {miss_distance_m:.1f} m</b>"],
        textposition="top center",
        textfont=dict(size=12, color='#FDE047', family='JetBrains Mono'),
        name=f"Miss Vector: {miss_distance_m:.1f} m",
    ))

    # Camera Perspective selection
    if camera_view == "B-Plane (Frontal)":
        camera = dict(eye=dict(x=0.0, y=-2.0, z=0.0), up=dict(x=0, y=0, z=1))
    elif camera_view == "Overhead (RIC)":
        camera = dict(eye=dict(x=0.0, y=0.0, z=2.2), up=dict(x=0, y=1, z=0))
    else:  # Perspective
        camera = dict(eye=dict(x=1.35, y=-1.45, z=0.95), up=dict(x=0, y=0, z=1))

    # Dynamic Scene Boundaries
    max_bound = max(sigma_r, sigma_i, sigma_c) * 1.2
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title=dict(text="Radial (m)", font=dict(size=10, color='#94A3B8')),
                range=[-max_bound * 0.7, max_bound * 0.7],
                backgroundcolor='#080B10',
                gridcolor='#1E293B',
                showbackground=True,
                zerolinecolor='#334155',
            ),
            yaxis=dict(
                title=dict(text="In-Track (m)", font=dict(size=10, color='#94A3B8')),
                range=[-max_bound, max_bound],
                backgroundcolor='#080B10',
                gridcolor='#1E293B',
                showbackground=True,
                zerolinecolor='#334155',
            ),
            zaxis=dict(
                title=dict(text="Cross-Track (m)", font=dict(size=10, color='#94A3B8')),
                range=[-max_bound * 0.7, max_bound * 0.7],
                backgroundcolor='#080B10',
                gridcolor='#1E293B',
                showbackground=True,
                zerolinecolor='#334155',
            ),
            camera=camera,
            aspectmode='data',
        ),
        margin=dict(l=0, r=0, t=10, b=10),
        height=520,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=0.02,
            xanchor="center", x=0.5,
            font=dict(size=10, color='#94A3B8'),
            bgcolor='rgba(13, 17, 23, 0.8)',
            bordercolor='#1E293B',
            borderwidth=1,
        ),
        paper_bgcolor='rgba(0,0,0,0)',
    )

    return fig
