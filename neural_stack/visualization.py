"""Visualization utilities for attention mechanisms."""

import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Union


def plot_attention_heatmap(
    attention_scores: Union[torch.Tensor, np.ndarray],
    batch_idx: int = 0
) -> go.Figure:
    """Plot attention heatmap with aggregated and individual head views.

    Creates a two-row visualization:
    - Top row: Average attention across all heads
    - Bottom rows: Individual attention patterns for each head

    Args:
        attention_scores: Attention weights of shape [B, num_heads, T_q, T_kv]
        batch_idx: Which batch sample to visualize. Default: 0.

    Returns:
        Plotly figure object with interactive heatmaps.
    """
    # Convert to numpy if needed
    if isinstance(attention_scores, torch.Tensor):
        attention_scores = attention_scores.detach().cpu().numpy()

    def head_to_row(head_idx: int) -> int:
        return 2 + head_idx // 4
    
    def head_to_col(head_idx: int) -> int:
        return 1 + head_idx % 4

    # Extract single batch sample
    attn = attention_scores[batch_idx]  # [num_heads, T_q, T_kv]
    num_heads, seq_len_q, seq_len_k = attn.shape

    # Compute average attention across all heads
    avg_attn = attn.mean(axis=0)  # [T_q, T_kv]

    # Create subplot structure: 2 rows x 4 cols
    num_rows = 1 + (num_heads // 4 + (1 if num_heads % 4 != 0 else 0))
    # Row 1: Single large heatmap spanning all columns
    # Row 2: 4 individual head heatmaps
    fig = make_subplots(
        rows=num_rows,
        cols=4,
        subplot_titles=(
            "Average Across All Heads",
            "",
            "",
            "",
            *(f"Head {head_idx}" for head_idx in range(num_heads))
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.08,
        specs=[
            [{"colspan": 4}, None, None, None],  # Row 1: single plot spanning 4 cols
            *([{}, {}, {}, {}] for _ in range(num_rows - 1))  # Row 2+: 4 individual plots
        ]
    )

    # Add aggregated heatmap (top row)
    fig.add_trace(
        go.Heatmap(
            z=avg_attn,
            colorscale="Plasma",
            zmin=0,
            zmax=1,
            hovertemplate="Query: %{y}<br>Key: %{x}<br>Attention: %{z:.3f}<extra></extra>",
            showscale=True,
            colorbar=dict(
                len=0.4,
                y=0.75,
                title="Weight"
            )
        ),
        row=1,
        col=1
    )

    # Add individual head heatmaps (bottom row)
    for head_idx in range(num_heads):
        fig.add_trace(
            go.Heatmap(
                z=attn[head_idx],
                colorscale="Plasma",
                zmin=0,
                zmax=1,
                hovertemplate="Query: %{y}<br>Key: %{x}<br>Attention: %{z:.3f}<extra></extra>",
                showscale=False  # Only show colorbar for aggregated plot
            ),
            row=head_to_row(head_idx),
            col=head_to_col(head_idx)
        )

    # Update axes for all subplots
    fig.update_xaxes(title_text="Key Position", row=1, col=1)
    fig.update_yaxes(title_text="Query Position", row=1, col=1, autorange="reversed")

    for head_idx in range(num_heads):
        fig.update_xaxes(title_text="Key Position", row=head_to_row(head_idx), col=head_to_col(head_idx))
        fig.update_yaxes(title_text="Query Position", row=head_to_row(head_idx), col=head_to_col(head_idx), autorange="reversed")

    # Update overall layout
    fig.update_layout(
        title_text="Attention Patterns",
        title_x=0.5,
        title_font_size=20,
        height=700,
        width=1200,
        showlegend=False
    )

    # Ensure square aspect ratio for all heatmaps
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    return fig
