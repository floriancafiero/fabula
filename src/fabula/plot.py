from __future__ import annotations

from typing import Optional, Tuple

from .schemas import ArcResult


def plot_arc(
    arc: ArcResult,
    *,
    title: str = "Fabula arc",
    subtitle: Optional[str] = None,
    xlabel: str = "Relative position",
    ylabel: str = "Score",
    figure_size: Tuple[float, float] = (8.0, 4.5),
    line_color: str = "#2E6F9E",
    fill_color: str = "#D5E5F2",
    raw_points: bool = False,
    raw_point_color: str = "#4D4D4D",
    zero_line: bool = True,
    save_path: Optional[str] = None,
    show: bool = False,
):
    """Plot a narrative arc with publication-friendly defaults.

    Parameters
    ----------
    arc:
        Narrative arc output from :meth:`Fabula.arc`.
    title, subtitle:
        Title text displayed at the top of the chart.
    xlabel, ylabel:
        Axis labels.
    figure_size:
        Figure size in inches.
    line_color, fill_color:
        Colors used for the curve and the filled area.
    raw_points:
        When True, plot the raw segment scores as points.
    zero_line:
        Whether to draw a horizontal zero baseline.
    save_path:
        When provided, save the plot to this path.
    show:
        Whether to display the plot interactively.

    Returns
    -------
    (fig, ax)
        Matplotlib figure and axes.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figure_size)
    ax.plot(arc.x, arc.y, color=line_color, linewidth=2.5)
    ax.fill_between(arc.x, arc.y, 0.0, color=fill_color, alpha=0.75)

    if raw_points and arc.raw_x and arc.raw_y:
        ax.scatter(
            arc.raw_x,
            arc.raw_y,
            color=raw_point_color,
            s=20,
            alpha=0.7,
            zorder=3,
            label="segments",
        )

    if zero_line:
        ax.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--", alpha=0.6)

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")

    if subtitle:
        ax.text(
            0.0,
            1.02,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10,
            color="#4D4D4D",
        )

    ax.grid(axis="y", alpha=0.25, linestyle="-")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if raw_points:
        ax.legend(frameon=False, loc="upper right")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax
