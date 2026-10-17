"""Headless PNG charts for the containment experiments.

Palette and chrome follow the dataviz skill's reference instance
(validated categorical order, light surface). Series colors are assigned
in fixed slot order — never cycled, never repainted when the series set
changes.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# categorical slots 1-6, fixed order (validated for the adjacent pairlist)
SERIES_COLORS = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
]

CHROME = {
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
}


def _cumulative_padded(series, max_len):
    """Cumulative reached counts, holding the final value past the end.

    Shorter cascades keep their final cumulative value so all series align
    to the same step axis.
    """
    out = []
    running = 0
    for step in range(max_len):
        if step < len(series):
            running += series[step]
        out.append(running)
    return out


def _style_axes(ax, title, xlabel, ylabel):
    fig = ax.figure
    fig.patch.set_facecolor(CHROME["surface"])
    ax.set_facecolor(CHROME["surface"])
    ax.set_title(title, color=CHROME["ink"], fontsize=12)
    ax.set_xlabel(xlabel, color=CHROME["secondary"])
    ax.set_ylabel(ylabel, color=CHROME["secondary"])
    ax.tick_params(colors=CHROME["muted"])
    for spine in ax.spines.values():
        spine.set_color(CHROME["axis"])
    ax.grid(True, color=CHROME["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def plot_reach_curves(
    curves,
    out_path,
    title,
    xlabel="time step",
    ylabel="mean cumulative users reached",
    max_steps=None,
):
    """Plot mean cumulative reach with a +-1 std band per curve.

    `curves` is a list of {"label": str, "series": list[list[int]]} where
    each series is a cascade's newly-reached-per-step counts. One entry per
    strategy; the order defines the color slots.
    """
    all_series = [s for curve in curves for s in curve["series"]]
    max_len = max(len(s) for s in all_series) if all_series else 1
    if max_steps is not None:
        max_len = min(max_len, max_steps)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    _style_axes(ax, title, xlabel, ylabel)

    for index, curve in enumerate(curves):
        color = SERIES_COLORS[index % len(SERIES_COLORS)]
        padded = [_cumulative_padded(s, max_len) for s in curve["series"]]
        means = np.mean(padded, axis=0)
        stds = np.std(padded, axis=0)
        steps = np.arange(max_len)
        ax.plot(steps, means, color=color, linewidth=2, label=curve["label"])
        ax.fill_between(
            steps, means - stds, means + stds, color=color, alpha=0.15, linewidth=0
        )

    # a single series needs no legend box — the title names it; up to four
    # series also get direct end labels so identity is never color alone
    if 1 < len(curves) <= 4:
        for index, curve in enumerate(curves):
            color = SERIES_COLORS[index % len(SERIES_COLORS)]
            padded = [_cumulative_padded(s, max_len) for s in curve["series"]]
            means = np.mean(padded, axis=0)
            ax.annotate(
                curve["label"],
                xy=(max_len - 1, means[-1]),
                xytext=(5, 0),
                textcoords="offset points",
                color=CHROME["secondary"],
                fontsize=9,
                va="center",
            )
    if len(curves) > 1:
        ax.legend(frameon=False, labelcolor=CHROME["ink"], fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=CHROME["surface"])
    plt.close(fig)


def plot_sweep_lines(x_values, x_label, series_by_strategy, out_path, title):
    """Sensitivity sweep: mean metric per strategy across swept values.

    series_by_strategy is a dict label -> {"mean": [..], "std": [..]}
    with one entry per x value, in x_values order. Colors come from the
    fixed categorical slots, so a strategy keeps its color across charts.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))
    _style_axes(ax, title, x_label, "mean final fraction reached (fake posts)")
    for index, (label, series) in enumerate(series_by_strategy.items()):
        color = SERIES_COLORS[index % len(SERIES_COLORS)]
        means = series["mean"]
        stds = series.get("std", [0.0] * len(means))
        ax.errorbar(
            x_values,
            means,
            yerr=stds,
            color=color,
            linewidth=2,
            marker="o",
            markersize=5,
            capsize=3,
            label=label,
        )
    if len(series_by_strategy) > 1:
        ax.legend(frameon=False, labelcolor=CHROME["ink"], fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=CHROME["surface"])
    plt.close(fig)


def plot_fp_cost_bar(costs, out_path):
    """False-positive cost per strategy: two stacked panels, one figure.

    `costs` is a list of {"label": str, "real_reach_lost_mean": float,
    "fp_count": int} in strategy order. Real-news reach lost and the
    number of wrongly flagged real posts are different measures, so each
    gets its own panel (no dual axis).
    """
    labels = [cost["label"] for cost in costs]
    lost = [cost["real_reach_lost_mean"] for cost in costs]
    counts = [cost["fp_count"] for cost in costs]
    x = np.arange(len(costs))

    fig, (ax_lost, ax_count) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True
    )
    _style_axes(ax_lost, "False-positive cost of containment", "strategy", "mean real-news reach lost (users)")
    _style_axes(ax_count, "", "strategy", "real posts wrongly flagged")

    for index in range(len(costs)):
        color = SERIES_COLORS[index % len(SERIES_COLORS)]
        ax_lost.bar(x[index], lost[index], width=0.6, color=color)
        ax_count.bar(x[index], counts[index], width=0.6, color=color)

    ax_lost.set_xticks(x)
    ax_lost.set_xticklabels(labels, rotation=30, ha="right", color=CHROME["secondary"])
    ax_count.set_xticks(x)
    ax_count.set_xticklabels(labels, rotation=30, ha="right", color=CHROME["secondary"])
    # baseline by definition has no false-positive cost; show the honest zero
    ax_lost.axhline(0, color=CHROME["axis"], linewidth=1)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=CHROME["surface"])
    plt.close(fig)
