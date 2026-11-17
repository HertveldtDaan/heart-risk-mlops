"""Charts for the evaluation step, logged to Azure ML (job -> Images tab) with mlflow.log_figure."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # no display on a compute cluster
import matplotlib.pyplot as plt  # noqa: E402

BLUE = "#2563eb"
GREY = "#6b7280"


def fairness_chart(fairness: dict, overall_recall: float, threshold: float):
    """Recall per group for every audited attribute, with the overall recall as reference line.

    Recall = which share of the people who really had a heart attack the model flags.
    A big gap between groups means the model misses more cases in one group.
    """
    names = list(fairness)
    fig, axes = plt.subplots(1, len(names), figsize=(5.2 * len(names), 4.6), sharey=True)
    for ax, name in zip(axes, names):
        groups = [g for g in fairness[name]["groups"] if g["group"] != "Unknown"]
        if name == "age_group":  # natural order instead of by group size
            groups.sort(key=lambda g: g["group"])
        labels = [g["group"].replace(", Non-Hispanic", "").replace(" only", "") for g in groups]
        recalls = [g["recall"] for g in groups]
        bars = ax.bar(labels, recalls, color=BLUE, width=0.6)
        ax.axhline(overall_recall, color=GREY, linestyle="--", linewidth=1.2,
                   label=f"overall {overall_recall:.2f}")
        for bar, g in zip(bars, groups):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                    f"{g['recall']:.2f}\nn={g['count']:,}", ha="center", va="bottom", fontsize=8)
        ax.set_title(f"{name.replace('_', ' ')}  (gap {fairness[name]['recall_difference']:.2f})")
        ax.set_ylim(0, 1.1)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper left", fontsize=8, frameon=False)
    axes[0].set_ylabel("Recall (share of heart attacks detected)")
    fig.suptitle(f"Fairness audit on the test set - recall per group (threshold {threshold})", fontsize=12)
    fig.tight_layout()
    return fig


def shap_importance_chart(importance: dict[str, float], top_n: int = 15):
    """Global feature importance: mean |SHAP value| per question (log-odds)."""
    items = list(importance.items())[:top_n][::-1]
    fig, ax = plt.subplots(figsize=(8, 0.38 * len(items) + 1.4))
    ax.barh([k for k, _ in items], [v for _, v in items], color=BLUE)
    for i, (_, v) in enumerate(items):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlabel("mean |SHAP value| (impact on log-odds)")
    ax.set_title(f"Global feature importance (SHAP) - top {len(items)}")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig
