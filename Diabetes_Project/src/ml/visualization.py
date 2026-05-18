from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/diabetes_project_mpl")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_parent_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_class_distribution_chart(target: pd.Series, output_path: Path) -> None:
    ensure_parent_directory(output_path)
    counts = target.value_counts().sort_index()

    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(
        ["Non-Diabetic", "Diabetic"],
        counts.tolist(),
        color=["#7FB3D5", "#E67E22"],
    )
    plt.title("Class Distribution")
    plt.ylabel("Number of Records")
    for bar, count in zip(bars, counts.tolist()):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{count:,}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_confusion_matrix_heatmap(
    confusion_matrix_data: list[list[int]],
    output_path: Path,
    labels: tuple[str, str] = ("Non-Diabetic", "Diabetic"),
) -> None:
    ensure_parent_directory(output_path)
    matrix = np.array(confusion_matrix_data)

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks([0, 1], labels=labels)
    ax.set_yticks([0, 1], labels=labels)

    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(
                column,
                row,
                f"{matrix[row, column]}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_roc_curve_plot(y_true: pd.Series, y_prob: Iterable[float], output_path: Path) -> None:
    from sklearn.metrics import RocCurveDisplay

    ensure_parent_directory(output_path)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax, color="#1F618D")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#A6ACAF", linewidth=1)
    ax.set_title("ROC Curve")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_feature_importance_bar_chart(
    feature_importance_items: list[dict[str, float | str]],
    output_path: Path,
    *,
    title: str,
    value_key: str = "mean_abs_shap",
    top_n: int = 12,
) -> None:
    ensure_parent_directory(output_path)

    trimmed = feature_importance_items[:top_n]
    features = [str(item["feature"]) for item in trimmed][::-1]
    scores = [float(item[value_key]) for item in trimmed][::-1]

    plt.figure(figsize=(8.5, 6))
    plt.barh(features, scores, color="#2E86C1")
    plt.xlabel("Importance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
