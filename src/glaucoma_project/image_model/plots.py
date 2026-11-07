"""Figures for the image-only model. Each function saves a PNG and closes the figure."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from glaucoma_project.data.classification import image_path
from glaucoma_project.evaluation.screening import reliability_table, validated_scores

RELIABILITY_BINS = 5


def _save(fig: plt.Figure, destination: Path) -> None:
    fig.tight_layout()
    fig.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_learning_curves(history: pd.DataFrame, destination: Path) -> None:
    fig, (loss_ax, rank_ax) = plt.subplots(1, 2, figsize=(11, 4))
    loss_ax.plot(history.epoch, history.train_bce, label="Train BCE")
    loss_ax.plot(history.epoch, history.validation_bce, label="Validation BCE")
    loss_ax.set(xlabel="Epoch", ylabel="Loss")
    rank_ax.plot(history.epoch, history.validation_auroc, label="Validation AUROC")
    rank_ax.plot(history.epoch, history.validation_average_precision, label="Validation AP")
    rank_ax.set(xlabel="Epoch", ylabel="Ranking metric")
    loss_ax.legend()
    rank_ax.legend()
    _save(fig, destination)


def plot_diagnostics(predictions: pd.DataFrame, threshold: float, title: str, destination: Path) -> None:
    """ROC curve, precision-recall curve and reliability (calibration) diagram side by side."""
    y, p = validated_scores(predictions.label, predictions.probability)
    fig, (roc_ax, pr_ax, rel_ax) = plt.subplots(1, 3, figsize=(14, 4))
    fpr, tpr, _ = roc_curve(y, p)
    roc_ax.plot(fpr, tpr, label=f"AUROC {roc_auc_score(y, p):.3f}")
    roc_ax.plot([0, 1], [0, 1], "--", color="gray")
    roc_ax.set(xlabel="False positive rate", ylabel="Sensitivity", title="ROC")
    precision, recall, _ = precision_recall_curve(y, p)
    pr_ax.plot(recall, precision, label=f"AP {average_precision_score(y, p):.3f}")
    pr_ax.axhline(y.mean(), linestyle="--", color="gray", label="Prevalence")
    pr_ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall")
    bins = reliability_table(y, p, RELIABILITY_BINS)
    rel_ax.plot([0, 1], [0, 1], "--", color="gray")
    rel_ax.scatter(bins.mean_score, bins.observed_rate, s=bins["count"] * 2 + 15)
    rel_ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted score",
               ylabel="Observed positive fraction", title="Reliability (marker size = bin count)")
    roc_ax.legend()
    pr_ax.legend()
    fig.suptitle(f"{title}; fixed threshold = {threshold:.4f}")
    _save(fig, destination)


def plot_errors(predictions: pd.DataFrame, root: Path, threshold: float, destination: Path,
                max_each: int = 3) -> None:
    """The most confident false negatives and false positives at ``threshold``, as full photographs.
    Writes nothing when there are no errors. Error-selected, so not a random sample."""
    missed = predictions[(predictions.label == 1) & (predictions.probability < threshold)]
    false_alarms = predictions[(predictions.label == 0) & (predictions.probability >= threshold)]
    examples = pd.concat([missed.nsmallest(max_each, "probability").assign(error="FN"),
                          false_alarms.nlargest(max_each, "probability").assign(error="FP")])
    if examples.empty:
        return
    fig, axes = plt.subplots(1, len(examples), figsize=(3.3 * len(examples), 3.5), squeeze=False)
    for ax, row in zip(axes[0], examples.itertuples()):
        with Image.open(image_path(root, row.classification_image)) as image:
            ax.imshow(np.asarray(image))
        ax.set_title(f"{row.error}: y={row.label}, p={row.probability:.3f}\n{row.camera}", fontsize=10)
        ax.set_xlabel(row.image_id, fontsize=6)
        ax.set_xticks([])
        ax.set_yticks([])
    _save(fig, destination)


def plot_training_examples(rows: pd.DataFrame, root: Path, destination: Path, per_class: int = 4,
                           seed: int = 0) -> None:
    """A random sample of training photographs from each class, to check the inputs by eye."""
    sample = rows.groupby("binary_label", group_keys=False).sample(n=per_class, random_state=seed)
    fig, axes = plt.subplots(2, per_class, figsize=(3 * per_class, 6))
    for ax, row in zip(axes.flat, sample.itertuples()):
        with Image.open(image_path(root, row.classification_image)) as image:
            ax.imshow(np.asarray(image))
        ax.set_title(f"label={row.binary_label} | {row.camera}")
        ax.axis("off")
    fig.suptitle("Training photographs; label 1 = CHAKSU glaucoma suspect")
    _save(fig, destination)
