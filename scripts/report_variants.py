"""Read grading training results back out of the cluster.

Run it as a CPU-only job (``gpus=0``) because MLflow and the shared PVC are only
reachable from inside the cluster. It reports two things per variant:

* the headline metrics of the finished MLflow run, and
* metrics recomputed from the per-slide prediction dumps
  (``predictions/test_epoch<N>.csv`` written by ``PredictionLogger``), for every
  epoch that was dumped, with the early-stopping best epoch marked.

The dumps are per-slide logits, so they support a paired comparison across
variants, which the MLflow scalars do not.

Usage::

    python scripts/report_variants.py --variants full_s456,full_ord_s456
    python scripts/report_variants.py --self-test
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
)


NUM_CLASSES = 4


def use_cluster_mlflow() -> str:
    """Point MLflow at the cluster server unless the environment already does.

    kube-jobs injects ``MLFLOW_TRACKING_URI`` into pods, which is why this only
    matters when running somewhere else; it is reported either way.
    """
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        uri = os.getenv("MLFLOW_URL", "http://mlflow:5000")
        os.environ["MLFLOW_TRACKING_URI"] = uri
    return os.environ["MLFLOW_TRACKING_URI"]


def metrics_from_logits(labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    """Slide-level metrics from raw logits (no AUROC, which needs probabilities)."""
    predictions = logits.argmax(axis=1)
    recalls = []
    for c in range(NUM_CLASSES):
        present = labels == c
        recalls.append(
            float((predictions[present] == c).mean()) if present.any() else float("nan")
        )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "quadratic_kappa": float(
            cohen_kappa_score(labels, predictions, weights="quadratic")
        ),
        "recall_c0": recalls[0],
        "recall_c1": recalls[1],
        "recall_c2": recalls[2],
        "recall_c3": recalls[3],
    }


def report_dumps(checkpoint_root: str, variant: str, best_epoch: int | None) -> None:
    """Print recomputed test metrics for every dumped epoch of one variant.

    ``variant`` is the MLflow param (underscores) while the checkpoint directory is
    named from the job variant (dashes), so the two spellings are not interchangeable.
    """
    pattern = os.path.join(
        checkpoint_root,
        f"ml_attention-{variant.replace('_', '-')}",
        "predictions",
        "test_epoch*.csv",
    )
    files = sorted(
        glob.glob(pattern), key=lambda p: int(p.rsplit("epoch", 1)[1].split(".")[0])
    )
    if not files:
        print(f"{variant:16s} no test prediction dumps")
        return
    for path in files:
        epoch = int(path.rsplit("epoch", 1)[1].split(".")[0])
        frame = pd.read_csv(path)
        labels = frame["label"].to_numpy()
        logits = frame[[f"logit_{c}" for c in range(NUM_CLASSES)]].to_numpy()
        m = metrics_from_logits(labels, logits)
        mark = (
            "  <- best"
            if best_epoch is not None and abs(epoch - best_epoch) <= 1
            else ""
        )
        print(
            f"{variant:16s} ep{epoch:<3d} n={len(frame):3d} "
            f"kappa={m['quadratic_kappa']:.4f} bal_acc={m['balanced_accuracy']:.4f} "
            f"f1={m['macro_f1']:.4f} acc={m['accuracy']:.4f} "
            f"recalls {m['recall_c0']:.2f}/{m['recall_c1']:.2f}/{m['recall_c2']:.2f}/{m['recall_c3']:.2f}{mark}"
        )


def report_mlflow(
    experiment_names: list[str], variants: list[str]
) -> dict[str, int | None]:
    """Print the finished MLflow run per variant; return its best epoch per variant."""
    try:
        from mlflow.tracking import MlflowClient
    except Exception as error:  # pragma: no cover - only outside the cluster
        print(f"MLflow unavailable: {error}")
        return dict.fromkeys(variants, None)

    print(f"tracking URI {use_cluster_mlflow()}")
    client = MlflowClient()
    experiment_ids = []
    for name in experiment_names:
        experiment = client.get_experiment_by_name(name)
        if experiment is not None:
            experiment_ids.append(experiment.experiment_id)
    if not experiment_ids:
        print(f"none of {experiment_names} exist")
        return dict.fromkeys(variants, None)

    best_epochs: dict[str, int | None] = {}
    for variant in variants:
        runs = client.search_runs(
            experiment_ids,
            f"params.variant = '{variant}' and status = 'FINISHED'",
            order_by=["attributes.end_time DESC"],
            max_results=1,
        )
        if not runs:
            print(f"{variant:16s} no finished MLflow run")
            best_epochs[variant] = None
            continue
        metrics = runs[0].data.metrics
        best_epochs[variant] = int(metrics["epoch"]) if "epoch" in metrics else None
        print(
            f"{variant:16s} mlflow ep={metrics.get('epoch', float('nan')):5.0f} "
            f"kappa={metrics.get('test/kappa', float('nan')):.4f} "
            f"bal_acc={metrics.get('test/balanced_accuracy', float('nan')):.4f} "
            f"f1={metrics.get('test/macro_f1', float('nan')):.4f} "
            f"auroc={metrics.get('test/auroc', float('nan')):.4f} "
            f"loss={metrics.get('test/loss', float('nan')):.4f}"
        )
    return best_epochs


def self_test() -> None:
    """The metric maths, checked without a cluster."""
    labels = np.array([0, 1, 2, 3])
    identity = np.eye(NUM_CLASSES)
    perfect = metrics_from_logits(labels, identity)
    assert perfect["balanced_accuracy"] == 1.0, perfect
    assert perfect["quadratic_kappa"] == 1.0, perfect
    # Predicting every slide one grade too high gets nothing right: recall is
    # zero for every class, so balanced accuracy is zero even though every error
    # is only one class apart.
    one_off = metrics_from_logits(labels, np.roll(identity, 1, axis=1))
    assert one_off["balanced_accuracy"] == 0.0, one_off
    # Single-class predictions must not crash macro F1 or the recalls.
    flat = metrics_from_logits(labels, np.tile(identity[0], (4, 1)))
    # Class 0 gets precision 1/4 and recall 1 (F1 0.4), the rest get 0, so the
    # macro average is 0.4 / 4.
    assert flat["macro_f1"] == 0.1, flat
    assert flat["balanced_accuracy"] == 0.25, flat
    assert flat["recall_c1"] == 0.0 and flat["recall_c0"] == 1.0, flat
    # Quadratic kappa must grade errors by distance: every slide one grade out
    # has to score far above mirroring the scale, where the end classes are three
    # grades out. (A symmetric shift is not enough to show this - with one slide
    # per class a +1 and a -1 slip are mirror images and score identically.)
    eight = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    near = metrics_from_logits(eight, np.eye(NUM_CLASSES)[np.roll(eight, 1)])
    mirrored = metrics_from_logits(eight, np.eye(NUM_CLASSES)[[3, 2, 1, 0, 3, 2, 1, 0]])
    assert near["quadratic_kappa"] > mirrored["quadratic_kappa"], (near, mirrored)
    print(
        "self-test ok: one-grade-out kappa",
        round(near["quadratic_kappa"], 3),
        "mirrored kappa",
        round(mirrored["quadratic_kappa"], 3),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variants",
        default="tiles2048_s456,tiles2048_s123,att2048_s789,mean_s456,mean_s789,"
        "meanpool_s123,full_s456,full_ord_s456,full_ord_s789,t2048_ord_s456",
    )
    parser.add_argument("--experiments", default="BRC Grading")
    parser.add_argument(
        "--checkpoint-root",
        default="/mnt/projects/breast_cancer/brc_grading/checkpoints",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    variants = [v for v in args.variants.split(",") if v]
    print("=== MLflow (finished runs)")
    best_epochs = report_mlflow(args.experiments.split(","), variants)
    print("=== recomputed from per-slide test predictions")
    for variant in variants:
        report_dumps(args.checkpoint_root, variant, best_epochs.get(variant))


if __name__ == "__main__":
    main()
