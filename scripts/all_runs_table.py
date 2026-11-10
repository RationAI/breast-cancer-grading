"""One-line-per-finished-run table, straight from MLflow (no pod, no PVC).

``report_variants.py`` answers questions about named variants; this answers "what do we
have, ranked" — every finished run in the grading experiment that actually reached test,
newest checkpoint deduplicated. The phantom side runs the provenance autologger leaves
behind carry ``params.variant`` too, so the filter is on a test metric, not on the param.

    MLFLOW_URL=https://mlflow-s3.rationai.cloud.trusted.e-infra.cz \
        python scripts/all_runs_table.py [experiment_id]
"""

from __future__ import annotations

from typing import Any

from mlflow.tracking import MlflowClient
from report_variants import use_cluster_mlflow


def main(experiment_id: str = "82") -> None:
    client = MlflowClient(tracking_uri=use_cluster_mlflow())
    runs = client.search_runs(
        [experiment_id],
        "params.variant IS NOT NULL and status='FINISHED'",
        order_by=["attributes.start_time"],
        max_results=100,
    )
    seen: set[str] = set()
    rows: list[tuple[Any, ...]] = []
    for run in runs:
        variant = run.data.params.get("variant")
        metrics, params = run.data.metrics, run.data.params
        if not variant or variant in seen or "test/accuracy" not in metrics:
            continue
        seen.add(variant)  # newest run wins: the list is ordered by start_time
        recalls = "/".join(
            f"{metrics.get(f'test/recall_class_{c}', float('nan')):.2f}"
            for c in range(4)
        )
        loss = "ordinal" if "Ordinal" in str(params.get("loss", "")) else "ce"
        rows.append(
            (
                variant,
                params.get("seed", "?"),
                str(params.get("max_tiles", "?"))[:5],
                str(
                    params.get("ml/aggregator", params.get("aggregator", "?"))
                ).rpartition(".")[2][:6],
                loss,
                int(metrics.get("epoch", 0)),
                (run.info.end_time - run.info.start_time) / 60000,
                metrics["test/accuracy"],
                metrics.get("test/auroc", float("nan")),
                metrics.get("test/f1", float("nan")),
                metrics.get("test/kappa", float("nan")),
                metrics.get("valid/loss", float("nan")),
                recalls,
                run.info.run_id[:8],
            )
        )

    header = (
        f"{'variant':18s} {'seed':4s} {'tiles':5s} {'pool':6s} {'loss':8s} {'ep':>3s}"
        f" {'min':>4s} {'bal':>6s} {'auroc':>6s} {'f1':>6s} {'kappa':>6s} {'vloss':>6s}"
        f"  recalls  run"
    )
    print(header)
    for row in sorted(rows, key=lambda r: -r[7]):
        print(
            f"{row[0]:18s} {row[1]!s:4s} {row[2]:5s} {row[3]:6s} {row[4]:8s} "
            f"{row[5]:3d} {row[6]:4.0f} {row[7]:.4f} {row[8]:.4f} {row[9]:.4f} "
            f"{row[10]:.4f} {row[11]:.4f}  {row[12]}  {row[13]}"
        )
    print(f"{len(rows)} finished runs with test metrics in experiment {experiment_id}")


if __name__ == "__main__":
    import sys

    main(*sys.argv[1:])
