"""One-off: prove the overlay report landed in the same MLflow experiment as the
MIL training runs, then this file goes away."""

import mlflow

mlflow.set_tracking_uri("http://mlflow-s3.rationai-mlflow")
exp = mlflow.get_experiment_by_name("Breast Cancer Grading")
runs = mlflow.search_runs([exp.experiment_id], output_format="list", max_results=60)
print("experiment", exp.experiment_id, "| runs:", len(runs), flush=True)
for r in sorted(runs, key=lambda r: r.info.start_time, reverse=True)[:8]:
    name = r.data.tags.get("mlflow.runName", "?")
    print(f"  {r.info.run_id[:12]}  {name[:44]:44s}  metrics={len(r.data.metrics)}", flush=True)
rep = mlflow.get_run("85bb637eed9f49fca4b98d15a0e9cd93")
print("\nreport run:", rep.info.run_id, "exp", rep.info.experiment_id, rep.info.status, flush=True)
print("  metrics:", dict(rep.data.metrics), flush=True)
from mlflow.tracking import MlflowClient

cl = MlflowClient()
print("  artifacts:", [f.path for f in cl.list_artifacts("85bb637eed9f49fca4b98d15a0e9cd93")], flush=True)
tmp = cl.download_artifacts("85bb637eed9f49fca4b98d15a0e9cd93", "test_heatmap.html", "/tmp")
import os
print("  downloaded", tmp, os.path.getsize(tmp), "bytes", flush=True)
