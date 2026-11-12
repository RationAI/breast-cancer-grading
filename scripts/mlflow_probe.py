"""One-off: verify the overlay report run in experiment 82, then this file goes away."""

import os

import mlflow
from mlflow.tracking import MlflowClient

RUN = "4cf76dc4079a4849a34eef2923815032"
mlflow.set_tracking_uri("http://mlflow-s3.rationai-mlflow")
exp = mlflow.get_experiment_by_name("BRC Grading")
graded = [r for r in mlflow.search_runs([exp.experiment_id], output_format="list", max_results=200)]
mine = [r for r in graded if r.info.run_id == RUN]
print("experiment", exp.experiment_id, exp.name, "| runs:", len(graded), "| report run present:", bool(mine), flush=True)
r = mlflow.get_run(RUN)
print("  name:", r.data.tags.get("mlflow.runName"), "| metrics:", dict(r.data.metrics), flush=True)
print("  graded siblings (test/kappa):", sum(1 for x in graded if "test/kappa" in x.data.metrics), flush=True)
cl = MlflowClient()
print("  artifacts:", [f.path for f in cl.list_artifacts(RUN)], flush=True)
p = cl.download_artifacts(RUN, "test_heatmap.html", "/tmp/chk")
print("  downloaded", os.path.getsize(p), "bytes | on PVC:", os.path.getsize("/mnt/projects/breast_cancer/brc_grading/reports/test_heatmap.html"), flush=True)
