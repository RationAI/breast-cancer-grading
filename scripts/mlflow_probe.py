"""One-off: find the experiment that holds the graded MIL runs (test/kappa)."""

import mlflow

mlflow.set_tracking_uri("http://mlflow-s3.rationai-mlflow")
for name in ["BRC Grading", "Breast Cancer Grading"]:
    exp = mlflow.get_experiment_by_name(name)
    print(f"\n{name!r} -> id", None if exp is None else exp.experiment_id, flush=True)
    if exp is None:
        continue
    runs = mlflow.search_runs([exp.experiment_id], output_format="list", max_results=200)
    graded = [r for r in runs if "test/kappa" in r.data.metrics]
    print("   runs:", len(runs), "| with test/kappa:", len(graded), flush=True)
    if graded:
        g = max(graded, key=lambda r: r.info.start_time)
        print("   newest graded:", g.info.run_id, g.data.tags.get("mlflow.runName"), flush=True)
        print("   metrics:", {k: round(v, 4) for k, v in sorted(g.data.metrics.items()) if k.startswith("test")}, flush=True)
        print("   variant param:", g.data.params.get("variant"), flush=True)
