import pandas as pd  # noqa
import os, mlflow
print("injected MLFLOW_TRACKING_URI =", repr(os.environ.get("MLFLOW_TRACKING_URI")), flush=True)
for uri in ["http://mlflow-s3.rationai-mlflow", "http://mlflow.rationai-mlflow:5000"]:
    mlflow.set_tracking_uri(uri)
    print("\n==", uri, flush=True)
    try:
        exp = mlflow.get_experiment_by_name("Breast Cancer Grading")
        print("   exp:", None if exp is None else exp.experiment_id, flush=True)
        if exp:
            runs = mlflow.search_runs([exp.experiment_id], max_results=300)
            print("   runs:", len(runs), flush=True)
            if len(runs):
                nm = runs.get("run_name")
                print("   names:", sorted(set((nm.astype(str) if nm is not None else runs.get("tags/mlflow.runName", pd.Series(['?'])).astype(str))))[:10], flush=True)
                print("   ids:", list(runs.run_id.astype(str))[:5], flush=True)
    except Exception as ex:
        print("   ERR", type(ex).__name__, str(ex)[:140], flush=True)
    for rid in ["b93a3936b0e14129952b56360436141f", "4cbe114de1364f299b863c7cc4cd52f4", "e0844acbbb894e709c3ece03fcbbf1b4"]:
        try:
            r = mlflow.get_run(rid)
            print("   found", rid[:8], "exp", r.info.experiment_id, r.info.status, flush=True)
        except Exception:
            print("   missing", rid[:8], flush=True)
