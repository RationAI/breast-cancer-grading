"""Throwaway: list the report runs in the experiment the jobs use."""
import mlflow

mlflow.set_tracking_uri(mlflow.get_tracking_uri())
exp = mlflow.set_experiment("BRC Grading")
runs = mlflow.search_runs([exp.experiment_id], order_by=["attributes.start_time DESC"], max_results=200)
NAME = "tags.mlflow.runName"
rep = runs[runs[NAME].str.contains("overlay report", na=False)]
client = mlflow.MlflowClient()
print(f"experiment {exp.experiment_id} {exp.name}: {len(runs)} runs, {len(rep)} report runs")
for _, r in rep.head(8).iterrows():
    art = client.list_artifacts(r["run_id"])
    print(f"  {r['run_id'][:8]}  {r[NAME]:44s} slides={r['metrics.slides']:.0f}  "
          f"scale={r.get('params.scale')} opacity={r.get('params.opacity')} "
          f"art={[a.path for a in art]}")
