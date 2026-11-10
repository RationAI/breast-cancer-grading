"""Register the per-slide embedding parquets as a provenance dataset.

Run it (not the module it submits) once the embeddings job is Complete:

    uv run --extra job python scripts/register_embeddings.py

The job lists ``<project_path>/embeddings/bc20-24/*.parquet`` into a cohort CSV,
registers it with ``rationai.provenance.register_dataset`` (which writes
``manifest.csv`` with the per-file size+mtime fingerprint next to the CSV) and
prints the ``run_id`` — paste that into ``configs/ml.yaml`` (``dataset.run_id``).
Absolute paths, so verification works from any working directory.
"""

from kube_jobs import storage, submit_job


EMBEDDINGS_DIR = "/mnt/projects/breast_cancer/brc_grading/embeddings/bc20-24"
COHORT_CSV = "/mnt/projects/breast_cancer/brc_grading/embeddings/bc20-24-embeddings.csv"

# ponytail: inline cohort build + registration; move into a repo module if the
# cohort ever needs more than "every parquet in the directory".
BUILD_AND_REGISTER = (
    "import pathlib; "
    f"d = pathlib.Path('{EMBEDDINGS_DIR}'); "
    "files = sorted(str(p) for p in d.glob('*.parquet')); "
    f"assert files, f'no parquets in {EMBEDDINGS_DIR}'; "
    f"pathlib.Path('{COHORT_CSV}').write_text('path\\n' + '\\n'.join(files) + '\\n'); "
    "print(f'{len(files)} slides'); "
    "from rationai.provenance import register_dataset; "
    "print('run_id =', register_dataset("
    f"csv='{COHORT_CSV}', path_column='path', dataset_name='bc20-24-embeddings', "
    "experiment_name='BRC Grading'))"
)


submit_job(
    job_name="brc-grading-register-embeddings",
    image="cerit.io/rationai/base:2.0.6",
    cpu=1,
    memory="4Gi",
    public=False,
    script=[
        "git clone https://github.com/RationAI/breast-cancer-grading.git workdir",
        "cd workdir",
        "uv sync --frozen",
        f'uv run python -c "{BUILD_AND_REGISTER}"',
    ],
    storage=[storage.secure.PROJECTS],
)
