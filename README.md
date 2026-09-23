# ML Pipeline

This project trains and evaluates a Random Forest classifier on the Pima Indians Diabetes dataset. It uses DVC for reproducible data and model pipelines, and MLflow with DagsHub for experiment tracking.

## Pipeline

Run the complete pipeline with:

```bash
export MLFLOW_TRACKING_PASSWORD="your-dagshub-token"
dvc repro
```

The pipeline has three stages:

1. `preprocess` creates `data/processed/data.csv` from `data/raw/data.csv`.
2. `train` creates `models/model.pkl` and logs the run to MLflow.
3. `evaluate` writes the model accuracy to `metrics.json`.

Use `dvc metrics show` to view the evaluation metric. Use `dvc push` to upload DVC-tracked outputs to the configured remote.
