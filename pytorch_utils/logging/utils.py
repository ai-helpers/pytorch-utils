import os
from typing import Optional

import mlflow
import pandas as pd


def get_or_create_mlflow_experiment(exp_name):
    exp = mlflow.get_experiment_by_name(exp_name)
    return mlflow.get_experiment(mlflow.create_experiment(exp_name)) if exp is None else exp


def get_active_mlflow_run_id():
    active_run = mlflow.active_run()
    return active_run.info.run_id if active_run else None


def get_active_mlflow_experiment_id():
    active_run = mlflow.active_run()
    return active_run.info.experiment_id if active_run else None


def mlflow_log_pandas_artifact(
    df: pd.DataFrame, df_name: str, artifact_path: Optional[str] = None
) -> None:
    tmp_path_csv = f"/tmp/{df_name}.csv"
    df.reset_index().to_csv(tmp_path_csv, index=False)
    mlflow.log_artifact(tmp_path_csv, artifact_path=artifact_path)
    os.remove(tmp_path_csv)
