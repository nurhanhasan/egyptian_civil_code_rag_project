# src/ragapp/evaluation/mlflow_run.py
"""The Evaluator seam — MLflow-backed."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow


class Evaluator:
    """
    MLflow instantiation and lifecycle for Evaluation.
    """

    def __init__(
        self,
        params: dict[str, Any],
        *,
        tracking_uri: str | None = None,
        experiment_name: str = "egyptian-civil-code-rag",
    ) -> None:
        self._params = params
        self._tracking_uri = tracking_uri
        self._experiment_name = experiment_name
        self._run: mlflow.ActiveRun | None = None

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    def __enter__(self) -> "Evaluator":
        """Opens the MLflow run and logs parameters."""
        self._run = mlflow.start_run()
        mlflow.log_params(self._params)
        return self

    def __exit__(self, *exc: Any) -> None:
        """Closes the MLflow run."""
        if self._run:
            mlflow.end_run()
            self._run = None

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics to the active run."""
        if not self._run:
            raise RuntimeError("Evaluator not in a run — use 'with evaluator:'")
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: Path) -> None:
        """Log an artifact to the active run."""
        if not self._run:
            raise RuntimeError("Evaluator not in a run — use 'with evaluator:'")
        mlflow.log_artifact(str(path))

    def log_text(self, note: str) -> None:
        """Log a note/error/change message to the active run."""
        if not self._run:
            raise RuntimeError("Evaluator not in a run — use 'with evaluator:'")
        mlflow.log_text(text=note, artifact_file="notes.txt")

    def log_description(self, description: str) -> None:
        """Log a note/error message to the active run."""
        if not self._run:
            raise RuntimeError("Evaluator not in a run — use 'with evaluator:'")
        mlflow.set_tag("mlflow.note.content", description)

    def set_tag(self, key: str, value: Any) -> None:
        """Set a tag to the active run."""
        if not self._run:
            raise RuntimeError("Evaluator not in a run — use 'with evaluator:'")
        mlflow.set_tag(key, value)

    @property
    def run_id(self) -> str | None:
        return self._run.info.run_id if self._run else None
