"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from .pipelines.data_understanding.pipeline import create_pipeline as create_data_understanding_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines(raise_errors=True)
    pipelines["data_understanding"] = create_data_understanding_pipeline()
    pipelines["__default__"] = sum(pipelines.values())
    return pipelines
