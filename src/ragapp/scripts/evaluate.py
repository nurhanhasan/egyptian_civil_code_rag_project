"""CLI commands for Egyptian Civil Code RAG with MLflow tracking."""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import click

from ragapp.core.config import get_settings
from ragapp.modules.rag.evaluation.mlflow_evaluator import Evaluator
from ragapp.modules.rag.factories import (
    get_embedder,
    get_generator,
    get_reranker,
    get_retriever,
)
from ragapp.modules.rag.offline_ingestion.ingestion import (
    IngestionResult,
    run_ingestion,
)
from ragapp.modules.rag.schema import AbstentionResponse
from ragapp.modules.rag.service import RAGPipelineService


@click.group()
def evaluate() -> None:
    """Egyptian Civil Code RAG evaluate."""
    pass


def _build_rag_service(settings) -> RAGPipelineService:
    """Build RAGPipelineService from settings using factories."""
    generator = get_generator(settings)
    embedder = get_embedder(settings)
    retriever = get_retriever(settings)
    reranker = get_reranker(settings)
    return RAGPipelineService(
        generator=generator,
        embedder=embedder,
        retriever=retriever,
        reranker=reranker,
        system_prompt=settings.GENERATOR_SYSTEM_PROMPT,
    )


@evaluate.command()
@click.argument("question", type=str)
@click.option(
    "--tracking-uri",
    type=str,
    default=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
    help="MLflow tracking URI (defaults to sqlite:///mlflow.db)",
)
@click.option(
    "--experiment-name",
    type=str,
    default=None,
    help="MLflow experiment name (defaults to egyptian-civil-code-rag)",
)
@click.option(
    "--change",
    type=str,
    default=None,
    help="Track changes incorporated into this run. (defaults to None)",
)
def ask(
    question: str,
    tracking_uri: str | None,
    experiment_name: str | None,
    change: str | None,
) -> None:
    """Ask a question and track metrics in MLflow."""
    settings = get_settings()
    params = settings.load_params()

    if tracking_uri:
        params["mlflow_tracking_uri"] = tracking_uri
    if experiment_name:
        params["mlflow_experiment_name"] = experiment_name

    evaluator = Evaluator(
        params,
        tracking_uri=params["mlflow_tracking_uri"],
        experiment_name=params["mlflow_experiment_name"],
    )

    rag_service = _build_rag_service(settings)

    with evaluator:
        RUN_NOTES = []
        RUN_NOTES.append(f"Change: {change}")
        # evaluator.log_note(f"Change: {change}")
        RUN_NOTES.append(f"Question: {question}")
        # evaluator.log_note(f"Question: {question}")

        start_total = time.perf_counter()

        # Run the pipeline
        response = asyncio.run(rag_service.run_pipeline(question))

        latency_ms_total = (time.perf_counter() - start_total) * 1000

        # Extract metrics from response
        if isinstance(response, AbstentionResponse):
            metrics = {
                "latency_ms_total": latency_ms_total,
                "retrieval_count": 0,
                "rerank_top_score": 0.0,
                "citation_count": 0,
                "abstained": 1,
                "tokens_prompt": 0,
                "tokens_completion": 0,
            }
            RUN_NOTES.append(f"Abstention Reason: {response.reason}")
            evaluator.log_description(f"Abstention Reason: {response.reason}")
            click.echo(f"Abstained: {response.reason}")
        else:
            metrics = {
                "latency_ms_total": latency_ms_total,
                "retrieval_count": len(response.sources),
                "rerank_top_score": response.sources[0].score
                if response.sources
                else 0.0,
                "citation_count": len(response.sources),
                "abstained": 0,
                "tokens_prompt": 0,
                "tokens_completion": 0,
            }
            RUN_NOTES.append(f"Response Answer: {response.answer}")
            RUN_NOTES.append(f"Response Sources: {response.sources}")

            click.echo(f"Answer: {response.answer}")
            click.echo(f"Citations: {len(response.sources)}")

        evaluator.log_metrics(metrics)
        evaluator.log_text("\n".join(RUN_NOTES))

        evaluator.set_tag(key="Query Path", value=True)

        click.echo(f"Logged to MLflow run: {evaluator.run_id}")


@evaluate.command()
@click.option(
    "--corpus-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default="data/corpus/processed",
    help="Path to processed corpus directory (default: data/corpus/processed)",
)
@click.option(
    "--qdrant-url",
    type=str,
    default=None,
    help="Qdrant URL (default: from settings)",
)
@click.option(
    "--alias",
    type=str,
    default="egyptian-civil-code",
    help="Stable alias name (default: egyptian-civil-code)",
)
@click.option(
    "--batch-size",
    type=int,
    default=32,
    help="Embedding batch size",
)
@click.option(
    "--upsert-batch",
    type=int,
    default=100,
    help="Upsert batch size",
)
@click.option(
    "--tracking-uri",
    type=str,
    default=None,
    help="MLflow tracking URI (defaults to sqlite:///mlflow.db)",
)
@click.option(
    "--experiment-name",
    type=str,
    default=None,
    help="MLflow experiment name (defaults to egyptian-civil-code-rag)",
)
def reindex(
    corpus_path: Path,
    qdrant_url: str | None,
    alias: str,
    batch_size: int,
    upsert_batch: int,
    tracking_uri: str | None,
    experiment_name: str | None,
) -> None:
    """Reindex corpus and track build metrics in MLflow."""
    settings = get_settings()
    params = settings.load_params()

    if qdrant_url is None:
        qdrant_url = settings.QDRANT_URL

    if tracking_uri:
        params["mlflow_tracking_uri"] = tracking_uri
    if experiment_name:
        params["mlflow_experiment_name"] = experiment_name

    evaluator = Evaluator(
        params,
        tracking_uri=params["mlflow_tracking_uri"],
        experiment_name=params["mlflow_experiment_name"],
    )

    with evaluator:
        result: IngestionResult = run_ingestion(
            settings=settings,
            qdrant_url=qdrant_url,
            alias=alias,
            chunks_path=corpus_path / "chunks.article_aware.jsonl",
            batch_size=batch_size,
            upsert_batch=upsert_batch,
        )

        # Update params with ingestion-specific values
        params["index_build_id"] = result.build_id
        params["corpus_revision"] = result.corpus_revision

        evaluator.log_metrics(
            {
                "chunks_indexed": result.chunks_indexed,
                "build_duration_ms": result.build_duration_ms,
            }
        )
        evaluator.log_artifact(result.manifest_path)
        evaluator.set_tag(key="Offline ingestion Path", value=True)

        click.echo(f"Ingestion complete: {result.chunks_indexed} chunks indexed")
        click.echo(f"Build duration: {result.build_duration_ms:.0f}ms")
        click.echo(f"Build ID: {result.build_id}")
        click.echo(f"Corpus revision: {result.corpus_revision}")
        click.echo(f"Manifest: {result.manifest_path}")
        click.echo(f"Logged to MLflow run: {evaluator.run_id}")


if __name__ == "__main__":
    evaluate()
