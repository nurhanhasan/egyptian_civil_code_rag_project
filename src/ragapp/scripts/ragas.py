"""RAGAS evaluation for the Egyptian Civil Code RAG pipeline.

Runs every gold question through the RAG pipeline, captures the generated
answer plus the retrieved contexts, and scores the run with RAGAS.

Usage:
    uv run python -m ragapp.scripts.ragas --gold data/gold/gold_set.json \
        --judge-model qwen3.8-27b-8bit --with-embeddings \
            --judge-base-url http://localhost:8000/v1  --strict
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import click
from langchain_openai import ChatOpenAI
from langfuse import get_client
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig

from ragapp.core.config import Settings, get_settings
from ragapp.modules.rag.domain.entities import EmbeddingResult
from ragapp.modules.rag.domain.ports import Embedder
from ragapp.modules.rag.experimental_tracking.mlflow_evaluator import Evaluator
from ragapp.modules.rag.factories import (
    get_embedder,
    get_generator,
    get_reranker,
    get_retriever,
    reset_embedder_cache,
)
from ragapp.modules.rag.service import RAGPipelineService
from ragapp.scripts.eval_rows import (
    attach_scores,
    error_row,
    extract_eval_row,
    is_scorable,
)

# `evaluate()` only accepts legacy `Metric` objects (imported lazily below);
# the relocated `ragas.metrics.collections` classes are incompatible with it.
warnings.filterwarnings(
    "ignore", message="Importing .* from 'ragas.metrics' is deprecated"
)

DEFAULT_GOLD_PATH = Path("data/gold/gold_set.json")
DEFAULT_OUT_PATH = Path("reports/ragas_results.json")
DEFAULT_CONCURRENCY = 4
# Per-metric wall-clock budget. RAGAS's default is 180s, which slow multi-call
# metrics (faithfulness, answer_correctness, answer_relevancy) exceed when the
# judge runs several metrics concurrently -> TimeoutError -> NaN.
DEFAULT_METRIC_TIMEOUT = 600
# Max concurrent judge requests RAGAS issues. The default (16) overwhelms a
# single slow judge and pushes per-metric latency past the timeout.
DEFAULT_JUDGE_CONCURRENCY = 4


def _run_sync(coro: Any) -> Any:
    """Run a coroutine from sync code, even inside a running event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


class _AppEmbedderRagas(BaseRagasEmbeddings):
    """Adapt the app's `Embedder` port (dense vectors) to RAGAS embeddings.

    Only the dense half of `EmbeddingResult` is used; sparse vectors are a
    retrieval concern and RAGAS has no notion of them.
    """

    def __init__(self, embedder: Embedder, run_config: RunConfig | None = None) -> None:
        super().__init__()
        self._embedder = embedder
        # BaseRagasEmbeddings declares `run_config` but does not initialize it;
        # RAGAS never calls set_run_config on custom embeddings, so embedding
        # metrics raise AttributeError unless we set it ourselves.
        self.set_run_config(run_config or RunConfig())

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        result: EmbeddingResult = await self._embedder.embed(list(texts))
        return result.dense

    async def aembed_query(self, text: str) -> list[float]:
        dense = await self.aembed_documents([text])
        return dense[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return _run_sync(self.aembed_documents(list(texts)))

    def embed_query(self, text: str) -> list[float]:
        return _run_sync(self.aembed_query(text))


def _file_sha256(path: Path) -> str:
    """Short content hash identifying a gold-set revision."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _load_gold_items(gold_path: Path) -> list[dict[str, Any]]:
    """Load answerable gold items with a question and a reference answer."""
    payload = json.loads(gold_path.read_text(encoding="utf-8"))
    items = payload.get("items", payload) if isinstance(payload, dict) else payload

    gold: list[dict[str, Any]] = []
    for item in items:
        if item.get("answerable") is False:
            continue
        reference = item.get("reference_answer") or item.get("answer")
        question = item.get("question")
        if not question or not reference:
            continue
        gold.append(
            {
                "id": item.get("id", question),
                "question": question,
                "reference": reference,
            }
        )
    return gold


def _build_rag_service(settings: Settings) -> RAGPipelineService:
    """Build RAGPipelineService from settings using factories."""
    return RAGPipelineService(
        generator=get_generator(settings),
        embedder=get_embedder(settings),
        retriever=get_retriever(settings),
        reranker=get_reranker(settings),
        system_prompt=settings.GENERATOR_SYSTEM_PROMPT,
    )


def _build_judge(
    settings: Settings,
    no_think: bool,
) -> Any:
    """Legacy RAGAS judge bound to the OpenAI-compatible vLLM endpoint.

    `no_think` sends `chat_template_kwargs.enable_thinking=false`, which Qwen3
    chat templates honor. Without it, reasoning traces break RAGAS's JSON
    parsing and metrics collapse to NaN.
    """
    extra_body: dict[str, Any] | None = None
    if no_think:
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

    # TODO: Replace
    chat = ChatOpenAI(
        model=settings.JUDGE_MODEL or "",
        base_url=settings.JUDGE_BASE_LLM or "",
        api_key=settings.OPENAI_API_KEY or "EMPTY",
        temperature=0.0,
        extra_body=extra_body,
    )
    return LangchainLLMWrapper(chat)


def _build_metrics(
    judge: Any, embedder: Embedder | None, run_config: RunConfig | None = None
) -> list[Any]:
    """Legacy RAGAS `Metric` instances, all bound to `judge`.

    LLM-only by default. Embedding-based metrics are added only when an
    embedder is supplied, since they are the ones RAGAS resolves through
    embeddings.
    """
    from ragas.metrics import (
        AnswerCorrectness,
        AnswerRelevancy,
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
    )

    metrics: list[Any] = [
        Faithfulness(llm=judge),
        LLMContextPrecisionWithReference(llm=judge),
        LLMContextRecall(llm=judge),
    ]
    if embedder is not None:
        # TODO: CHeck with get_embedder
        embeddings = _AppEmbedderRagas(embedder, run_config)
        metrics.append(AnswerCorrectness(llm=judge, embeddings=embeddings))
        metrics.append(AnswerRelevancy(llm=judge, embeddings=embeddings))
    return metrics


async def _collect_responses(
    service: RAGPipelineService,
    gold: list[dict[str, Any]],
    concurrency: int,
) -> list[dict[str, Any]]:
    """Answer every gold question concurrently; preserve gold order."""
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def _answer(item: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            start = time.perf_counter()
            # `sources` is `list[str]` by contract, so contexts and rerank
            # scores come from the chunks the pipeline actually quoted.
            response, trace_id, reranked = await service.run_pipeline_with_context(
                item["question"]
            )
            latency_ms = (time.perf_counter() - start) * 1000

        return extract_eval_row(
            item, response, reranked, latency_ms=latency_ms, trace_id=trace_id
        )

    results = await asyncio.gather(
        *(_answer(item) for item in gold), return_exceptions=True
    )

    rows: list[dict[str, Any]] = [
        error_row(item, result) if isinstance(result, BaseException) else result
        for item, result in zip(gold, results, strict=True)
    ]

    await service.close()
    return rows


def _json_safe(value: Any) -> Any:
    """Replace non-finite floats (NaN/Inf) with null so the JSON is valid."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _mean(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None and math.isfinite(value)]
    return sum(present) / len(present) if present else None


def _build_run_params(
    settings: Settings,
    *,
    gold_path: Path,
    judge_name: str,
    limit: int | None,
    concurrency: int,
    metric_timeout: int,
    judge_concurrency: int,
    strict: bool,
    with_embeddings: bool,
    no_think: bool,
    tracking_uri: str | None,
    experiment_name: str | None,
) -> dict[str, Any]:
    """Assemble the scalar param set logged for a RAGAS run."""
    params = settings.load_params()
    if tracking_uri:
        params["mlflow_tracking_uri"] = tracking_uri
    if experiment_name:
        params["mlflow_experiment_name"] = experiment_name
    params.update(
        {
            "eval_tier": 2,
            "judge_model": judge_name,
            "judge_version": "",
            "gold_set_version": _file_sha256(gold_path),
            "gold_path": str(gold_path),
            "ragas_limit": limit if limit is not None else "all",
            "ragas_concurrency": concurrency,
            "ragas_metric_timeout": metric_timeout,
            "ragas_judge_concurrency": judge_concurrency,
            "ragas_strict": strict,
            "ragas_with_embeddings": with_embeddings,
            "ragas_judge_no_think": no_think,
        }
    )
    return params


def _build_mlflow_metrics(
    rows: list[dict[str, Any]],
    metric_names: list[str],
    summary: dict[str, float | None],
    *,
    abstentions: int,
    errors: int,
    scored_total: int,
) -> dict[str, float]:
    """Aggregate metric means plus per-metric parse-failure rates.

    Requires the parse-failure rate to be reported alongside every score, since
    RAGAS silently drops unparseable items. The rate is computed over the rows
    submitted for scoring, not over every gold item: abstentions and crashes
    were never going to produce a score and must not look like judge failures.
    """
    metrics: dict[str, float] = {}
    for name in metric_names:
        value = summary.get(name)
        if value is not None and math.isfinite(value):
            metrics[name] = value
        valid = sum(
            1
            for row in rows
            if (score := row["scores"].get(name)) is not None and math.isfinite(score)
        )
        metrics[f"{name}_n_valid"] = float(valid)
        metrics[f"{name}_parse_failure_rate"] = (
            (scored_total - valid) / scored_total if scored_total else 0.0
        )
    metrics.update(
        {
            "items": float(len(rows)),
            "scored_items": float(scored_total),
            "answered": float(len(rows) - abstentions - errors),
            "abstentions": float(abstentions),
            "errors": float(errors),
        }
    )
    return metrics


@click.command()
@click.option(
    "--gold",
    "gold_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_GOLD_PATH,
    show_default=True,
    help="Gold JSON with questions and reference answers.",
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=DEFAULT_OUT_PATH,
    show_default=True,
    help="Path to write the per-item RAGAS results JSON.",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Evaluate only the first N gold items.",
)
@click.option(
    "--concurrency",
    type=int,
    default=DEFAULT_CONCURRENCY,
    show_default=True,
    help="Max in-flight pipeline runs (drives vLLM/batching concurrency).",
)
@click.option(
    "--metric-timeout",
    type=int,
    default=DEFAULT_METRIC_TIMEOUT,
    show_default=True,
    help="Per-metric wall-clock budget (seconds) passed to RAGAS RunConfig. "
    "Too low turns multi-call metrics into silent NaNs.",
)
@click.option(
    "--judge-concurrency",
    type=int,
    default=DEFAULT_JUDGE_CONCURRENCY,
    show_default=True,
    help="Max concurrent judge requests RAGAS issues (RunConfig.max_workers).",
)
@click.option(
    "--judge-model",
    type=str,
    default=None,
    help="Judge LLM model (defaults to JUDGE_MODEL).",
)
@click.option(
    "--judge-base-url",
    type=str,
    default=None,
    help="Override JUDGE_BASE_URL for the judge LLM.",
)
@click.option(
    "--no-think/--think",
    "no_think",
    default=True,
    show_default=True,
    help="Disable Qwen3 thinking via vLLM chat_template_kwargs "
    "(enable_thinking=false). Reasoning traces break RAGAS JSON parsing and "
    "collapse metrics to NaN, so this is on by default; pass --think to "
    "re-enable.",
)
@click.option(
    "--strict",
    is_flag=True,
    default=False,
    help="Raise on metric errors instead of recording NaN.",
)
@click.option(
    "--with-embeddings",
    is_flag=True,
    default=False,
    help="Add embedding metrics (answer_correctness, answer_relevancy) using "
    "the app embedder.",
)
@click.option(
    "--tracking-uri",
    type=str,
    default=None,
    help="MLflow tracking URI (defaults to MLFLOW_TRACKING_URI or sqlite:///mlflow.db).",
)
@click.option(
    "--experiment-name",
    type=str,
    default=None,
    help="MLflow experiment name (defaults to MLFLOW_EXPERIMENT_NAME).",
)
@click.option(
    "--change",
    type=str,
    default=None,
    help="Track changes incorporated into this run. (defaults to None)",
)
def main(
    gold_path: Path,
    out_path: Path,
    limit: int | None,
    concurrency: int,
    metric_timeout: int,
    judge_concurrency: int,
    judge_model: str | None,
    judge_base_url: str | None,
    no_think: bool,
    strict: bool,
    with_embeddings: bool,
    tracking_uri: str | None,
    experiment_name: str | None,
    change: str | None,
) -> None:
    """Evaluate the RAG pipeline against a gold set with RAGAS.

    One MLflow run per evaluation: config params, aggregate metric means and
    parse-failure rates, and the per-item JSON as an artifact.
    """
    settings = get_settings()
    judge_name = judge_model or settings.JUDGE_MODEL
    judge_base_url = judge_base_url or settings.JUDGE_BASE_LLM

    params = _build_run_params(
        settings,
        gold_path=gold_path,
        judge_name=judge_name,
        limit=limit,
        concurrency=concurrency,
        metric_timeout=metric_timeout,
        judge_concurrency=judge_concurrency,
        strict=strict,
        with_embeddings=with_embeddings,
        no_think=no_think,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
    )

    evaluator = Evaluator(
        params,
        tracking_uri=params["mlflow_tracking_uri"],
        experiment_name=params["mlflow_experiment_name"],
    )

    gold = _load_gold_items(gold_path)
    if limit is not None:
        gold = gold[:limit]
    if not gold:
        raise click.ClickException(f"No answerable items found in {gold_path}")

    click.echo(f"Loaded {len(gold)} gold items from {gold_path}")

    with evaluator:
        service = _build_rag_service(settings)
        rows = asyncio.run(_collect_responses(service, gold, concurrency))

        abstentions = sum(1 for row in rows if row["abstained"])
        errors = sum(1 for row in rows if row["error"])
        answered = len(rows) - abstentions - errors
        click.echo(
            f"Pipeline answered {answered}/{len(rows)} "
            f"(abstained {abstentions}, errored {errors})"
        )
        for row in rows:
            if row["error"]:
                click.echo(f"  error {row['id']}: {row['error']}")

        # Score answers only: a refusal has no answer to be faithful to, and
        # averaging one in as 0.0 reports a policy as a grounding defect.
        scorable = [row for row in rows if is_scorable(row)]
        if not scorable:
            raise click.ClickException(
                f"Nothing to score: {abstentions} abstention(s), {errors} error(s). "
                "Fix the pipeline before trusting any metric."
            )
        click.echo(
            f"Scoring {len(scorable)}/{len(rows)} rows; "
            f"{sum(len(row['contexts']) for row in scorable)} contexts total"
        )

        dataset = EvaluationDataset(
            samples=[
                SingleTurnSample(
                    user_input=row["question"],
                    retrieved_contexts=row["contexts"],
                    response=row["answer"],
                    reference=row["reference"],
                )
                for row in scorable
            ]
        )

        run_config = RunConfig(
            timeout=metric_timeout,
            max_workers=judge_concurrency,
        )
        judge = _build_judge(settings, no_think)

        if with_embeddings:
            # _collect_responses() closed the cached singleton embedder; evict it
            # so the RAGAS embedding metrics get a live model, not model=None.
            reset_embedder_cache()
            embedder = get_embedder(settings)
        else:
            embedder = None
        metrics = _build_metrics(judge, embedder, run_config)

        result = evaluate(
            dataset,
            metrics=metrics,
            llm=judge,
            show_progress=True,
            raise_exceptions=strict,
            run_config=run_config,
        )

        langfuse = get_client()

        metric_names = [metric.name for metric in metrics]
        attach_scores(rows, scorable, list(result.scores), metric_names)

        for row in rows:
            trace_id = row.get("trace_id")
            if not trace_id:
                continue

            for score_name, score_value in row["scores"].items():
                if score_value is None:
                    continue
                langfuse.create_score(
                    name=score_name,  # "faithfulness", "context_recall", ...
                    value=score_value,
                    trace_id=trace_id,
                    comment=row.get("question"),
                )

        summary = {
            name: _mean([row["scores"].get(name) for row in rows])
            for name in metric_names
        }

        mlflow_metrics = _build_mlflow_metrics(
            rows,
            metric_names,
            summary,
            abstentions=abstentions,
            errors=errors,
            scored_total=len(scorable),
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                _json_safe(
                    {
                        "meta": {
                            "gold_path": str(gold_path),
                            "gold_set_version": params["gold_set_version"],
                            "judge_model": judge_name,
                            "judge_no_think": no_think,
                            "metric_timeout": metric_timeout,
                            "judge_concurrency": judge_concurrency,
                            "strict": strict,
                            "with_embeddings": with_embeddings,
                            "items": len(rows),
                            "scored_items": len(scorable),
                            "abstentions": abstentions,
                            "errors": errors,
                        },
                        "summary": summary,
                        "items": rows,
                    }
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        run_notes = []
        if change:
            run_notes.append(f"Change: {change}")
        run_notes.append(
            f"RAGAS Tier-2 eval: {len(rows)} items, judge={judge_name} "
            f"(no_think={no_think})"
        )
        run_notes.append(f"Gold set: {gold_path} ({params['gold_set_version']})")

        evaluator.log_metrics(mlflow_metrics)
        evaluator.log_artifact(out_path)
        evaluator.log_text("\n".join(run_notes))
        evaluator.set_tag(key="RAGAS Evaluation Path", value=True)
        # evaluator.set_tag("eval_tier", 2)
        evaluator.set_tag("RAGAS Judge Model", judge_name)
        evaluator.log_description(
            f"RAGAS Tier-2 evaluation over {len(rows)} items; judge={judge_name}."
        )

        click.echo("\nRAGAS summary:")
        for name, value in summary.items():
            click.echo(
                f"  {name}: {value:.4f}" if value is not None else f"  {name}: n/a"
            )
        click.echo(f"\nWrote per-item results to {out_path}")
        click.echo(f"Logged to MLflow run: {evaluator.run_id}")


if __name__ == "__main__":
    main()
