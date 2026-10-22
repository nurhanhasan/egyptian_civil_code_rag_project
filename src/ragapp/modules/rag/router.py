from fastapi import APIRouter, Depends, Response, status
from fastapi.responses import JSONResponse

from ragapp.core.config import Settings, get_settings
from ragapp.modules.rag.dependency.rag import get_rag_pipeline_service
from ragapp.modules.rag.schema import AbstentionResponse, AskRequest, AskResponse
from ragapp.modules.rag.service import RAGPipelineService

router = APIRouter(tags=["RAG"])


@router.post(
    "/ask",
    response_model=AskResponse | AbstentionResponse,
    responses={
        422: {"description": "Validation error (empty question)"},
        502: {"description": "Generator upstream error"},
    },
)
async def ask(
    request: AskRequest,
    response: Response,
    service: RAGPipelineService = Depends(get_rag_pipeline_service),
    settings: Settings = Depends(get_settings),
) -> AskResponse | AbstentionResponse:
    """Answer a question about the Egyptian Civil Code.

    Returns structured response with answer and citations,
    or abstention with reason and no answer text.

    """

    backend = (settings.RETRIEVER_BACKEND or "").strip().lower()

    result, trace_id = await service.run_pipeline(request.question)

    headers = {"X-Trace-ID": trace_id, "X-Retriever-Backend": backend}
    response.headers["X-Trace-ID"] = trace_id
    # TODO: nginx adds `X-Upstream-Variant`Maybe remove this later,
    # leave it now to inspect canary pattern.
    response.headers["X-Retriever-Backend"] = backend

    # Abstention responses should be 200 but with abstained=True
    if isinstance(result, AbstentionResponse):
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=result.model_dump(mode="json"),
            headers=headers,
        )

    return result
