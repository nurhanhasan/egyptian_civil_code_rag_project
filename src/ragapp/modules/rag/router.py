from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from ragapp.dependencies.rag import get_rag_pipeline_service
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
    service: RAGPipelineService = Depends(get_rag_pipeline_service),
) -> AskResponse | AbstentionResponse:
    """Answer a question about the Egyptian Civil Code.

    Returns structured response with answer and citations,
    or abstention with reason and no answer text.
    """
    result = await service.run_pipeline(request.question)

    # Abstention responses should be 200 but with abstained=True
    if isinstance(result, AbstentionResponse):
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=result.model_dump(mode="json"),
        )

    return result
