from fastapi import APIRouter, Depends

from ragapp.dependencies.rag import get_rag_pipeline_service
from ragapp.modules.rag.service import RAGPipelineService

router = APIRouter(tags=["RAG"])


@router.post("/ask")
async def ask(
    question: str, service: RAGPipelineService = Depends(get_rag_pipeline_service)
) -> str:
    return service.run_pipeline(question)
