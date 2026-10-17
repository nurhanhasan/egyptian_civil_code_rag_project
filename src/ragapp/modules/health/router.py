from fastapi import APIRouter, Depends

from ragapp.modules.health.service import HealthCheck, get_health_check

router = APIRouter(tags=["Health"])


@router.get("/health", status_code=200, response_model=None)
async def health_check(service: HealthCheck = Depends(get_health_check)):
    return await service.health_check()
