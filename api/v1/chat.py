from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from schemas.chat import ChatRequest

router = APIRouter(tags=["chat"])


@router.post("/chat", status_code=status.HTTP_501_NOT_IMPLEMENTED)
async def chat(request: ChatRequest) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        content={
            "detail": "The travel-planning chat workflow has not been implemented yet.",
            "status": "error",
        },
    )
