from fastapi import APIRouter, Depends, Request

from schemas.chat import ChatRequest, ChatResponse
from services.chat_service import ChatService

router = APIRouter(tags=["chat"])


def get_chat_service(request: Request) -> ChatService:
    return request.app.state.chat_service


@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, service: ChatService = Depends(get_chat_service)) -> ChatResponse:
    return await service.handle(body)
