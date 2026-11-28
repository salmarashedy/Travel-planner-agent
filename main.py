from fastapi import FastAPI

from api.router import api_router
from core.config import get_settings
from core.exceptions import register_exception_handlers
from core.logging import configure_logging
from services.chat_service import ChatService, build_chat_service


def create_app(chat_service: ChatService | None = None) -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(title=settings.app_name, version=settings.app_version)
    app.state.chat_service = chat_service or build_chat_service(settings)
    app.include_router(api_router)
    register_exception_handlers(app)
    return app


app = create_app()
