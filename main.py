from fastapi import FastAPI

from api.router import api_router
from core.config import get_settings
from core.exceptions import register_exception_handlers
from core.logging import configure_logging


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(title=settings.app_name, version=settings.app_version)
    app.include_router(api_router)
    register_exception_handlers(app)
    return app


app = create_app()
