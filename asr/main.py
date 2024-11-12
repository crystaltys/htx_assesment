from fastapi import FastAPI
from src.api.predict import router
from src.core.config import (APP_NAME, APP_VERSION, API_PORT, IS_DEBUG)
from src.core.event_handlers import (start_app_handler, stop_app_handler)
import uvicorn


def get_app() -> FastAPI:
    fast_app = FastAPI(title=APP_NAME, version=APP_VERSION, debug=IS_DEBUG)
    fast_app.include_router(router)
    fast_app.add_event_handler("startup", start_app_handler(fast_app))
    fast_app.add_event_handler("shutdown", stop_app_handler(fast_app))

    return fast_app

app = get_app()

if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=API_PORT, reload=True)