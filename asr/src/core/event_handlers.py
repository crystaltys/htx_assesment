from typing import Callable
from fastapi import FastAPI
from src.core.config import DEFAULT_CONFIG_PATH
from src.services.cv_decode import DataPipeline

def _startup_engine(app: FastAPI) -> None:
    cfg_path = DEFAULT_CONFIG_PATH
    pipeline = DataPipeline(cfg_path)
    app.state.engine = pipeline

def _shutdown_engine(app: FastAPI) -> None:
    app.state.engine = None

def start_app_handler(app: FastAPI) -> Callable:
    def startup() -> None:
        _startup_engine(app)
    return startup

def stop_app_handler(app: FastAPI) -> Callable:
    def shutdown() -> None:
        _shutdown_engine(app)
    return shutdown