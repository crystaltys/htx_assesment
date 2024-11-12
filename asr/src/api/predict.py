from fastapi import APIRouter
from starlette.requests import Request

from src.interface.payload import AsrPayload
from src.interface.response import AsrResponse
from src.services.cv_decode import DataPipeline

router = APIRouter()

@router.post("/asr", response_model=AsrResponse, name="asr_inference")
def post_predict(request: Request, 
                 block_data: AsrPayload = None) -> AsrResponse:
    engine: DataPipeline = request.app.state.engine
    results : AsrResponse = engine.run(block_data)
    return results