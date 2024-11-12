import requests
import logging
from src.interface.response import AsrResponse
from src.core.config import HF_BEARER_TOKEN

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Inference:
    def __init__(self, model_name: str):
        self._model_name = model_name
            
    def query(self, data: bytes, duration: float):
        api_url = f"https://api-inference.huggingface.co/models/facebook/{self._model_name}"
        headers = {
            "Authorization": f"Bearer hf_{HF_BEARER_TOKEN}", 
            "Content-Type": "multipart/form-data",
            "x-wait-for-model": "true"
        }
        try:
            response = requests.post(api_url, headers=headers, data=data.getvalue())
            logging.info(f"{response.text}")
            if response.status_code != 200:
                raise ValueError
        except Exception as e:
            print(e)
            raise
        out = AsrResponse(transcription = str(response.json()['text']), 
                          duration = str(duration))
        
        return out