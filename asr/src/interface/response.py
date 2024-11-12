from pydantic import BaseModel

class AsrResponse(BaseModel):
    transcription : str
    duration      : str