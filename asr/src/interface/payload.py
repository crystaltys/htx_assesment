from pydantic import BaseModel

class AsrPayload(BaseModel):
    file : str