from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str
    session_id: str
    persona: str


class ChatResponse(BaseModel):
    response: str
    session_id: str
