

from typing import List
from pydantic import BaseModel


class Chat(BaseModel):
    User: str
    AI: str
    class Config:
        orm_mode = True

class ChatHistory(BaseModel):
    History: List[Chat]
    UserMessage: str