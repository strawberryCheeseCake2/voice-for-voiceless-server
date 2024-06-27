# pydantic models
from pydantic import BaseModel

class UserBase(BaseModel):
    name: str


class UserCreate(UserBase):
    name: str


class User(UserBase):
    id: int

    class Config:
        from_attributes = True

class MessageBase(BaseModel):
    content: str
    sender: str
    sentTime: str

class MessageCreate(BaseModel):
    content: str
    sender: str
    sentTime: str

class Message(MessageBase):
    id: int
    
class WSMessageCreate(MessageCreate):
    isStream: bool = False
    isFirstToken: bool = False