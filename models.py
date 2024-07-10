# sqlalchemy models
from sqlalchemy import Boolean, Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from .database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, index=True)

class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True)
    content = Column(String)
    sender = Column(String)
    sentTime = Column(String)

class SecretDm(Base):
    __tablename__ = "secretDms"

    id = Column(Integer, primary_key=True)

    content = Column(String)
    sender = Column(String)
    sentTime = Column(String)
    isUsed = Column(Boolean)
    
