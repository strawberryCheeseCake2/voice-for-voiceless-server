from typing import List
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from datetime import datetime

import json

from .database import SessionLocal, engine

from sqlalchemy.orm import Session
from . import crud, schemas
      
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


manager = ConnectionManager()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

router = APIRouter()
@router.websocket("/ws/{username}")
# async def websocket_endpoint(websocket: WebSocket, username: str):
async def websocket_endpoint(websocket: WebSocket, username: str ,db: Session = Depends(get_db)):
    await manager.connect(websocket)
    now = datetime.now()
    current_time = now.strftime("%H:%M")
    try:
        while True:
            data = await websocket.receive_text()
            message = {"sentTime":current_time,"username":username,"message":data}

            crud.log_message(db=db, message=schemas.MessageCreate(
                content=message["message"],
                sender=message["username"],
                sentTime=message["sentTime"]
            ))
            print(json.dumps(message))
            await manager.broadcast(json.dumps(message))
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
