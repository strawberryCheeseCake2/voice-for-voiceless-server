from typing import List, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager

import json

from .database import SessionLocal, engine

from sqlalchemy.orm import Session
from . import crud, schemas
from .devil import DevilManager


class ConnectionManager:
    def __init__(self, devil: Optional[DevilManager] = None):
        # init counter
        # init client, assistant, thread
        self.active_connections: List[WebSocket] = []
        self.devil = devil
        self.counter = 0

    def reset_counter(self):
        self.counter = 0

    def increase_counter(self):
        self.counter += 1

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


devil = DevilManager()
connection_manager = ConnectionManager(devil)


router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# https://github.com/tiangolo/fastapi/discussions/9664
@router.on_event("startup")
async def startup_event():
    await devil.setup_devil()


@router.websocket("/ws/{username}")
async def websocket_endpoint(websocket: WebSocket, username: str, db: Session = Depends(get_db)):
    await connection_manager.connect(websocket)

    try:
        while True:
            data = await websocket.receive_text()

            now = datetime.now()
            current_time = now.strftime("%Y-%m-%d %H:%M:%S")
            message = {"sentTime": current_time,
                       "username": username, "message": data}

            crud.log_message(db=db, message=schemas.MessageCreate(
                content=message["message"],
                sender=message["username"],
                sentTime=message["sentTime"]
            ))

            connection_manager.increase_counter()

            print(json.dumps(message))
            await connection_manager.broadcast(json.dumps(message))

            
            await connection_manager.devil.add_user_message(message["message"])
            print(connection_manager.counter)
            print(len(connection_manager.active_connections))
            if connection_manager.counter >= len(connection_manager.active_connections):
                if connection_manager.devil:
                    print("hit!")
                    devil_message_content = await connection_manager.devil.run_devill()
                    devil_message = {
                        "sentTime": current_time, "username": "Annoymous Devil", "message": devil_message_content}
                    print(devil_message)
                    await connection_manager.broadcast(json.dumps(devil_message))
                connection_manager.reset_counter()


    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
