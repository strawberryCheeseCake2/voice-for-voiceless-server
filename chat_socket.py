from typing import List, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from datetime import datetime
from pytz import timezone
import asyncio
from contextlib import asynccontextmanager

import json

from .database import SessionLocal, engine

from sqlalchemy.orm import Session
from . import crud, schemas, constants
from .devil import DevilManager
from .rag_devil import RagDevil

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


devil = DevilManager()
# devil = RagDevil()
connection_manager = ConnectionManager()


router = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.websocket("/ws/{username}")
async def websocket_endpoint(websocket: WebSocket, username: str, db: Session = Depends(get_db)):

    await connection_manager.connect(websocket)

    try:
        while True:
            data = await websocket.receive_text()

            current_time = datetime.now(timezone('Asia/Seoul')).strftime("%Y-%m-%d %H:%M:%S")
            message = schemas.WSMessageCreate(
                sentTime=current_time,
                sender=username,
                content=data
            )
            crud.log_message(db=db, message=message)

            await connection_manager.broadcast(message.model_dump_json())

            devil.add_user_message(sender=message.sender ,message=message.content)
            if devil.get_counter() >= len(connection_manager.active_connections):

                async def handle_stream(streamed_content: str, isFirstToken: bool = False):
                    devil_message = schemas.WSMessageCreate(
                        content=streamed_content,
                        sender=constants.devil_name,
                        sentTime=current_time,
                        isStream=True,
                        isFirstToken=isFirstToken
                    )
                    await connection_manager.broadcast(devil_message.model_dump_json())
                    
                def handle_stream_complete(completion: str):
                    completed_message = schemas.WSMessageCreate(
                        content=completion,
                        sender=constants.devil_name,
                        sentTime=current_time,
                    )
                    crud.log_message(db=db, message=completed_message)

                await devil.get_streamed_content(streamHandler=handle_stream, 
                                                 completionHandler=handle_stream_complete)

                devil.reset_counter()

    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
