from typing import List, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from datetime import datetime
from pytz import timezone
import asyncio
from contextlib import asynccontextmanager

import json

from database import SessionLocal, engine, get_db

from sqlalchemy.orm import Session
import crud
import schemas
import constants
# from devil import DevilManager
from devil_rag import RagDevil
from devil import get_devil


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


connection_manager = ConnectionManager()


router = APIRouter()


@router.websocket("/ws/{username}")
async def websocket_endpoint(
    websocket: WebSocket, 
    username: str, 
    db: Session = Depends(get_db), 
    devil: RagDevil = Depends(get_devil),
    ):

    await connection_manager.connect(websocket)

    try:
        while True:
            data = await websocket.receive_text()

            current_time = datetime.now(
                timezone('Asia/Seoul')).strftime("%Y-%m-%d %H:%M:%S")
            message = schemas.WSMessageCreate(
                sentTime=current_time,
                sender=username,
                content=data
            )
            crud.log_message(db=db, message=message)

            await connection_manager.broadcast(message.model_dump_json())

            devil.add_user_message(sender=message.sender,
                                   message=message.content)
            if devil.is_enabled() and (devil.get_counter() >= 5 * len(connection_manager.active_connections)):
                print("CA Triggered!!!")
                asyncio.create_task(generate_counterarg(devil=devil, db=db))
                devil.reset_counter()


    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)

async def generate_counterarg(devil: RagDevil, db: Session):

    # Request couterargument
    completion = await devil.get_critique()

    if completion is not None: 

        # Add counterargument to db & broadcast

        current_time = datetime.now(
                timezone('Asia/Seoul')).strftime("%Y-%m-%d %H:%M:%S")

        db_message = schemas.WSMessageCreate(
            content=completion,
            sender=constants.devil_name,
            sentTime=current_time
        )

        await connection_manager.broadcast(db_message.model_dump_json())

        crud.log_message(db=db, message=db_message)

        await devil.remove_used_dm(completion)


    
