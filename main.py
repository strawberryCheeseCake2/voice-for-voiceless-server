from typing import List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import json

from . import user

app = FastAPI()
app.include_router(user.router)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # can alter with time
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.get("/")
async def get():
    return "Welcome Home"

# 이름 중복 x, 4명 4번, GPT API, assistant api 사용
@app.websocket("/ws/{username}")
# async def websocket_endpoint(websocket: WebSocket, client_id: int):
async def websocket_endpoint(websocket: WebSocket, username: str):
    await manager.connect(websocket)
    now = datetime.now()
    current_time = now.strftime("%H:%M")
    try:
        while True:
            data = await websocket.receive_text()
            
            message = {"sentTime":current_time,"username":username,"message":data}
            print(json.dumps(message))
            await manager.broadcast(json.dumps(message))
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        # message = {"sentTime":current_time,"clientId":client_id,"message":"Offline"}
        # await manager.broadcast(json.dumps(message))