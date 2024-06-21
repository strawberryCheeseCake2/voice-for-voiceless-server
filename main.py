from typing import List

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware


from . import user
from .database import SessionLocal
from . import chat_socket

app = FastAPI()
app.include_router(user.router)
app.include_router(chat_socket.router)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # can alter with time
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def get():
    return "Welcome Home"

# 이름 중복 x, 4명 4번, GPT API, assistant api 사용
