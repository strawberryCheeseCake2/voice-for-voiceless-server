from typing import List

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware


from . import user
from .database import SessionLocal, engine
from . import chat_socket, models, secretDm


# lt --port 8000 --subdomain voiceforvoiceless

app = FastAPI()
app.include_router(user.router)
app.include_router(chat_socket.router)
app.include_router(secretDm.router)

models.Base.metadata.create_all(bind=engine)

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

