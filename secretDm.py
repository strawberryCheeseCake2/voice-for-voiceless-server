from fastapi import APIRouter, Depends, FastAPI, HTTPException
from pydantic import BaseModel

from datetime import datetime
from pytz import timezone

from sqlalchemy.orm import Session

import crud, models, schemas
from database import get_db

import uuid


# Router
router = APIRouter()

@router.post("/secretDms/", response_model=schemas.MessageCreate)
def create_secret_dm(dm: schemas.MessageCreate, db: Session = Depends(get_db)):

    if dm.sentTime is None:
      dm.sentTime = datetime.now(timezone('Asia/Seoul')).strftime("%Y-%m-%d %H:%M:%S")
    return crud.log_secretDm(db=db, message=dm)

