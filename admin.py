from fastapi import APIRouter, Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel


from sqlalchemy.orm import Session

import crud
import models
import schemas
from database import get_db
from devil_rag import RagDevil

from devil import get_devil

import uuid


# Router
router = APIRouter()

@router.get("/admin/download-db/", response_class=FileResponse)
def download_db():
    db_path = "sql_app.db"
    return FileResponse(path=db_path)

@router.get("/admin/inspect-history/")
def show_devil_history(devil: RagDevil = Depends(get_devil)):
    
    history = devil.history
    print(history)
    return history

@router.get("/admin/inspect-history/")
def show_devil_history(devil: RagDevil = Depends(get_devil)):
    
    history = devil.get_history()
    return history

@router.post("/admin/reset-history/")
def reset_devil_history(for_real: bool, devil: RagDevil = Depends(get_devil)):
    if for_real:
        devil.reset_history()
    return devil.get_history()

@router.get("/admin/inspect-secret-dms/")
def show_secret_dms(db: Session = Depends(get_db)):
    
    dms = crud.get_all_secret_dms(db=db)
    return dms

@router.post("/admin/reset-secret-dms/")
def reset_devil_history(for_real: bool, db: Session = Depends(get_db)):
    if for_real:
        crud.reset_secret_dms(db=db)
    return crud.get_all_secret_dms(db=db)