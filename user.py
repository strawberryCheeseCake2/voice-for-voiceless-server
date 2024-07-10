from fastapi import APIRouter, Depends, FastAPI, HTTPException
from pydantic import BaseModel


from sqlalchemy.orm import Session

from . import crud, models, schemas
from .database import get_db

import uuid


# Router
router = APIRouter()

@router.post("/users/", response_model=schemas.User)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_name(db, name=user.name)
    if db_user:
        raise HTTPException(status_code=400, detail="Username was already taken")
    return crud.create_user(db=db, user=user)

@router.get("/users/{username}")
def validate_user(username: str, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_name(db, name=username)

    if db_user is None:
        return False
    
    return True
        
