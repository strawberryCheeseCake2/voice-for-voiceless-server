from fastapi import APIRouter, Depends, FastAPI, HTTPException
from pydantic import BaseModel


from sqlalchemy.orm import Session

from . import crud, models, schemas
from .database import SessionLocal, engine

import uuid


models.Base.metadata.create_all(bind=engine)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Router
router = APIRouter()

@router.post("/users/", response_model=schemas.User)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_name(db, name=user.name)
    if db_user:
        
        new_user = schemas.UserCreate(name=f"{user.name}{uuid.uuid1().int}")
    return crud.create_user(db=db, user=new_user)
