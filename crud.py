from sqlalchemy.orm import Session

from . import models, schemas

def get_user_by_name(db: Session, name: str):
    return db.query(models.User).filter(models.User.name == name).first()

def create_user(db: Session, user: schemas.UserCreate):
    db_user = models.User(name=user.name)
    db.add(db_user) # add sqlalchemy model instance object to your database session.
    db.commit() # save changes
    db.refresh(db_user) # so that it contains any new data from the database, like the generated ID
    return db_user

