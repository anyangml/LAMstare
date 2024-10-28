import os
from typing import List

from sqlalchemy import (
    Column,
    Float,
    Integer,
    String,
    create_engine,
    asc
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv


Base = declarative_base()
load_dotenv()
db_username = os.environ.get("MYSQL_USERNAME")
db_password = os.environ.get("MYSQL_PASSWORD")
db_host = os.environ.get("MYSQL_HOST")
db_name = "crystal_philately"
db = create_engine(
    "mysql+pymysql://%s:%s@%s:3306/%s?charset=utf8"
    % (db_username, db_password, db_host, db_name)
)
Session = sessionmaker(db)


class Record(Base):
    __tablename__ = "multitask_train"

    id = Column(Integer, primary_key=True)
    run_id = Column(String(256), index=True)
    run_name = Column(String(256))
    step = Column(Integer)
    head = Column(String(100))
    energy_rmse = Column(Float)
    energy_mae = Column(Float)
    energy_rmse_natoms = Column(Float)
    energy_mae_natoms = Column(Float)
    force_rmse = Column(Float)
    force_mae = Column(Float)
    virial_rmse = Column(Float)
    virial_mae = Column(Float)
    virial_rmse_natoms = Column(Float)
    virial_mae_natoms = Column(Float)
    
    def __repr__(self):
        return "<Record id=%s step='%s' head='%s' Energy MAE/Natoms=%s>" % (
            self.run_id,
            self.step,
            self.head,
            self.energy_mae_natoms,
        )

    def insert(self) -> int:
        session = Session()
        session.add(self)
        session.flush()
        id = self.id
        session.commit()
        session.close()
        return id

    @classmethod
    def query_by_filter(cls, *criteria) -> List["Record"]:
        session = Session()
        record = session.query(cls).filter(*criteria).all()
        session.close()
        return record

    @classmethod
    def query(cls, **kwargs) -> List["Record"]:
        session = Session()
        records = session.query(cls).filter_by(**kwargs).order_by(asc(cls.step)).all()
        session.close()
        return records

    @classmethod
    def query_by_run(cls, run_id: str) -> List["Record"]:
        return cls.query(run_id=run_id)
    
    @classmethod
    def query_by_name(cls, run_name: str) -> List["Record"]:
        return cls.query(run_name=run_name)
