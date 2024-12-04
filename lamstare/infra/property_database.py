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


class PropertyRecord(Base):
    __tablename__ = "property_demo"

    id = Column(Integer, primary_key=True)
    run_id = Column(String(256), index=True)
    run_name = Column(String(256))
    model_version = Column(String(100))
    task_name = Column(String(256))
    step = Column(Integer)
    property_rmse = Column(Float)
    property_mae = Column(Float)


    def __repr__(self):
        return "<PropertyRecord id=%s step='%s' head='%s' TASK = '%s' Property MAE=%s>" % (
            self.run_id,
            self.step,
            self.head,
            self.task_name,
            self.property_mae,
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
    def query_by_filter(cls, *criteria) -> List["PropertyRecord"]:
        session = Session()
        record = session.query(cls).filter(*criteria).all()
        session.close()
        return record

    @classmethod
    def query(cls, **kwargs) -> List["PropertyRecord"]:
        session = Session()
        records = session.query(cls).filter_by(**kwargs).order_by(asc(cls.step)).all()
        session.close()
        return records

    @classmethod
    def query_by_run(cls, run_id: str) -> List["PropertyRecord"]:
        return cls.query(run_id=run_id)

    @classmethod
    def query_by_name(cls, run_name: str) -> List["PropertyRecord"]:
        return cls.query(run_name=run_name)

    @classmethod
    def query_latest_step(cls, run_id:str) -> int:
        return cls.query_by_run(run_id)[-1].step

    @classmethod
    def query_best_by_run(cls, run_id: str) -> List["PropertyRecord"]:
        records = cls.query_by_run(run_id)
        latest_step = cls.query_latest_step(run_id)
        tasks = set([record.task_name for record in records])
        best_records = []
        for task in tasks:
            records_ood = [record for record in records if record.task_name == task and record.step == latest_step]
            best_record = min(records_ood, key=lambda x: (x.property_mae, x.property_rmse))
            best_records.append(best_record)
        return best_records

if __name__ == "__main__":
    print(PropertyRecord.query_best_by_run("1103_shallow_fitting_medium_l6_atton_37head_tanh_40GPU_bs_auto256"))