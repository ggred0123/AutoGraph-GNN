from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base


# 메인 DB 연결
MAIN_DB_URL = "postgresql://postgres.vwmtnjgbuhdtponnubuw:ruaend22123@aws-0-ap-northeast-2.pooler.supabase.com:5432/postgres"
main_engine = create_engine(MAIN_DB_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=main_engine)

Base = declarative_base()


def get_db_connection():
    """메인 DB 연결을 반환"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

