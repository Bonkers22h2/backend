from __future__ import annotations

from datetime import datetime
from pathlib import Path

from sqlalchemy import DateTime, Integer, LargeBinary, String, Text, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


PROJECT_ROOT = Path(__file__).resolve().parent
AUDIT_DB_PATH = (PROJECT_ROOT / "audit.db").resolve()  # absolute path as requested

SQLALCHEMY_DATABASE_URL = f"sqlite:///{AUDIT_DB_PATH.as_posix()}"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    encrypted_student_no: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    encrypted_input: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    recommendation: Mapped[str] = mapped_column(Text, nullable=False)
    health_status: Mapped[str | None] = mapped_column(Text, nullable=True)
    actionable_advice: Mapped[str | None] = mapped_column(Text, nullable=True)
    signature: Mapped[str] = mapped_column(String(64), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)

    # Lightweight schema migration for existing SQLite files
    with engine.begin() as conn:
        cols = conn.execute(text("PRAGMA table_info(audit_log)")).fetchall()
        existing = {row[1] for row in cols}  # row[1] = column name

        if "health_status" not in existing:
            conn.execute(text("ALTER TABLE audit_log ADD COLUMN health_status TEXT"))
        if "actionable_advice" not in existing:
            conn.execute(text("ALTER TABLE audit_log ADD COLUMN actionable_advice TEXT"))


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
