"""Persistent storage of every prediction (PostgreSQL in Kubernetes, SQLite locally)."""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True,
                                                 default=lambda: datetime.now(timezone.utc))
    model_name: Mapped[str] = mapped_column(String(64))
    model_version: Mapped[str] = mapped_column(String(32), index=True)
    input: Mapped[dict] = mapped_column(JSON)
    risk_probability: Mapped[float] = mapped_column(Float)
    above_threshold: Mapped[bool] = mapped_column(Boolean)
    risk_level: Mapped[str] = mapped_column(String(16))
    top_factors: Mapped[list] = mapped_column(JSON)


def database_url() -> str:
    """DATABASE_URL wins; otherwise build it from the POSTGRES_* variables (Kubernetes Secret)."""
    if url := os.getenv("DATABASE_URL"):
        return url
    if os.getenv("POSTGRES_HOST"):
        user = os.environ["POSTGRES_USER"]
        password = os.environ["POSTGRES_PASSWORD"]
        host = os.environ["POSTGRES_HOST"]
        port = os.getenv("POSTGRES_PORT", "5432")
        db = os.environ["POSTGRES_DB"]
        return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{db}"
    # Fallback for local development / a single container without Postgres
    return f"sqlite:///{os.getenv('SQLITE_PATH', './predictions.db')}"


class Database:
    def __init__(self, url: str):
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        self.engine = create_engine(url, pool_pre_ping=True, connect_args=connect_args)
        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False)

    def init(self, retries: int = 15, delay: float = 2.0) -> None:
        """Create the table. Retries because Postgres may still be starting in Kubernetes."""
        for attempt in range(1, retries + 1):
            try:
                Base.metadata.create_all(self.engine)
                return
            except Exception as exc:  # noqa: BLE001 - log and retry any connection error
                print(f"Database not ready (attempt {attempt}/{retries}): {exc.__class__.__name__}")
                time.sleep(delay)
        raise RuntimeError("Could not connect to the database")

    def is_up(self) -> bool:
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:  # noqa: BLE001
            return False

    def session(self) -> Session:
        return self.SessionLocal()
