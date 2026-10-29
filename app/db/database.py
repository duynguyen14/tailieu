from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import URL, Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import PROJECT_ROOT, Settings


class Base(DeclarativeBase):
    pass


def _runtime_database_url(configured_url: str) -> str:
    """Resolve a relative SQLite URL internally without changing the ENV value."""

    prefix = "sqlite:///./"
    if not configured_url.startswith(prefix):
        return configured_url

    relative_part = configured_url.removeprefix(prefix)
    relative_path = Path(relative_part)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("Relative SQLite DATABASE_URL cannot contain an absolute path or '..'.")
    absolute_path = (PROJECT_ROOT / relative_path).resolve()
    if not absolute_path.is_relative_to(PROJECT_ROOT):
        raise ValueError("SQLite database must stay inside the project root.")
    absolute_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{absolute_path.as_posix()}"


def _sql_server_url(settings: Settings) -> URL:
    query = {
        "driver": settings.db_driver,
        "Encrypt": "yes" if settings.db_encrypt else "no",
        "TrustServerCertificate": ("yes" if settings.db_trust_server_certificate else "no"),
    }
    username = settings.db_user
    password = settings.db_password.get_secret_value() if settings.db_password else None
    if settings.db_trusted_connection:
        username = None
        password = None
        query["Trusted_Connection"] = "yes"

    return URL.create(
        drivername="mssql+pyodbc",
        username=username,
        password=password,
        host=settings.db_server,
        port=settings.db_port,
        database=settings.db_name,
        query=query,
    )


class Database:
    def __init__(self, settings: Settings) -> None:
        runtime_url: str | URL
        if settings.database_url:
            runtime_url = _runtime_database_url(settings.database_url)
        else:
            runtime_url = _sql_server_url(settings)
        engine_kwargs: dict = {"pool_pre_ping": True}
        runtime_url_text = str(runtime_url)
        if runtime_url_text == "sqlite:///:memory:":
            engine_kwargs.update(
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
            )
        elif runtime_url_text.startswith("sqlite:"):
            engine_kwargs["connect_args"] = {"check_same_thread": False}

        self.engine: Engine = create_engine(runtime_url, **engine_kwargs)
        self._session_factory = sessionmaker(
            bind=self.engine,
            autoflush=False,
            expire_on_commit=False,
            class_=Session,
        )

    def create_tables(self) -> None:
        from app.db import models  # noqa: F401

        Base.metadata.create_all(self.engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def close(self) -> None:
        self.engine.dispose()
