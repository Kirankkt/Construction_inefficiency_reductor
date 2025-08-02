# db.py
import datetime as dt
from typing import Dict, List

import pandas as pd
import streamlit as st
from sqlalchemy import (
    create_engine, MetaData, Table, Column, Integer, String, Date, DateTime,
    Boolean, Numeric, Text, func, select, insert, delete
)
from sqlalchemy.engine import Engine
from sqlalchemy.sql import text as sql_text


# ---------- Engine from Streamlit secrets ----------
@st.cache_resource(show_spinner=False)
def get_engine() -> Engine:
    """
    Prefer DATABASE_URL from Streamlit secrets; fall back to env or local SQLite.
    Example secret:
      DATABASE_URL = "postgresql+psycopg2://USER:PASS@HOST:5432/DB?sslmode=require"
    """
    db_url = st.secrets.get("DATABASE_URL", None)
    if not db_url:
        # Fallback (local dev): SQLite file
        db_url = "sqlite:///app.db"
    return create_engine(db_url, pool_pre_ping=True, future=True)


# ---------- Schema (SQLAlchemy Core) ----------
metadata = MetaData()

projects = Table(
    "projects", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", String, nullable=False),
    Column("start_date", Date, nullable=False),
    Column("hours_per_day", Numeric(5, 2), nullable=False, server_default="8"),
    Column("base_rate_inr", Numeric(12, 2), nullable=False, server_default="112"),
    Column("labour_burden", Numeric(5, 2), nullable=False, server_default="0.20"),
    Column("inefficiency", Numeric(5, 2), nullable=False, server_default="0.20"),
    Column("contingency", Numeric(5, 2), nullable=False, server_default="0.07"),
    Column("created_at", DateTime(timezone=True), server_default=func.now()),
)

tasks = Table(
    "tasks", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("project_id", Integer, nullable=False),
    Column("name", String, nullable=False),
    Column("area_name", String),
    Column("trade", String),
    Column("description", Text),
    Column("planned_start", Date, nullable=False),
    Column("planned_finish", Date, nullable=False),
    Column("duration_days", Integer, nullable=False),
    Column("es", Integer), Column("ef", Integer),
    Column("ls", Integer), Column("lf", Integer),
    Column("slack", Integer),
    Column("is_critical", Boolean, server_default="0"),
)

progress_updates = Table(
    "progress_updates", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("task_id", Integer, nullable=False),
    Column("update_date", Date, nullable=False),
    Column("percent_complete", Numeric(5, 2), nullable=False),
    Column("notes", Text),
    Column("created_at", DateTime(timezone=True), server_default=func.now()),
)


def init_db() -> None:
    """Create tables if they don't exist."""
    engine = get_engine()
    metadata.create_all(engine)


# ---------- CRUD helpers ----------
def create_project(
    name: str,
    start_date: dt.date,
    hours_per_day: float,
    base_rate_inr: float,
    labour_burden: float,
    inefficiency: float,
    contingency: float,
) -> int:
    engine = get_engine()
    with engine.begin() as conn:
        stmt = (
            insert(projects)
            .values(
                name=name,
                start_date=start_date,
                hours_per_day=hours_per_day,
                base_rate_inr=base_rate_inr,
                labour_burden=labour_burden,
                inefficiency=inefficiency,
                contingency=contingency,
            )
            .returning(projects.c.id)
        )
        try:
            pid = conn.execute(stmt).scalar_one()
        except Exception:
            # SQLite older versions may not support RETURNING; fallback
            res = conn.execute(
                insert(projects).values(
                    name=name,
                    start_date=start_date,
                    hours_per_day=hours_per_day,
                    base_rate_inr=base_rate_inr,
                    labour_burden=labour_burden,
                    inefficiency=inefficiency,
                    contingency=contingency,
                )
            )
            pid = int(res.inserted_primary_key[0])
    return pid


def upsert_tasks(project_id: int, df: pd.DataFrame) -> None:
    """
    Replace all tasks for a project with df rows.
    Expected df columns: Activity, Area, Trade, Description,
                         Start, Finish, Duration (days), ES, EF, LS, LF, Slack, Critical?
    """
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(delete(tasks).where(tasks.c.project_id == project_id))
        rows = []
        for _, r in df.iterrows():
            rows.append(
                dict(
                    project_id=project_id,
                    name=r["Activity"],
                    area_name=r.get("Area"),
                    trade=r.get("Trade"),
                    description=r.get("Description", ""),
                    planned_start=pd.to_datetime(r["Start"]).date(),
                    planned_finish=pd.to_datetime(r["Finish"]).date(),
                    duration_days=int(r["Duration (days)"]),
                    es=int(r["ES"]), ef=int(r["EF"]),
                    ls=int(r["LS"]), lf=int(r["LF"]),
                    slack=int(r["Slack"]),
                    is_critical=bool(r["Critical?"]),
                )
            )
        if rows:
            conn.execute(insert(tasks), rows)


def fetch_project(project_id: int) -> dict:
    engine = get_engine()
    with engine.begin() as conn:
        row = conn.execute(
            select(projects).where(projects.c.id == project_id)
        ).mappings().first()
    return dict(row) if row else {}


def fetch_tasks(project_id: int) -> pd.DataFrame:
    engine = get_engine()
    with engine.begin() as conn:
        rows = conn.execute(
            select(
                tasks.c.id,
                tasks.c.name,
                tasks.c.area_name.label("Area"),
                tasks.c.trade.label("Trade"),
                tasks.c.description.label("Description"),
                tasks.c.planned_start.label("Start"),
                tasks.c.planned_finish.label("Finish"),
                tasks.c.duration_days.label("Duration (days)"),
                tasks.c.es.label("ES"),
                tasks.c.ef.label("EF"),
                tasks.c.ls.label("LS"),
                tasks.c.lf.label("LF"),
                tasks.c.slack.label("Slack"),
                tasks.c.is_critical.label("Critical?"),
            ).where(tasks.c.project_id == project_id)
            .order_by(tasks.c.planned_start, tasks.c.area_name, tasks.c.trade, tasks.c.name)
        ).mappings().all()
    return pd.DataFrame(rows)


def latest_percent_by_task_id(task_ids: List[int]) -> Dict[int, float]:
    if not task_ids:
        return {}
    engine = get_engine()
    with engine.begin() as conn:
        # For portability across engines, use a simple correlated subquery in text
        q = sql_text(
            """
            SELECT t.id AS task_id,
                   COALESCE((
                       SELECT percent_complete
                       FROM progress_updates pu
                       WHERE pu.task_id = t.id
                       ORDER BY update_date DESC, id DESC
                       LIMIT 1
                   ), 0) AS pct
            FROM tasks t
            WHERE t.id = ANY(:ids)
            """
        )
        rows = conn.execute(q, {"ids": task_ids}).mappings().all()
    return {int(r["task_id"]): float(r["pct"]) for r in rows}


def save_progress_updates(date_sel: dt.date, df_updates: pd.DataFrame) -> None:
    engine = get_engine()
    with engine.begin() as conn:
        to_insert = []
        for _, r in df_updates.iterrows():
            to_insert.append(
                dict(
                    task_id=int(r["task_id"]),
                    update_date=date_sel,
                    percent_complete=float(r["Percent Complete"]),
                    notes=r.get("Notes", None),
                )
            )
        if to_insert:
            conn.execute(insert(progress_updates), to_insert)
