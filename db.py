# db.py
import os, pandas as pd, datetime as dt
from typing import Dict, List
from sqlalchemy import create_engine, text

def engine():
    url = os.getenv("DATABASE_URL", "sqlite:///app.db")  # Postgres recommended; SQLite fallback
    return create_engine(url, future=True)

def init_db():
    with engine().begin() as conn:
        conn.exec_driver_sql("""
        CREATE TABLE IF NOT EXISTS projects (
          id SERIAL PRIMARY KEY,
          name TEXT NOT NULL,
          start_date DATE NOT NULL,
          hours_per_day NUMERIC(5,2) NOT NULL DEFAULT 8,
          base_rate_inr NUMERIC(12,2) NOT NULL DEFAULT 112,
          labour_burden NUMERIC(5,2) NOT NULL DEFAULT 0.20,
          inefficiency NUMERIC(5,2) NOT NULL DEFAULT 0.20,
          contingency NUMERIC(5,2) NOT NULL DEFAULT 0.07,
          created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
        """)
        conn.exec_driver_sql("""
        CREATE TABLE IF NOT EXISTS tasks (
          id SERIAL PRIMARY KEY,
          project_id INT REFERENCES projects(id) ON DELETE CASCADE,
          name TEXT NOT NULL,
          area_name TEXT,
          trade TEXT,
          description TEXT,
          planned_start DATE NOT NULL,
          planned_finish DATE NOT NULL,
          duration_days INT NOT NULL,
          es INT, ef INT, ls INT, lf INT, slack INT,
          is_critical BOOLEAN DEFAULT FALSE,
          UNIQUE(project_id, name)
        );
        """)
        conn.exec_driver_sql("""
        CREATE TABLE IF NOT EXISTS progress_updates (
          id SERIAL PRIMARY KEY,
          task_id INT REFERENCES tasks(id) ON DELETE CASCADE,
          update_date DATE NOT NULL,
          percent_complete NUMERIC(5,2) NOT NULL CHECK (percent_complete BETWEEN 0 AND 100),
          notes TEXT,
          created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
        """)
        conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_progress_task_date ON progress_updates(task_id, update_date);")

def create_project(name: str, start_date: dt.date,
                   hours_per_day: float, base_rate_inr: float,
                   labour_burden: float, inefficiency: float, contingency: float) -> int:
    with engine().begin() as conn:
        pid = conn.execute(text("""
            INSERT INTO projects(name, start_date, hours_per_day, base_rate_inr,
                                 labour_burden, inefficiency, contingency)
            VALUES(:n,:s,:h,:r,:b,:i,:c) RETURNING id
        """), {"n":name, "s":start_date, "h":hours_per_day, "r":base_rate_inr,
               "b":labour_burden, "i":inefficiency, "c":contingency}).scalar_one()
    return pid

def upsert_tasks(project_id: int, df: pd.DataFrame) -> None:
    with engine().begin() as conn:
        conn.execute(text("DELETE FROM tasks WHERE project_id=:p"), {"p":project_id})
        for _, r in df.iterrows():
            conn.execute(text("""
                INSERT INTO tasks(project_id, name, area_name, trade, description,
                                  planned_start, planned_finish, duration_days,
                                  es, ef, ls, lf, slack, is_critical)
                VALUES(:p,:n,:a,:t,:d,:ps,:pf,:dur,:es,:ef,:ls,:lf,:sl,:crit)
            """), {
                "p": project_id, "n": r["Activity"], "a": r["Area"],
                "t": r["Trade"], "d": r.get("Description",""),
                "ps": r["Start"], "pf": r["Finish"], "dur": int(r["Duration (days)"]),
                "es": int(r["ES"]), "ef": int(r["EF"]), "ls": int(r["LS"]), "lf": int(r["LF"]),
                "sl": int(r["Slack"]), "crit": bool(r["Critical?"])
            })

def fetch_project(project_id: int) -> dict:
    with engine().begin() as conn:
        row = conn.execute(text("SELECT * FROM projects WHERE id=:p"), {"p":project_id}).mappings().first()
    return dict(row) if row else {}

def fetch_tasks(project_id: int) -> pd.DataFrame:
    with engine().begin() as conn:
        df = pd.read_sql(text("""
            SELECT id, name, area_name AS Area, trade AS Trade, description AS Description,
                   planned_start AS Start, planned_finish AS Finish,
                   duration_days AS "Duration (days)", es AS ES, ef AS EF, ls AS LS, lf AS LF,
                   slack AS Slack, is_critical AS "Critical?"
            FROM tasks WHERE project_id=:p ORDER BY Start, Area, Trade, name
        """), conn, params={"p":project_id})
    return df

def latest_percent_by_task_id(task_ids: List[int]) -> Dict[int, float]:
    if not task_ids:
        return {}
    with engine().begin() as conn:
        rows = conn.execute(text("""
            SELECT t.id AS task_id,
                   COALESCE((SELECT percent_complete
                             FROM progress_updates pu
                             WHERE pu.task_id = t.id
                             ORDER BY update_date DESC, id DESC LIMIT 1), 0) AS pct
            FROM tasks t WHERE t.id = ANY(:ids)
        """), {"ids": task_ids}).mappings().all()
    return {int(r["task_id"]): float(r["pct"]) for r in rows}

def save_progress_updates(date_sel: dt.date, df_updates: pd.DataFrame):
    with engine().begin() as conn:
        for _, r in df_updates.iterrows():
            conn.execute(text("""
                INSERT INTO progress_updates(task_id, update_date, percent_complete, notes)
                VALUES(:tid,:d,:pct,:notes)
            """), {"tid": int(r["task_id"]), "d": date_sel,
                   "pct": float(r["Percent Complete"]), "notes": r.get("Notes", None)})
