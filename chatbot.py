# chatbot.py
import datetime as dt
from decimal import Decimal
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
from openai import OpenAI

from agent_core import (
    Activity, estimate_hours_and_cost, compute_contingency, earned_value,
    daily_workload_by_trade, gaps_by_area, fragmentation_by_trade,
)
from db import fetch_project, fetch_tasks


# --------- utilities ----------
def _f(x) -> float:
    if isinstance(x, Decimal):
        return float(x)
    if isinstance(x, (int, float)):
        return float(x)
    try:
        return float(x)
    except Exception:
        return 0.0


def _fmt_date(x) -> str:
    try:
        return pd.to_datetime(x).date().isoformat()
    except Exception:
        return str(x)


def _sget(r: pd.Series, *names, default=0):
    """Series-get: return first present name (works for ES/ef etc.)."""
    for n in names:
        if n in r.index:
            return r[n]
    return default


# --------- context builder ----------
def build_context(project_id: int) -> Tuple[str, List[Dict]]:
    """
    Compose a compact, grounded context string from DB that the model must rely on.
    Also return 'sources' we can show as citations in the UI.
    """
    proj = fetch_project(project_id)
    tasks_df = fetch_tasks(project_id)

    if not proj:
        return "No active project selected.", []
    if tasks_df.empty:
        return f"Project '{proj['name']}' has no tasks in the database.", []

    # Normalise names exactly as app.py expects
    rename_map = {
        "area_name": "Area", "planned_start": "Start", "planned_finish": "Finish",
        "duration_days": "Duration (days)", "is_critical": "Critical?",
    }
    df = tasks_df.rename(columns=rename_map).copy()

    # Build Activity list (robust to ES/EF/LS/LF/Slack vs es/ef/ls/lf/slack)
    acts: List[Activity] = []
    for _, r in df.iterrows():
        acts.append(Activity(
            name=r["name"],
            area=r.get("Area", ""),
            trade=r.get("Trade", ""),
            description=r.get("Description", ""),
            start_day=1, end_day=1,
            duration=int(_sget(r, "Duration (days)", "duration_days", default=1)),
            es=int(_sget(r, "ES", "es", default=0)),
            ef=int(_sget(r, "EF", "ef", default=0)),
            ls=int(_sget(r, "LS", "ls", default=0)),
            lf=int(_sget(r, "LF", "lf", default=0)),
            slack=int(_sget(r, "Slack", "slack", default=0)),
            is_critical=bool(_sget(r, "Critical?", "is_critical", default=False)),
            start_date=pd.to_datetime(r["Start"]).date(),
            end_date=pd.to_datetime(r["Finish"]).date(),
        ))

    # Project parameters
    hours = _f(proj["hours_per_day"])
    rate  = _f(proj["base_rate_inr"])
    burden = _f(proj["labour_burden"])
    ineff  = _f(proj["inefficiency"])
    cont   = _f(proj["contingency"])

    # Costs + EV (baseline view as of today; actual progress can be added later)
    today = dt.date.today()
    total_hours, total_cost = estimate_hours_and_cost(acts, hours, rate, burden, ineff)
    total_with_cont = total_cost + compute_contingency(total_cost, cont)
    pct_by_name: Dict[str, float] = {}  # baseline-only EV for now
    ev = earned_value(acts, today, hours, rate, burden, ineff, pct_by_name)

    # Critical path and upcoming
    crit = df[df["Critical?"] == True].sort_values(["Start", "Area", "Trade"])
    upcoming = df.sort_values("Start").head(20)

    crit_lines = [
        f"[Task {int(r.id)}] {r['name']} | Area={r['Area']} | Trade={r['Trade']} | "
        f"Start={_fmt_date(r['Start'])} | Finish={_fmt_date(r['Finish'])} | "
        f"Dur={int(_sget(r, 'Duration (days)', 'duration_days', default=1))}d | "
        f"Slack={int(_sget(r, 'Slack', 'slack', default=0))}"
        for _, r in crit.iterrows()
    ][:20]

    up_lines = [
        f"[Task {int(r.id)}] {r['name']} | Area={r['Area']} | Trade={r['Trade']} | "
        f"Start={_fmt_date(r['Start'])} | Finish={_fmt_date(r['Finish'])} | "
        f"Dur={int(_sget(r, 'Duration (days)', 'duration_days', default=1))}d | "
        f"Critical={bool(_sget(r, 'Critical?', 'is_critical', default=False))}"
        for _, r in upcoming.iterrows()
    ]

    # Diagnostics
    gapdf = gaps_by_area(acts).head(12)
    fragdf = fragmentation_by_trade(acts).head(12)
    wl = daily_workload_by_trade(acts, proj["start_date"]).head(20)

    gap_lines = [f"{r.area}: {int(r.gap_days)}d idle after '{r.after_task}'" for r in gapdf.itertuples()] if not gapdf.empty else []
    frag_lines = [f"{r.trade}: {int(r.segments)} segments" for r in fragdf.itertuples()] if not fragdf.empty else []
    wl_lines = [f"{r.date}: {r.trade}={int(r.task_count)} tasks" for r in wl.itertuples()] if not wl.empty else []

    # Lightweight date index (today -3 … +21)
    start_idx = today - dt.timedelta(days=3)
    end_idx   = today + dt.timedelta(days=21)
    day_rows = df[(pd.to_datetime(df["Start"]).dt.date <= end_idx) &
                  (pd.to_datetime(df["Finish"]).dt.date >= start_idx)].copy()
    by_day: Dict[dt.date, List[str]] = {}
    for _, r in day_rows.iterrows():
        s, f = pd.to_datetime(r["Start"]).date(), pd.to_datetime(r["Finish"]).date()
        for d in (s + dt.timedelta(days=i) for i in range((f - s).days + 1)):
            if start_idx <= d <= end_idx:
                by_day.setdefault(d, [])
                if len(by_day[d]) < 6:
                    by_day[d].append(f"[{r['Trade']}] {r['name']} ({r['Area']})")

    day_index_lines = []
    for d in sorted(by_day.keys()):
        items = "; ".join(by_day[d])
        day_index_lines.append(f"{d.isoformat()}: {items}")

    context = f"""
PROJECT
- Name: {proj['name']}
- Start: {proj['start_date']}
- Parameters: hours/day={hours}, base ₹/h={rate}, labour_burden={burden}, inefficiency={ineff}, contingency={cont}

COSTS/EV (baseline view as of {today})
- Total hours: {total_hours:.1f}
- Labour cost (excl contingency): ₹{total_cost:,.0f}
- Incl contingency: ₹{total_with_cont:,.0f}
- SPI: {ev['SPI']:.2f} | CPI: {ev['CPI']:.2f} | SV: ₹{ev['SV']:,.0f} | CV: ₹{ev['CV']:,.0f}

CRITICAL PATH (top)
{chr(10).join(crit_lines) if crit_lines else 'None detected.'}

UPCOMING (soonest)
{chr(10).join(up_lines) if up_lines else 'None detected.'}

IDLE GAPS (largest)
{chr(10).join(gap_lines) if gap_lines else 'None detected.'}

TRADE FRAGMENTATION (worst)
{chr(10).join(frag_lines) if frag_lines else 'None detected.'}

WORKLOAD SAMPLES (tasks/day)
{chr(10).join(wl_lines) if wl_lines else 'None.'}

ON-DATE INDEX (today±3 .. today+21)
{chr(10).join(day_index_lines) if day_index_lines else 'No tasks in index window.'}
""".strip()

    # Sources — we cite a few tasks and gaps we used
    sources: List[Dict] = []
    if not crit.empty:
        for _, r in crit.head(6).iterrows():
            sources.append({"type": "task", "id": int(r.id), "name": r["name"]})
    if not gapdf.empty:
        for r in gapdf.head(3).itertuples():
            sources.append({"type": "gap", "area": r.area, "after_task": r.after_task, "gap_days": int(r.gap_days)})

    return context, sources


SYSTEM_PROMPT = """
You are an experienced Construction Project Manager and operations-research minded planner
working on building refurbishment projects in Kerala, India.

RULES
- Answer ONLY using the provided project context. If info is missing, say so.
- Be concrete and data-driven: cite [Task id] + name, areas, trades, dates, ES/EF/LS/LF/slack where relevant.
- Focus on actions that reduce idle gaps, batch trades logically, resequence non-critical tasks,
  protect the critical path, and even out daily workload.
- Explain trade-offs (time vs ₹ vs risk). Use Indian ₹ for currency.
- Keep replies concise and actionable (prefer bullets). Avoid generic textbook talk unless asked.

You can also explain how to use the app features (baseline ingest, filters, progress entry, EV metrics).
""".strip()


def _call_openai(messages: List[Dict], model: str, api_key: str, base_url: str | None = None) -> str:
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=900,
    )
    return resp.choices[0].message.content


def answer_question(project_id: int, user_q: str) -> Tuple[str, List[Dict]]:
    """
    Build grounded context, call LLM, return (answer, sources).
    """
    # Secrets example:
    # [llm]
    # provider="openai"
    # model="gpt-4o-mini"
    # api_key="sk-..."
    # base_url=""  # optional
    llm = st.secrets.get("llm", {})
    provider = llm.get("provider", "openai").lower()
    model = llm.get("model", "gpt-4o-mini")
    api_key = llm.get("api_key", "")
    base_url = llm.get("base_url", None)

    if provider != "openai" or not api_key:
        msg = ("LLM is not configured. Add to secrets:\n\n"
               "[llm]\nprovider = \"openai\"\nmodel = \"gpt-4o-mini\"\napi_key = \"YOUR_KEY\"")
        return msg, []

    ctx, sources = build_context(project_id)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"PROJECT CONTEXT:\n{ctx}\n\nUSER QUESTION:\n{user_q}"},
    ]
    try:
        answer = _call_openai(messages, model=model, api_key=api_key, base_url=base_url)
    except Exception as e:
        answer = f"LLM error: {e}"
    return answer, sources
