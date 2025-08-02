# agent_core.py
import os, re, datetime as dt
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import pandas as pd
from openpyxl import load_workbook
import plotly.express as px
import plotly.graph_objects as go

# -------------------------------
# Data model
# -------------------------------
@dataclass
class Activity:
    name: str
    area: str
    trade: str
    description: str
    start_day: int
    end_day: int
    duration: int
    # CPM (relative day index, 1-based)
    es: Optional[int] = None
    ef: Optional[int] = None
    ls: Optional[int] = None
    lf: Optional[int] = None
    slack: Optional[int] = None
    is_critical: bool = False
    # Calendar dates (set once start date known)
    start_date: Optional[dt.date] = None
    end_date: Optional[dt.date] = None

# -------------------------------
# Excel parsing
# -------------------------------
def _find_header_row(ws) -> int:
    # Search for "Day 1" in the worksheet; fallback to 2
    for r in range(1, ws.max_row + 1):
        for c in range(1, ws.max_column + 1):
            v = ws.cell(r, c).value
            if isinstance(v, str) and re.match(r"(?i)^day\\s*1$", v.strip()):
                return r
    # Fallback scan: a row with >=2 "Day N" headers
    for r in range(1, ws.max_row + 1):
        hits = 0
        for c in range(1, ws.max_column + 1):
            v = ws.cell(r, c).value
            if isinstance(v, str) and re.match(r"(?i)^day\\s*\\d+$", v.strip()):
                hits += 1
        if hits >= 2:
            return r
    return 2

def _day_columns(ws, header_row):
    day_cols, labels = [], []
    for c in range(1, ws.max_column + 1):
        v = ws.cell(header_row, c).value
        if isinstance(v, str) and re.match(r"(?i)^day\\s*\\d+$", v.strip()):
            day_cols.append(c)
            labels.append(v.strip())
    return day_cols, labels

def parse_excel_schedule(path: str) -> Tuple[List[Activity], Dict[str, int]]:
    """
    Interpret each column labeled Day N as one day.
    Row label in column A is the Area. Non-empty cells across days mean planned work.
    Each contiguous island across days becomes one Activity (segment).
    'Trade' parsed as prefix before ':'; description = full cell text.
    Returns (activities, meta) where meta has 'num_days'.
    """
    wb = load_workbook(path, data_only=True)
    ws = wb.active

    header_row = _find_header_row(ws)
    day_cols, labels = _day_columns(ws, header_row)
    num_days = len(day_cols)

    activities: List[Activity] = []
    for r in range(header_row + 1, ws.max_row + 1):
        area = ws.cell(r, 1).value
        if area is None or str(area).strip() == "":
            continue
        area = str(area).strip()

        # Row vector of texts under days
        texts = []
        flags = []
        for c in day_cols:
            v = ws.cell(r, c).value
            txt = str(v).strip() if isinstance(v, str) else ""
            texts.append(txt)
            flags.append(bool(txt))

        # Split into contiguous segments
        segs = []
        in_seg = False
        s = None
        for j, flag in enumerate(flags, start=1):
            if flag and not in_seg:
                in_seg = True
                s = j
            elif not flag and in_seg:
                segs.append((s, j-1))
                in_seg = False
        if in_seg:
            segs.append((s, len(flags)))

        # Build activities
        for (sd, ed) in segs:
            # trade/description from first non-empty cell in segment
            trade = "General"
            desc = ""
            for d in range(sd, ed + 1):
                cell_txt = texts[d-1]
                if cell_txt:
                    desc = cell_txt
                    m = re.match(r"^\\s*([A-Za-z ]+)\\s*:", cell_txt)
                    if m:
                        trade = m.group(1).strip().title()
                    break
            name = f"{area} — {trade}"
            activities.append(Activity(
                name=name,
                area=area,
                trade=trade,
                description=desc,
                start_day=sd,
                end_day=ed,
                duration=ed - sd + 1
            ))

    return activities, {"num_days": num_days, "header_row": header_row, "day_labels": labels}

# -------------------------------
# Dependencies & CPM
# -------------------------------
def infer_sequential_dependencies(n: int) -> Dict[int, List[int]]:
    # Purely sequential by default; replace later with a real precedence editor.
    deps = {i: [] for i in range(n)}
    for i in range(1, n):
        deps[i-1].append(i)
    return deps

def compute_cpm(acts: List[Activity], deps: Dict[int, List[int]]) -> None:
    n = len(acts)
    preds = {i: [] for i in range(n)}
    for u, vs in deps.items():
        for v in vs:
            preds[v].append(u)

    # Forward pass
    for i, a in enumerate(acts):
        if not preds[i]:
            a.es = 1
        else:
            a.es = max(acts[p].ef for p in preds[i]) + 1
        a.ef = a.es + a.duration - 1

    # Backward pass
    project_end = max(a.ef for a in acts) if acts else 0
    for i in reversed(range(n)):
        a = acts[i]
        succs = deps.get(i, [])
        if not succs:
            a.lf = project_end
        else:
            a.lf = min(acts[s].ls for s in succs)
        a.ls = a.lf - a.duration + 1
        a.slack = a.ls - a.es
        a.is_critical = (a.slack == 0)

def apply_calendar(acts: List[Activity], start_date: dt.date) -> None:
    for a in acts:
        a.start_date = start_date + dt.timedelta(days=a.es - 1)
        a.end_date = start_date + dt.timedelta(days=a.ef - 1)

# -------------------------------
# Costs & EV
# -------------------------------
def estimate_hours_and_cost(
    acts: List[Activity],
    hours_per_day: float,
    base_rate_inr: float,
    labour_burden: float,
    inefficiency: float,
) -> Tuple[float, float]:
    total_hours = sum(a.duration for a in acts) * hours_per_day
    loaded_rate = base_rate_inr * (1 + labour_burden) * (1 + inefficiency)
    total_cost = total_hours * loaded_rate
    return total_hours, total_cost

def compute_contingency(cost: float, contingency: float) -> float:
    return cost * contingency

def earned_value(
    acts: List[Activity],
    today: dt.date,
    hours_per_day: float,
    base_rate_inr: float,
    labour_burden: float,
    inefficiency: float,
    pct_by_task: Dict[str, float],
) -> Dict[str, float]:
    loaded_rate = base_rate_inr * (1 + labour_burden) * (1 + inefficiency)
    PV = EV = AC = 0.0
    for a in acts:
        budget = a.duration * hours_per_day * loaded_rate
        if a.end_date < today:
            planned_pct = 1.0
        elif a.start_date > today:
            planned_pct = 0.0
        else:
            planned_days = (today - a.start_date).days + 1
            planned_pct = min(1.0, max(0.0, planned_days / a.duration))
        actual_pct = min(1.0, max(0.0, pct_by_task.get(a.name, 0.0) / 100.0))
        PV += budget * planned_pct
        EV += budget * actual_pct
        AC += budget * actual_pct  # proxy unless you collect real hours
    SPI = (EV / PV) if PV > 0 else 0.0
    CPI = (EV / AC) if AC > 0 else 0.0
    return {"PV": PV, "EV": EV, "AC": AC, "SV": EV - PV, "CV": EV - AC, "SPI": SPI, "CPI": CPI}

# -------------------------------
# Inefficiency diagnostics
# -------------------------------
def daily_workload_by_trade(acts: List[Activity], start_date: dt.date) -> pd.DataFrame:
    """Return dataframe with date, trade, task_count for concurrent work -> helps spot overload/underload & switching."""
    rows = []
    for a in acts:
        for d in range(a.es, a.ef + 1):
            rows.append({
                "date": start_date + dt.timedelta(days=d-1),
                "trade": a.trade,
                "task": a.name
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["date","trade","task_count"])
    return df.groupby(["date","trade"]).agg(task_count=("task","nunique")).reset_index()

def gaps_by_area(acts: List[Activity]) -> pd.DataFrame:
    """Identify idle gaps between segments per area to suggest batching/re-sequencing."""
    df = pd.DataFrame([{"area":a.area,"es":a.es,"ef":a.ef,"name":a.name} for a in acts]).sort_values(["area","es"])
    rows = []
    for area, g in df.groupby("area"):
        prev_ef = None
        for _, r in g.iterrows():
            if prev_ef is not None and r["es"] > prev_ef + 1:
                rows.append({"area": area, "gap_days": r["es"] - prev_ef - 1, "after_task": r["name"]})
            prev_ef = r["ef"]
    return pd.DataFrame(rows).sort_values(["area","gap_days"], ascending=[True, False])

def fragmentation_by_trade(acts: List[Activity]) -> pd.DataFrame:
    """Count how many segments per trade; high counts imply context switching."""
    df = pd.DataFrame([{"trade":a.trade, "area":a.area, "name":a.name} for a in acts])
    if df.empty: 
        return pd.DataFrame(columns=["trade","segments"])
    return df.groupby("trade").agg(segments=("name","nunique")).reset_index().sort_values("segments", ascending=False)

def improvement_suggestions(acts: List[Activity], start_date: dt.date) -> List[str]:
    tips = []
    # 1) Minimize trade switches per day
    wl = daily_workload_by_trade(acts, start_date)
    if not wl.empty:
        peaks = wl.groupby("trade")["task_count"].max().sort_values(ascending=False)
        busy = list(peaks.head(3).items())
        tips.append(f"Trade load peaks (max concurrent tasks): " + ", ".join([f"{t}: {c}" for t,c in busy]))
    # 2) Large idle gaps by area
    gaps = gaps_by_area(acts)
    if not gaps.empty:
        worst = gaps.head(5)
        tips.append("Largest idle gaps by area (consider batching tasks): " + "; ".join([f"{r.area}: {int(r.gap_days)}d" for r in worst.itertuples()]))
    # 3) Fragmentation by trade
    frag = fragmentation_by_trade(acts)
    if not frag.empty:
        top = frag.head(3)
        tips.append("Most fragmented trades (reduce context switching): " + ", ".join([f"{r.trade} ({int(r.segments)} segments)" for r in top.itertuples()]))
    # 4) Critical path compression
    crit_count = sum(1 for a in acts if a.is_critical)
    if crit_count:
        tips.append(f"{crit_count} tasks are on the critical path. Focus progress and crew continuity there to reduce overall duration.")
    return tips

# -------------------------------
# Plotly Gantt (clean + progress)
# -------------------------------
def make_gantt(acts: List[Activity], pct_by_task: Dict[str, float], height_per_task: int = 24) -> go.Figure:
    rows = []
    for a in acts:
        rows.append({
            "Task": a.name, "Type": "Planned", "Start": a.start_date, "Finish": a.end_date,
            "Area": a.area, "Trade": a.trade, "Critical": a.is_critical, "Percent": 100
        })
        pct = min(100.0, max(0.0, pct_by_task.get(a.name, 0.0)))
        days = round(pct/100.0 * a.duration)
        prog_end = max(a.start_date, a.start_date + dt.timedelta(days=max(0, days-1)))
        rows.append({
            "Task": a.name, "Type": "Progress", "Start": a.start_date, "Finish": prog_end,
            "Area": a.area, "Trade": a.trade, "Critical": a.is_critical, "Percent": pct
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return go.Figure()
    fig = px.timeline(
        df, x_start="Start", x_end="Finish", y="Task", color="Type",
        color_discrete_map={"Planned":"#D3D3D3", "Progress":"#2ca02c"},
        hover_data=["Area","Trade","Percent","Start","Finish"]
    )
    fig.update_yaxes(autorange="reversed")
    # annotate % on progress
    for tr in fig.data:
        if tr.name == "Progress":
            mask = (df["Type"]=="Progress")
            tr.text = [f"{p:.0f}%" for p in df.loc[mask, "Percent"]]
            tr.textposition = "inside"
            tr.insidetextanchor = "middle"
    # outline critical planned bars
    for tr in fig.data:
        if tr.name == "Planned":
            tr.marker.line.color = "red"
            tr.marker.line.width = 1.2
    # today line
    today = dt.date.today()
    fig.add_shape(type="line", x0=today, x1=today, y0=-0.5, y1=len(df["Task"].unique())+0.5,
                  line=dict(color="orange", width=2, dash="dash"))
    # dynamic height
    uniq_tasks = df["Task"].nunique()
    fig.update_layout(height=max(400, min(1400, uniq_tasks * height_per_task)),
                      margin=dict(l=200,r=40,t=50,b=40),
                      title="Gantt (Planned vs Progress) — critical outlined in red",
                      legend_title_text="")
    return fig
