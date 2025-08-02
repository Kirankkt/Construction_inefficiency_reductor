# app.py
import datetime as dt
import tempfile

import pandas as pd
import streamlit as st

from agent_core import (
    Activity,
    parse_excel_schedule,
    infer_sequential_dependencies,
    compute_cpm,
    apply_calendar,
    estimate_hours_and_cost,
    compute_contingency,
    earned_value,
    daily_workload_by_trade,
    gaps_by_area,
    fragmentation_by_trade,
    improvement_suggestions,
    make_gantt,
)
from db import (
    init_db,
    create_project,
    upsert_tasks,
    fetch_project,
    fetch_tasks,
    latest_percent_by_task_id,
    save_progress_updates,
)

st.set_page_config(page_title="Construction Manager — Kerala", layout="wide")


def parse_and_save(upload_file, project_name, start_date, hours_per_day, base_rate_inr, labour_burden, inefficiency, contingency) -> int:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        tmp.write(upload_file.read())
        tmp_path = tmp.name

    acts, _meta = parse_excel_schedule(tmp_path)
    if not acts:
        raise ValueError("No tasks were parsed. Ensure a sheet contains 'Day 1 ... Day N' headers and non-empty cells.")

    deps = infer_sequential_dependencies(len(acts))
    compute_cpm(acts, deps)
    apply_calendar(acts, start_date)

    df = pd.DataFrame({
        "Activity": [a.name for a in acts],
        "Area": [a.area for a in acts],
        "Trade": [a.trade for a in acts],
        "Description": [a.description for a in acts],
        "Start": [a.start_date for a in acts],
        "Finish": [a.end_date for a in acts],
        "Duration (days)": [a.duration for a in acts],
        "ES": [a.es for a in acts],
        "EF": [a.ef for a in acts],
        "LS": [a.ls for a in acts],
        "LF": [a.lf for a in acts],
        "Slack": [a.slack for a in acts],
        "Critical?": [a.is_critical for a in acts],
    })

    pid = create_project(project_name, start_date, hours_per_day, base_rate_inr, labour_burden, inefficiency, contingency)
    upsert_tasks(pid, df)
    return pid


def normalize_task_columns(df_tasks: pd.DataFrame) -> pd.DataFrame:
    """Robustly map snake_case to display names; create missing columns safely."""
    rename_map = {
        "area_name": "Area",
        "trade": "Trade",
        "description": "Description",
        "planned_start": "Start",
        "planned_finish": "Finish",
        "duration_days": "Duration (days)",
        "es": "ES",
        "ef": "EF",
        "ls": "LS",
        "lf": "LF",
        "slack": "Slack",
        "is_critical": "Critical?",
    }
    df = df_tasks.rename(columns=rename_map).copy()
    for col in ["Area", "Trade", "Description", "Start", "Finish",
                "Duration (days)", "ES", "EF", "LS", "LF", "Slack", "Critical?"]:
        if col not in df.columns:
            if col in ["Start", "Finish"]:
                df[col] = pd.NaT
            elif col == "Critical?":
                df[col] = False
            elif col in ["Description", "Area", "Trade"]:
                df[col] = ""
            else:
                df[col] = 0
    return df


def main():
    init_db()
    st.title("AI Construction Manager (CPM + Progress + EV) — Kerala")

    with st.sidebar:
        st.header("Setup")
        project_name = st.text_input("Project name", "B13 Remodeling")
        start_date = st.date_input("Project start date", value=dt.date(2025, 8, 1))
        hours_per_day = st.number_input("Hours per day", 1.0, 12.0, 8.0, 1.0)
        base_rate_inr = st.number_input("Base labour rate (₹/hour)", 50.0, 2000.0, 112.0, 1.0)
        labour_burden = st.number_input("Labour burden (0–1)", 0.0, 1.0, 0.20, 0.01, format="%.2f")
        inefficiency = st.number_input("Inefficiency (0–1)", 0.0, 1.0, 0.20, 0.01, format="%.2f")
        contingency = st.number_input("Contingency (0–1)", 0.0, 1.0, 0.07, 0.01, format="%.2f")
        upload = st.file_uploader("Upload colour-coded Excel", type=["xlsx"])

        with st.expander("What do these knobs mean?"):
            st.markdown(
                "- **Labour burden**: employer costs (PF/ESI/insurance/bonuses) beyond wages.\n"
                "- **Inefficiency**: lost time due to coordination/rework/waiting; reducing this is a core goal.\n"
                "- **Contingency**: reserve (~7–8%) for unforeseen conditions in renovations."
            )

    if "pid" not in st.session_state:
        st.session_state.pid = None

    if upload and st.button("Generate Baseline (Parse + Save)"):
        try:
            st.session_state.pid = parse_and_save(
                upload, project_name, start_date,
                hours_per_day, base_rate_inr, labour_burden, inefficiency, contingency
            )
            st.success("Baseline saved.")
        except ValueError as e:
            st.error(str(e))
            return

    if st.session_state.pid is None:
        st.info("Upload the Excel and click **Generate Baseline (Parse + Save)**.")
        return

    pid = st.session_state.pid
    proj = fetch_project(pid)
    df_tasks_raw = fetch_tasks(pid)
    if df_tasks_raw.empty:
        st.warning("No tasks exist for this project. Re-upload the Excel and try again.")
        return
    df_tasks = normalize_task_columns(df_tasks_raw)

    st.subheader("Planned Activities (with CPM)")
    # Filters for readability
    col1, col2, col3 = st.columns(3)
    with col1:
        areas = sorted([a for a in df_tasks["Area"].dropna().unique().tolist() if a != ""])
        areas_sel = st.multiselect("Filter by Area", areas, default=areas[:5] if areas else [])
    with col2:
        trades = sorted([t for t in df_tasks["Trade"].dropna().unique().tolist() if t != ""])
        trades_sel = st.multiselect("Filter by Trade", trades, default=trades[:5] if trades else [])
    with col3:
        max_rows = st.slider("Max tasks on Gantt", min_value=10, max_value=200, value=60, step=10)

    df_view = df_tasks.copy()
    if areas_sel:
        df_view = df_view[df_view["Area"].isin(areas_sel)]
    if trades_sel:
        df_view = df_view[df_view["Trade"].isin(trades_sel)]
    df_view = df_view.sort_values(["Start", "Area", "Trade"]).head(max_rows)

    st.dataframe(df_view, use_container_width=True, height=400)

    # Rebuild activities from DB for analytics/costs/EV
    acts: list[Activity] = []
    for _, r in df_tasks.iterrows():
        a = Activity(
            name=r["name"],
            area=r["Area"],
            trade=r["Trade"],
            description=r.get("Description", ""),
            start_day=1,
            end_day=1,
            duration=int(r["Duration (days)"]),
            es=int(r["ES"]),
            ef=int(r["EF"]),
            ls=int(r["LS"]),
            lf=int(r["LF"]),
            slack=int(r["Slack"]),
            is_critical=bool(r["Critical?"]),
            start_date=pd.to_datetime(r["Start"]).date(),
            end_date=pd.to_datetime(r["Finish"]).date(),
        )
        acts.append(a)

    total_hours, total_cost = estimate_hours_and_cost(
        acts, proj["hours_per_day"], proj["base_rate_inr"], proj["labour_burden"], proj["inefficiency"]
    )
    total_with_cont = total_cost + compute_contingency(total_cost, proj["contingency"])

    st.markdown(
        f"**Total hours:** {total_hours:.1f} h &nbsp;&nbsp; "
        f"**Total labour cost:** ₹{total_cost:,.2f} &nbsp;&nbsp; "
        f"**Incl. contingency:** ₹{total_with_cont:,.2f}"
    )

    # Progress entry
    st.markdown("### Progress Updates")
    pct_by_id = latest_percent_by_task_id(list(df_tasks["id"].astype(int)))
    pct_by_name = {r["name"]: pct_by_id.get(int(r["id"]), 0.0) for _, r in df_tasks.iterrows()}

    today = st.date_input("Update date", value=dt.date.today())
    mask_today = (pd.to_datetime(df_tasks["Start"]).dt.date <= today) & (pd.to_datetime(df_tasks["Finish"]).dt.date >= today)
    todays = df_tasks.loc[mask_today, ["id", "name", "Area", "Trade"]].copy()
    todays["Percent Complete"] = todays["id"].map(pct_by_id).fillna(0.0)
    todays["Notes"] = ""
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Mark all **today's scheduled** tasks 100%"):
            todays["Percent Complete"] = 100.0
    with c2:
        st.caption("Or edit below, then **Save progress**.")
    edited = st.data_editor(todays.rename(columns={"id": "task_id"}), use_container_width=True, key="progress_editor")
    if st.button("Save progress"):
        save_progress_updates(today, edited)
        st.success("Progress saved.")
        st.rerun()

    # EV / Schedule Health
    st.markdown("### Earned Value & Schedule Health")
    ev = earned_value(
        acts, today, proj["hours_per_day"], proj["base_rate_inr"], proj["labour_burden"], proj["inefficiency"], pct_by_name
    )
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("SPI", f"{ev['SPI']:.2f}")
    k2.metric("CPI", f"{ev['CPI']:.2f}")
    k3.metric("SV (₹)", f"{ev['SV']:,.0f}")
    k4.metric("CV (₹)", f"{ev['CV']:,.0f}")

    # Inefficiency diagnostics & suggestions
    st.markdown("### Inefficiency Diagnostics")
    wl = daily_workload_by_trade(acts, proj["start_date"])
    gapdf = gaps_by_area(acts)
    fragdf = fragmentation_by_trade(acts)
    colA, colB, colC = st.columns(3)
    with colA:
        st.caption("Daily workload by trade (tasks/day)")
        st.dataframe(wl.head(50), use_container_width=True, height=250)
    with colB:
        st.caption("Largest idle gaps by area")
        st.dataframe(gapdf.head(20), use_container_width=True, height=250)
    with colC:
        st.caption("Trade fragmentation (segments count)")
        st.dataframe(fragdf.head(20), use_container_width=True, height=250)

    tips = improvement_suggestions(acts, proj["start_date"])
    if tips:
        st.info("**Suggestions to reduce inefficiencies:**\n\n- " + "\n- ".join(tips))

    # Gantt (clean, filtered, with progress overlay)
    st.markdown("### Gantt (Filtered)")
    f_acts: list[Activity] = []
    for _, r in df_view.iterrows():
        a = Activity(
            name=r["name"],
            area=r["Area"],
            trade=r["Trade"],
            description=r.get("Description", ""),
            start_day=1,
            end_day=1,
            duration=int(r["Duration (days)"]),
            es=int(r["ES"]),
            ef=int(r["EF"]),
            ls=int(r["LS"]),
            lf=int(r["LF"]),
            slack=int(r["Slack"]),
            is_critical=bool(r["Critical?"]),
            start_date=pd.to_datetime(r["Start"]).date(),
            end_date=pd.to_datetime(r["Finish"]).date(),
        )
        f_acts.append(a)

    fig = make_gantt(f_acts, {k: v for k, v in pct_by_name.items() if k in df_view["name"].tolist()}, height_per_task=26)
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
