# app.py
import os, datetime as dt, tempfile
import pandas as pd
import streamlit as st

from agent_core import (
    Activity, parse_excel_schedule, infer_sequential_dependencies, compute_cpm,
    apply_calendar, estimate_hours_and_cost, compute_contingency, earned_value,
    daily_workload_by_trade, gaps_by_area, fragmentation_by_trade, improvement_suggestions,
    make_gantt
)
from db import (
    init_db, list_projects, fetch_project_by_name, get_or_create_project,
    save_schedule_file, upsert_tasks, fetch_project, fetch_tasks,
    latest_percent_by_task_id, save_progress_updates
)
from chatbot import answer_question  # NEW

st.set_page_config(page_title="Construction Manager — Kerala", layout="wide")


def ingest_or_replace_baseline(project_id: int, upload_file, start_date: dt.date) -> None:
    """Persist original Excel (versioned), parse, compute CPM, and replace tasks for this project."""
    file_bytes = upload_file.read()
    save_schedule_file(project_id, upload_file.name, file_bytes)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    acts, meta = parse_excel_schedule(tmp_path)
    deps = infer_sequential_dependencies(len(acts))
    compute_cpm(acts, deps)
    apply_calendar(acts, start_date)

    df = pd.DataFrame({
        "Activity":[a.name for a in acts],
        "Area":[a.area for a in acts],
        "Trade":[a.trade for a in acts],
        "Description":[a.description for a in acts],
        "Start":[a.start_date for a in acts],
        "Finish":[a.end_date for a in acts],
        "Duration (days)":[a.duration for a in acts],
        "ES":[a.es for a in acts], "EF":[a.ef for a in acts],
        "LS":[a.ls for a in acts], "LF":[a.lf for a in acts],
        "Slack":[a.slack for a in acts], "Critical?":[a.is_critical for a in acts],
    })
    upsert_tasks(project_id, df)


def main():
    init_db()
    st.title("AI Construction Manager (CPM + Progress + EV) — Kerala")

    # ---------------- Sidebar: Select/Create Project & Baseline mgmt ----------
    with st.sidebar:
        st.header("Setup")

        proj_rows = list_projects()
        proj_names = [r["name"] for r in proj_rows]

        mode = st.radio("Project selection", ["Use existing", "New project"], horizontal=True)
        if mode == "Use existing" and proj_names:
            project_name = st.selectbox("Active project", proj_names)
        else:
            project_name = st.text_input("New project name", "B13 Remodeling")

        start_date = st.date_input("Project start date", value=dt.date(2025,8,1))
        hours_per_day = st.number_input("Hours per day", 1.0, 12.0, 8.0, 1.0)
        base_rate_inr = st.number_input("Base labour rate (₹/hour)", 50.0, 2000.0, 112.0, 1.0)
        labour_burden = st.number_input("Labour burden (0–1)", 0.0, 1.0, 0.20, 0.01, format="%.2f")
        inefficiency = st.number_input("Inefficiency (0–1)", 0.0, 1.0, 0.20, 0.01, format="%.2f")
        contingency = st.number_input("Contingency (0–1)", 0.0, 1.0, 0.07, 0.01, format="%.2f")

        st.subheader("Baseline")
        upload = st.file_uploader("Upload/replace baseline (Excel)", type=["xlsx"])
        gen_btn = st.button("Ingest/Replace baseline (Parse + Save)")

    # Resolve active project id
    if "pid" not in st.session_state:
        chosen = fetch_project_by_name(project_name)
        st.session_state.pid = chosen["id"] if chosen else None

    # Create or update project + parse if requested
    if gen_btn and upload:
        pid = get_or_create_project(
            project_name, start_date, hours_per_day, base_rate_inr, labour_burden, inefficiency, contingency
        )
        ingest_or_replace_baseline(pid, upload, start_date)
        st.session_state.pid = pid
        st.success("Baseline ingested and saved (tasks + original file version).")

    pid = st.session_state.get("pid")
    if not pid:
        st.info("Pick a project (or ingest a baseline) to continue.")
        return

    # Fetch project + tasks
    proj = fetch_project(pid)
    df_tasks = fetch_tasks(pid)

    if df_tasks.empty:
        st.warning("No tasks found for this project. Upload a baseline to see schedule & analytics.")
        return

    # ---------------- Planned Activities (with CPM) ----------------
    st.subheader("Planned Activities (with CPM)")

    areas = sorted(df_tasks.get("Area", pd.Series(dtype=str)).dropna().unique().tolist())
    trades = sorted(df_tasks.get("Trade", pd.Series(dtype=str)).dropna().unique().tolist())

    col1, col2, col3 = st.columns([1.2, 1.2, 1])
    with col1:
        areas_sel = st.multiselect("Filter by Area", areas, default=areas[:5] if areas else [])
    with col2:
        trades_sel = st.multiselect("Filter by Trade", trades, default=trades[:5] if trades else [])
    with col3:
        max_rows = st.slider("Max tasks on Gantt", min_value=10, max_value=200, value=60, step=10)

    df_view = df_tasks.copy()
    if areas_sel:
        df_view = df_view[df_view["Area"].isin(areas_sel)]
    if trades_sel:
        df_view = df_view[df_view["Trade"].isin(trades_sel)]
    df_view = df_view.sort_values(["Start","Area","Trade"]).head(max_rows)

    st.dataframe(df_view, use_container_width=True, height=420)

    # Reconstruct activities
    acts = []
    for _, r in df_tasks.iterrows():
        acts.append(Activity(
            name=r["name"], area=r.get("Area",""), trade=r.get("Trade",""), description=r.get("Description",""),
            start_day=1, end_day=1, duration=int(r["Duration (days)"]),
            es=int(r["ES"]), ef=int(r["EF"]), ls=int(r["LS"]), lf=int(r["LF"]), slack=int(r["Slack"]),
            is_critical=bool(r["Critical?"]),
            start_date=pd.to_datetime(r["Start"]).date(), end_date=pd.to_datetime(r["Finish"]).date()
        ))

    total_hours, total_cost = estimate_hours_and_cost(
        acts, float(proj["hours_per_day"]), float(proj["base_rate_inr"]),
        float(proj["labour_burden"]), float(proj["inefficiency"])
    )
    total_with_cont = total_cost + compute_contingency(total_cost, float(proj["contingency"]))

    st.markdown(
        f"**Total hours:** {total_hours:.1f} h &nbsp;&nbsp; "
        f"**Total labour cost:** ₹{total_cost:,.2f} &nbsp;&nbsp; "
        f"**Incl. contingency:** ₹{total_with_cont:,.2f}"
    )

    # ---------------- Progress Updates ----------------
    st.markdown("### Progress Updates")
    pct_by_id = latest_percent_by_task_id(list(df_tasks["id"].astype(int)))

    today = st.date_input("Update date", value=dt.date.today())
    mask_today = (pd.to_datetime(df_tasks["Start"]).dt.date <= today) & (pd.to_datetime(df_tasks["Finish"]).dt.date >= today)
    todays = df_tasks.loc[mask_today, ["id","name","Area","Trade"]].copy()
    todays["Percent Complete"] = todays["id"].map(pct_by_id).fillna(0.0)
    todays["Notes"] = ""

    c1, c2 = st.columns([1,1])
    with c1:
        if st.button("Mark all **today's scheduled** tasks 100%"):
            todays["Percent Complete"] = 100.0
    with c2:
        st.caption("Or edit below, then **Save progress**.")

    edited = st.data_editor(todays.rename(columns={"id":"task_id"}), use_container_width=True, key="progress_editor")
    if st.button("Save progress"):
        save_progress_updates(today, edited)
        st.success("Progress saved.")
        st.rerun()

    # ---------------- EV / Schedule Health ----------------
    st.markdown("### Earned Value & Schedule Health")
    pct_by_name = {r["name"]: pct_by_id.get(int(r["id"]), 0.0) for _, r in df_tasks.iterrows()}
    ev = earned_value(
        acts, today, float(proj["hours_per_day"]), float(proj["base_rate_inr"]),
        float(proj["labour_burden"]), float(proj["inefficiency"]), pct_by_name
    )
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("SPI", f"{ev['SPI']:.2f}")
    k2.metric("CPI", f"{ev['CPI']:.2f}")
    k3.metric("SV (₹)", f"{ev['SV']:,.0f}")
    k4.metric("CV (₹)", f"{ev['CV']:,.0f}")

    # ---------------- Inefficiency diagnostics & suggestions ----------------
    st.markdown("### Inefficiency Diagnostics")
    wl = daily_workload_by_trade(acts, proj["start_date"])
    gapdf = gaps_by_area(acts)
    fragdf = fragmentation_by_trade(acts)
    colA, colB, colC = st.columns(3)
    with colA:
        st.caption("Daily workload by trade (tasks/day)")
        st.dataframe(wl.head(50), use_container_width=True, height=260)
    with colB:
        st.caption("Largest idle gaps by area")
        st.dataframe(gapdf.head(20), use_container_width=True, height=260)
    with colC:
        st.caption("Trade fragmentation (segments count)")
        st.dataframe(fragdf.head(20), use_container_width=True, height=260)

    tips = improvement_suggestions(acts, proj["start_date"])
    if tips:
        st.info("**Suggestions to reduce inefficiencies:**\n\n- " + "\n- ".join(tips))

    # ---------------- Gantt (Filtered) ----------------
    st.markdown("### Gantt (Filtered)")
    f_acts = []
    for _, r in df_view.iterrows():
        f_acts.append(Activity(
            name=r["name"], area=r.get("Area",""), trade=r.get("Trade",""), description=r.get("Description",""),
            start_day=1, end_day=1, duration=int(r["Duration (days)"]),
            es=int(r["ES"]), ef=int(r["EF"]), ls=int(r["LS"]), lf=int(r["LF"]), slack=int(r["Slack"]),
            is_critical=bool(r["Critical?"]),
            start_date=pd.to_datetime(r["Start"]).date(), end_date=pd.to_datetime(r["Finish"]).date()
        ))
    fig = make_gantt(f_acts, {k:v for k,v in pct_by_name.items() if k in df_view["name"].tolist()}, height_per_task=26)
    st.plotly_chart(fig, use_container_width=True)

    # ---------------- AI CM Assistant (LLM chatbot) ----------------
    st.markdown("### AI CM Assistant")
    st.caption("Ask about the plan, daily activities, bottlenecks, or ways to reduce inefficiency. The assistant only uses your project context.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # render history
    for role, content in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(content)

    q = st.chat_input("e.g., \"What’s scheduled on 2025-08-15?\" or \"How do I shorten the Dining Room work?\"")
    if q:
        with st.chat_message("user"):
            st.write(q)
        answer, sources = answer_question(pid, q)
        with st.chat_message("assistant"):
            st.write(answer)
            if sources:
                st.markdown(
                    "**Sources:** " + ", ".join(
                        [f"[Task {s['id']}] {s['name']}" if s["type"] == "task"
                         else f"[Gap] {s['area']} ({s['gap_days']}d after '{s['after_task']}')" for s in sources]
                    )
                )
        st.session_state.chat_history.append(("user", q))
        st.session_state.chat_history.append(("assistant", answer))


if __name__ == "__main__":
    main()
