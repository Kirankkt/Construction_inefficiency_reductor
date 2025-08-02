# AI Construction Manager — Kerala (CPM + Progress + EV)

## Quick start
```bash
pip install -r requirements.txt
# (Optional) PostgreSQL URL; SQLite used if not set
export DATABASE_URL="postgresql+psycopg2://USER:PASS@HOST:5432/DB"
streamlit run app.py
```

## What it does
- Parses your colour-coded Excel: **column = day**, **row label = area**, **cell text = scheduled work**.
- Splits contiguous islands into activities, extracts **trade** from text prefix, and description.
- Saves a **baseline** in DB (projects, tasks). No manual SQL needed.
- Computes **CPM** (ES/EF/LS/LF, slack, critical).
- Lets contractor update **% complete** by date or individually.
- Shows a clean **Plotly Gantt** (planned vs progress, critical outlined).
- Computes **EV metrics** (PV/EV/AC, SPI/CPI) and **inefficiency diagnostics** (gaps, trade load, fragmentation).
