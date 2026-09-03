# 🍍 Ananasa (أناناسة)

A lightweight, bilingual (English / Arabic) Streamlit app that predicts a
product's final market price from four indicators: USD exchange rate, seed
price, petrol price, and historical raw product price.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

On first run, Ananasa automatically trains a baseline `RandomForestRegressor`
on synthetic data (no `model.pkl` needed) and creates `ananasa_history.db`
(SQLite) — so the app works immediately with zero setup.

## What's inside

- **🔮 Prediction Hub** — sliders for USD / seed / petrol / raw product price,
  a "Predict Price" button, and a result view showing predicted price, profit
  margin, and a chart of the model's key cost drivers.
- **📊 Market Data Manager** — upload CSV/XLSX historical data (columns:
  `usd_price, seed_price, petrol_price, raw_product_price,
  final_product_price`), ingested into SQLite in chunks, with trend and
  correlation charts, and a one-click "Retrain Model on Uploaded Data" action.
- **📜 Prediction History & Logs** — every prediction is saved to SQLite;
  browse, export to CSV, log actual observed outcomes, or clear history.
- **🌐 Settings & Localization** — toggle English / العربية (with RTL layout),
  reset the database, or reset the model to its synthetic baseline.

## Architecture notes

- `DatabaseHandler` — the only class that touches SQLite. Uses WAL mode,
  indexes on `record_date` / `timestamp`, and chunked `to_sql` inserts
  (`CHUNK_SIZE = 50_000` rows) so large uploads scale toward ~1 GB without
  blowing up memory.
- `ModelManager` — owns the model lifecycle: bootstrap-on-first-run,
  `retrain()` on real data, `reset_to_baseline()`, and `predict()`.
- Streamlit caching: `@st.cache_resource` for the DB handler and model
  (singleton, version-keyed so retraining invalidates the cache), and
  `@st.cache_data` for data reads/parses (TTL + content-hash keyed).
- All UI copy lives in one `TRANSLATIONS` dict — add a language by adding a
  new key.

## File outputs

- `ananasa_history.db` — SQLite database (created automatically).
- `model.pkl` — trained model + metrics (created automatically).
