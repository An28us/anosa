"""
================================================================================
 Ananasa (أناناسة) — Market-Driven Product Price Predictor
================================================================================
A lightweight, high-performance, bilingual (EN/AR) Streamlit application that
predicts a product's final market price from four economic indicators:

    1. USD Exchange Rate
    2. Seed Price (raw agricultural input)
    3. Petrol Price (fuel / transportation cost)
    4. Historical Raw Product Price

Architecture
------------
- DatabaseHandler   : encapsulates ALL SQLite operations (market data +
                       prediction history), using indexed, chunked I/O so the
                       app stays fast even as tables grow toward ~1 GB.
- ModelManager       : encapsulates ML lifecycle — trains a baseline
                       RandomForestRegressor on synthetic data if no
                       model.pkl exists, and exposes predict().
- i18n dictionary    : all UI strings for English / Arabic, with RTL support.
- Streamlit pages    : Prediction Hub, Market Data Manager, History & Logs,
                       Settings & Localization.

Run with:  streamlit run app.py
================================================================================
"""

import os
import io
import sqlite3
import pickle
import datetime as dt
from contextlib import contextmanager

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ==============================================================================
# 0. GLOBAL CONFIG & CONSTANTS
# ==============================================================================

APP_NAME = "Ananasa"
APP_NAME_AR = "أناناسة"
DB_PATH = "ananasa_history.db"
MODEL_PATH = "model.pkl"
CHUNK_SIZE = 50_000  # rows per chunk when ingesting large CSV/Excel files

FEATURE_COLUMNS = ["usd_price", "seed_price", "petrol_price", "raw_product_price"]
FEATURE_LABELS_EN = {
    "usd_price": "USD Exchange Rate",
    "seed_price": "Seed Price",
    "petrol_price": "Petrol Price",
    "raw_product_price": "Raw Product Price",
}
FEATURE_LABELS_AR = {
    "usd_price": "سعر صرف الدولار",
    "seed_price": "سعر البذور",
    "petrol_price": "سعر البنزين",
    "raw_product_price": "سعر المنتج الخام",
}

st.set_page_config(
    page_title="Ananasa | أناناسة",
    page_icon="🍍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# 1. INTERNATIONALIZATION (i18n)
# ==============================================================================

TRANSLATIONS = {
    "en": {
        "app_title": "🍍 Ananasa — Smart Price Predictor",
        "app_subtitle": "Forecast product prices from real market indicators",
        "nav_prediction": "🔮 Prediction Hub",
        "nav_data": "📊 Market Data Manager",
        "nav_history": "📜 Prediction History & Logs",
        "nav_settings": "🌐 Settings & Localization",
        "sidebar_language": "Language / اللغة",
        "sidebar_model_status": "Model Status",
        "model_ready": "✅ Model ready",
        "model_missing": "⚠️ Training baseline model…",
        "predict_header": "Enter Current Market Values",
        "usd_price": "USD Exchange Rate (per local currency)",
        "seed_price": "Seed Price (per unit)",
        "petrol_price": "Petrol Price (per liter)",
        "raw_product_price": "Historical Raw Product Price (before processing)",
        "predict_btn": "🔮 Predict Price",
        "predicted_price": "Predicted Final Price",
        "profit_margin": "Estimated Profit Margin",
        "cost_drivers": "Key Cost Drivers",
        "prediction_saved": "Prediction saved to history ✅",
        "currency_unit": "EGP",
        "data_header": "Upload Historical Market Data",
        "data_upload_help": "Upload a CSV or Excel file containing historical columns: usd_price, seed_price, petrol_price, raw_product_price, final_product_price",
        "data_uploaded_success": "File ingested successfully into SQLite ✅",
        "data_rows_loaded": "Rows loaded",
        "data_summary": "Data Summary & Trends",
        "retrain_btn": "🔁 Retrain Model on Uploaded Data",
        "retrain_success": "Model retrained successfully on your data ✅",
        "retrain_fail": "Not enough valid rows to retrain. Need columns: {cols}",
        "history_header": "Prediction History",
        "history_empty": "No predictions logged yet. Go to the Prediction Hub to create one!",
        "export_csv": "⬇️ Export History to CSV",
        "clear_history": "🗑️ Clear All History",
        "clear_history_confirm": "This will permanently delete all logged predictions. Are you sure?",
        "settings_header": "Settings & Localization",
        "settings_language": "Interface Language",
        "settings_danger_zone": "⚠️ Danger Zone",
        "reset_db": "Reset Prediction Database",
        "reset_db_help": "Deletes all stored predictions and market data permanently.",
        "reset_model": "Reset Model to Baseline",
        "reset_model_help": "Discards any retrained model and restores the synthetic baseline model.",
        "confirm_action": "Yes, I understand — Proceed",
        "action_done": "Done ✅",
        "no_data_warning": "No market data uploaded yet. Upload a file in the Market Data Manager page.",
        "table_timestamp": "Timestamp",
        "table_predicted": "Predicted Price",
        "table_actual": "Actual Price",
        "add_actual": "Log Actual Outcome",
        "select_row": "Select prediction row (by ID)",
        "actual_price_input": "Actual market price observed",
        "save_actual": "💾 Save Actual Price",
        "chart_trend_title": "Market Indicator Trends Over Time",
        "chart_corr_title": "Feature Correlation with Final Price",
        "records_count": "Total Records",
        "avg_prediction": "Average Predicted Price",
        "footer_note": "Built with Streamlit • SQLite • scikit-learn • Plotly",
    },
    "ar": {
        "app_title": "🍍 أناناسة — التنبؤ الذكي بالأسعار",
        "app_subtitle": "توقّع أسعار المنتجات بناءً على مؤشرات السوق الحقيقية",
        "nav_prediction": "🔮 مركز التنبؤ",
        "nav_data": "📊 إدارة بيانات السوق",
        "nav_history": "📜 سجل التنبؤات",
        "nav_settings": "🌐 الإعدادات واللغة",
        "sidebar_language": "Language / اللغة",
        "sidebar_model_status": "حالة النموذج",
        "model_ready": "✅ النموذج جاهز",
        "model_missing": "⚠️ جاري تدريب النموذج الأساسي…",
        "predict_header": "أدخل القيم الحالية للسوق",
        "usd_price": "سعر صرف الدولار",
        "seed_price": "سعر البذور (للوحدة)",
        "petrol_price": "سعر البنزين (لليتر)",
        "raw_product_price": "سعر المنتج الخام التاريخي (قبل التصنيع)",
        "predict_btn": "🔮 توقّع السعر",
        "predicted_price": "السعر النهائي المتوقع",
        "profit_margin": "هامش الربح التقديري",
        "cost_drivers": "أهم محركات التكلفة",
        "prediction_saved": "تم حفظ التوقع في السجل ✅",
        "currency_unit": "جنيه",
        "data_header": "رفع بيانات السوق التاريخية",
        "data_upload_help": "ارفع ملف CSV أو Excel يحتوي على الأعمدة: usd_price, seed_price, petrol_price, raw_product_price, final_product_price",
        "data_uploaded_success": "تم استيراد الملف بنجاح إلى قاعدة البيانات ✅",
        "data_rows_loaded": "عدد الصفوف المحمّلة",
        "data_summary": "ملخص البيانات والاتجاهات",
        "retrain_btn": "🔁 إعادة تدريب النموذج على البيانات المرفوعة",
        "retrain_success": "تم إعادة تدريب النموذج بنجاح ✅",
        "retrain_fail": "لا توجد بيانات كافية لإعادة التدريب. الأعمدة المطلوبة: {cols}",
        "history_header": "سجل التنبؤات",
        "history_empty": "لا توجد تنبؤات مسجلة بعد. اذهب إلى مركز التنبؤ لإنشاء واحد!",
        "export_csv": "⬇️ تصدير السجل إلى CSV",
        "clear_history": "🗑️ مسح كل السجل",
        "clear_history_confirm": "سيؤدي هذا إلى حذف جميع التنبؤات المسجلة نهائيًا. هل أنت متأكد؟",
        "settings_header": "الإعدادات واللغة",
        "settings_language": "لغة الواجهة",
        "settings_danger_zone": "⚠️ منطقة الخطر",
        "reset_db": "إعادة تعيين قاعدة بيانات التنبؤات",
        "reset_db_help": "يحذف جميع التنبؤات وبيانات السوق المخزنة نهائيًا.",
        "reset_model": "إعادة تعيين النموذج إلى الحالة الأساسية",
        "reset_model_help": "يتجاهل أي نموذج معاد تدريبه ويستعيد النموذج الأساسي.",
        "confirm_action": "نعم، أوافق — متابعة",
        "action_done": "تم ✅",
        "no_data_warning": "لم يتم رفع بيانات سوق بعد. ارفع ملفًا من صفحة إدارة بيانات السوق.",
        "table_timestamp": "التوقيت",
        "table_predicted": "السعر المتوقع",
        "table_actual": "السعر الفعلي",
        "add_actual": "تسجيل النتيجة الفعلية",
        "select_row": "اختر صف التنبؤ (بالرقم)",
        "actual_price_input": "السعر الفعلي الملاحظ في السوق",
        "save_actual": "💾 حفظ السعر الفعلي",
        "chart_trend_title": "اتجاهات مؤشرات السوق عبر الزمن",
        "chart_corr_title": "علاقة كل عامل بالسعر النهائي",
        "records_count": "إجمالي السجلات",
        "avg_prediction": "متوسط السعر المتوقع",
        "footer_note": "بُني باستخدام Streamlit • SQLite • scikit-learn • Plotly",
    },
}


def t(key: str) -> str:
    """Fetch a translated string for the currently selected language."""
    lang = st.session_state.get("lang", "en")
    return TRANSLATIONS[lang].get(key, key)


def feature_label(col: str) -> str:
    lang = st.session_state.get("lang", "en")
    return (FEATURE_LABELS_AR if lang == "ar" else FEATURE_LABELS_EN)[col]


# ==============================================================================
# 2. DATABASE HANDLER — encapsulates all SQLite operations
# ==============================================================================

class DatabaseHandler:
    """
    Modular helper class for all SQLite persistence in Ananasa.

    Two logical tables:
      - market_data : historical rows uploaded by the user (scalable to ~1GB
                      via chunked inserts + indexed lookups).
      - predictions : every prediction run, with inputs, output, and an
                      optional later-logged actual outcome.

    All connections are opened per-operation via a context manager to keep
    the app thread-safe under Streamlit's execution model, and WAL mode is
    enabled for better concurrent read/write performance at scale.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_schema()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")   # better concurrent perf
            conn.execute("PRAGMA synchronous=NORMAL;")  # faster, still safe
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    record_date TEXT,
                    usd_price REAL,
                    seed_price REAL,
                    petrol_price REAL,
                    raw_product_price REAL,
                    final_product_price REAL
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    usd_price REAL,
                    seed_price REAL,
                    petrol_price REAL,
                    raw_product_price REAL,
                    predicted_price REAL,
                    actual_price REAL
                );
            """)
            # Indexes to keep queries fast as tables scale toward ~1GB.
            conn.execute("CREATE INDEX IF NOT EXISTS idx_market_date ON market_data(record_date);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_pred_timestamp ON predictions(timestamp);")

    # ---------------------------------------------------------------- market_data

    def bulk_insert_market_data(self, df: pd.DataFrame) -> int:
        """
        Insert a (potentially large) DataFrame into market_data in chunks,
        so multi-gigabyte uploads never blow up memory or a single
        transaction.
        """
        required = ["usd_price", "seed_price", "petrol_price",
                    "raw_product_price", "final_product_price"]
        df = df.copy()
        if "record_date" not in df.columns:
            df["record_date"] = pd.Timestamp.today().strftime("%Y-%m-%d")
        df = df[["record_date"] + required]
        total = 0
        with self._connect() as conn:
            for start in range(0, len(df), CHUNK_SIZE):
                chunk = df.iloc[start:start + CHUNK_SIZE]
                chunk.to_sql("market_data", conn, if_exists="append", index=False)
                total += len(chunk)
        return total

    def get_market_data(self, limit: int = 5000) -> pd.DataFrame:
        """Indexed, limited read — avoids loading an entire huge table into memory."""
        with self._connect() as conn:
            return pd.read_sql_query(
                "SELECT * FROM market_data ORDER BY record_date DESC LIMIT ?;",
                conn, params=(limit,),
            )

    def market_data_row_count(self) -> int:
        with self._connect() as conn:
            cur = conn.execute("SELECT COUNT(*) FROM market_data;")
            return cur.fetchone()[0]

    def get_market_data_for_training(self) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query("SELECT * FROM market_data;", conn)

    def clear_market_data(self):
        with self._connect() as conn:
            conn.execute("DELETE FROM market_data;")

    # ---------------------------------------------------------------- predictions

    def log_prediction(self, usd, seed, petrol, raw, predicted) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO predictions
                   (timestamp, usd_price, seed_price, petrol_price,
                    raw_product_price, predicted_price, actual_price)
                   VALUES (?, ?, ?, ?, ?, ?, NULL);""",
                (dt.datetime.now().isoformat(timespec="seconds"),
                 usd, seed, petrol, raw, predicted),
            )
            return cur.lastrowid

    def update_actual_price(self, pred_id: int, actual_price: float):
        with self._connect() as conn:
            conn.execute(
                "UPDATE predictions SET actual_price = ? WHERE id = ?;",
                (actual_price, pred_id),
            )

    def get_predictions(self) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query(
                "SELECT * FROM predictions ORDER BY timestamp DESC;", conn
            )

    def clear_predictions(self):
        with self._connect() as conn:
            conn.execute("DELETE FROM predictions;")


# ==============================================================================
# 3. MODEL MANAGER — encapsulates ML lifecycle
# ==============================================================================

class ModelManager:
    """
    Owns the lifecycle of the price-prediction model:
      - loads model.pkl if present
      - otherwise trains a baseline RandomForestRegressor on synthetic data
        so the app is fully usable out of the box
      - exposes retrain() to fit on real uploaded market data
      - exposes predict() and feature_importance()
    """

    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.metrics = {}
        self.load_or_bootstrap()

    # ---------------------------------------------------------------- synthetic data

    @staticmethod
    def _generate_synthetic_data(n: int = 2000, seed: int = 42) -> pd.DataFrame:
        """
        Creates a plausible synthetic dataset so the app trains a sensible
        baseline model on first boot, with no external data required.
        Final price is modeled as a noisy function of the four inputs,
        reflecting realistic economic relationships (higher USD / seed /
        petrol costs and higher raw price all push the final price up).
        """
        rng = np.random.default_rng(seed)
        usd = rng.normal(48, 4, n).clip(30, 65)
        seed_price = rng.normal(25, 6, n).clip(5, 60)
        petrol = rng.normal(12, 2.5, n).clip(5, 25)
        raw_product = rng.normal(40, 10, n).clip(10, 100)

        noise = rng.normal(0, 3, n)
        final_price = (
            raw_product * 1.35
            + usd * 0.9
            + seed_price * 0.4
            + petrol * 1.1
            + 5
            + noise
        )
        return pd.DataFrame({
            "usd_price": usd,
            "seed_price": seed_price,
            "petrol_price": petrol,
            "raw_product_price": raw_product,
            "final_product_price": final_price,
        })

    # ---------------------------------------------------------------- lifecycle

    def load_or_bootstrap(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                payload = pickle.load(f)
            self.model = payload["model"]
            self.metrics = payload.get("metrics", {})
        else:
            df = self._generate_synthetic_data()
            self._fit_and_save(df, source="synthetic baseline")

    def _fit_and_save(self, df: pd.DataFrame, source: str):
        X = df[FEATURE_COLUMNS]
        y = df["final_product_price"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = RandomForestRegressor(
            n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        self.metrics = {
            "mae": float(mean_absolute_error(y_test, preds)),
            "r2": float(r2_score(y_test, preds)),
            "n_rows": len(df),
            "source": source,
            "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
        }
        self.model = model
        with open(self.model_path, "wb") as f:
            pickle.dump({"model": model, "metrics": self.metrics}, f)

    def retrain(self, df: pd.DataFrame) -> bool:
        """Retrain on real uploaded data. Returns False if data is unusable."""
        needed = FEATURE_COLUMNS + ["final_product_price"]
        df = df.dropna(subset=needed)
        if len(df) < 20:
            return False
        self._fit_and_save(df[needed], source="user-uploaded data")
        return True

    def reset_to_baseline(self):
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
        df = self._generate_synthetic_data()
        self._fit_and_save(df, source="synthetic baseline")

    def predict(self, usd, seed, petrol, raw) -> float:
        X = pd.DataFrame([{
            "usd_price": usd,
            "seed_price": seed,
            "petrol_price": petrol,
            "raw_product_price": raw,
        }])
        return float(self.model.predict(X)[0])

    def feature_importance(self) -> dict:
        if hasattr(self.model, "feature_importances_"):
            return dict(zip(FEATURE_COLUMNS, self.model.feature_importances_))
        return {c: 0.25 for c in FEATURE_COLUMNS}


# ==============================================================================
# 4. CACHED RESOURCE / DATA LOADERS
# ==============================================================================

@st.cache_resource(show_spinner=False)
def get_db_handler() -> DatabaseHandler:
    """Cached singleton DB handler — connection setup runs once per session pool."""
    return DatabaseHandler()


@st.cache_resource(show_spinner=False)
def get_model_manager(_version: int = 0) -> ModelManager:
    """
    Cached singleton ModelManager. `_version` is a cache-busting key we bump
    (via session_state) whenever the model is retrained/reset, forcing a
    fresh load from disk instead of serving a stale cached model object.
    """
    return ModelManager()


@st.cache_data(show_spinner=False, ttl=300)
def load_market_data_cached(_db_version: int, limit: int = 5000) -> pd.DataFrame:
    """Cached market-data read; keyed on a version counter bumped on new uploads."""
    db = get_db_handler()
    return db.get_market_data(limit=limit)


@st.cache_data(show_spinner=False)
def read_uploaded_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Parses an uploaded CSV/Excel file into a DataFrame (cached by content)."""
    buffer = io.BytesIO(file_bytes)
    if filename.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(buffer)
    return pd.read_csv(buffer)


# ==============================================================================
# 5. CUSTOM CSS — modern, high-contrast, touch-friendly, RTL-aware
# ==============================================================================

def inject_css():
    is_rtl = st.session_state.get("lang", "en") == "ar"
    direction = "rtl" if is_rtl else "ltr"
    text_align = "right" if is_rtl else "left"

    st.markdown(f"""
    <style>
        html, body, [class*="css"] {{
            direction: {direction};
        }}
        .stApp {{
            background: linear-gradient(180deg, #0f1720 0%, #16202b 100%);
        }}
        /* ---- Headers ---- */
        h1, h2, h3 {{
            text-align: {text_align};
            font-weight: 800 !important;
            letter-spacing: -0.5px;
        }}
        /* ---- Hero banner ---- */
        .ananasa-hero {{
            background: linear-gradient(120deg, #FFC93C 0%, #FF8C42 55%, #FF5D5D 100%);
            padding: 1.4rem 1.8rem;
            border-radius: 18px;
            margin-bottom: 1.4rem;
            box-shadow: 0 8px 24px rgba(255, 140, 66, 0.25);
            text-align: {text_align};
        }}
        .ananasa-hero h1 {{
            color: #16202b !important;
            margin: 0;
            font-size: 2rem;
        }}
        .ananasa-hero p {{
            color: #16202b;
            opacity: 0.85;
            margin: 0.2rem 0 0 0;
            font-size: 1rem;
        }}
        /* ---- Metric / KPI cards ---- */
        .metric-card {{
            background: #1c2733;
            border: 1px solid #2c3a4a;
            border-radius: 16px;
            padding: 1.2rem 1.4rem;
            text-align: {text_align};
            box-shadow: 0 4px 14px rgba(0,0,0,0.25);
            transition: transform 0.15s ease;
        }}
        .metric-card:hover {{ transform: translateY(-2px); }}
        .metric-card .label {{
            color: #9fb0c1;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }}
        .metric-card .value {{
            color: #FFC93C;
            font-size: 1.9rem;
            font-weight: 800;
            margin-top: 0.15rem;
        }}
        .metric-card.positive .value {{ color: #4ADE80; }}
        .metric-card.negative .value {{ color: #FB7185; }}

        /* ---- Buttons: touch-friendly, high contrast ---- */
        .stButton > button {{
            background: linear-gradient(120deg, #FF8C42, #FF5D5D);
            color: white;
            border: none;
            border-radius: 14px;
            padding: 0.85rem 1.4rem;
            font-weight: 700;
            font-size: 1rem;
            width: 100%;
            min-height: 48px; /* touch-friendly */
            box-shadow: 0 4px 14px rgba(255, 93, 93, 0.3);
            transition: transform 0.1s ease, box-shadow 0.1s ease;
        }}
        .stButton > button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 6px 18px rgba(255, 93, 93, 0.45);
        }}
        .stButton > button:active {{ transform: translateY(0px) scale(0.98); }}

        /* ---- Inputs ---- */
        .stSlider, .stNumberInput, .stSelectbox {{
            text-align: {text_align};
        }}

        /* ---- Sidebar ---- */
        section[data-testid="stSidebar"] {{
            background: #121a24;
        }}

        /* ---- Dataframe / table container ---- */
        .stDataFrame {{ border-radius: 12px; overflow: hidden; }}

        /* ---- Footer ---- */
        .ananasa-footer {{
            text-align: center;
            color: #5c6b7a;
            font-size: 0.8rem;
            margin-top: 2.5rem;
            padding-top: 1rem;
            border-top: 1px solid #2c3a4a;
        }}

        /* ---- Responsive tweaks ---- */
        @media (max-width: 640px) {{
            .ananasa-hero h1 {{ font-size: 1.5rem; }}
            .metric-card .value {{ font-size: 1.5rem; }}
        }}
    </style>
    """, unsafe_allow_html=True)


def metric_card(label: str, value: str, kind: str = "") -> str:
    """Returns HTML for a styled KPI card (kind: '', 'positive', 'negative')."""
    return f"""
    <div class="metric-card {kind}">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
    </div>
    """


# ==============================================================================
# 6. SESSION STATE INITIALIZATION
# ==============================================================================

def init_session_state():
    if "lang" not in st.session_state:
        st.session_state.lang = "en"
    if "db_version" not in st.session_state:
        st.session_state.db_version = 0
    if "model_version" not in st.session_state:
        st.session_state.model_version = 0
    if "confirm_clear_history" not in st.session_state:
        st.session_state.confirm_clear_history = False
    if "confirm_reset_db" not in st.session_state:
        st.session_state.confirm_reset_db = False
    if "confirm_reset_model" not in st.session_state:
        st.session_state.confirm_reset_model = False


# ==============================================================================
# 7. PAGE: PREDICTION HUB
# ==============================================================================

def page_prediction_hub(db: DatabaseHandler, model_mgr: ModelManager):
    st.markdown(f"### {t('predict_header')}")

    col_inputs, col_result = st.columns([1.1, 1], gap="large")

    with col_inputs:
        with st.container(border=True):
            usd = st.slider(t("usd_price"), min_value=20.0, max_value=90.0,
                             value=48.0, step=0.1, help=t("usd_price"))
            seed = st.slider(t("seed_price"), min_value=1.0, max_value=100.0,
                              value=25.0, step=0.5)
            petrol = st.slider(t("petrol_price"), min_value=1.0, max_value=40.0,
                                value=12.0, step=0.1)
            raw = st.slider(t("raw_product_price"), min_value=1.0, max_value=150.0,
                             value=40.0, step=0.5)

            predict_clicked = st.button(t("predict_btn"), use_container_width=True)

    with col_result:
        if predict_clicked:
            predicted = model_mgr.predict(usd, seed, petrol, raw)
            margin = predicted - raw
            margin_pct = (margin / raw * 100) if raw > 0 else 0.0
            db.log_prediction(usd, seed, petrol, raw, predicted)
            st.session_state.db_version += 1
            st.toast(t("prediction_saved"), icon="✅")

            currency = t("currency_unit")
            st.markdown(
                metric_card(t("predicted_price"), f"{predicted:,.2f} {currency}"),
                unsafe_allow_html=True,
            )
            st.write("")
            kind = "positive" if margin >= 0 else "negative"
            st.markdown(
                metric_card(t("profit_margin"),
                            f"{margin:+,.2f} {currency} ({margin_pct:+.1f}%)",
                            kind=kind),
                unsafe_allow_html=True,
            )
            st.write("")

            # Key cost drivers chart
            importances = model_mgr.feature_importance()
            imp_df = pd.DataFrame({
                "feature": [feature_label(c) for c in FEATURE_COLUMNS],
                "importance": [importances.get(c, 0) for c in FEATURE_COLUMNS],
            }).sort_values("importance", ascending=True)

            fig = px.bar(
                imp_df, x="importance", y="feature", orientation="h",
                title=t("cost_drivers"),
                color="importance", color_continuous_scale=["#FFC93C", "#FF5D5D"],
            )
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=280,
                margin=dict(l=10, r=10, t=40, b=10),
                showlegend=False,
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👈 " + t("predict_header"))


# ==============================================================================
# 8. PAGE: MARKET DATA MANAGER
# ==============================================================================

def page_market_data_manager(db: DatabaseHandler, model_mgr: ModelManager):
    st.markdown(f"### {t('data_header')}")
    st.caption(t("data_upload_help"))

    uploaded = st.file_uploader(
        t("data_header"), type=["csv", "xlsx", "xls"], label_visibility="collapsed"
    )

    if uploaded is not None:
        try:
            df = read_uploaded_file(uploaded.getvalue(), uploaded.name)
            required_cols = ["usd_price", "seed_price", "petrol_price",
                              "raw_product_price", "final_product_price"]
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                st.error(
                    ("Missing required columns: " if st.session_state.lang == "en"
                     else "أعمدة مفقودة: ") + ", ".join(missing)
                )
            else:
                inserted = db.bulk_insert_market_data(df)
                st.session_state.db_version += 1
                st.success(f"{t('data_uploaded_success')} — {t('data_rows_loaded')}: {inserted:,}")

                if st.button(t("retrain_btn")):
                    ok = model_mgr.retrain(df)
                    if ok:
                        st.session_state.model_version += 1
                        st.success(t("retrain_success"))
                        st.cache_resource.clear()
                        st.rerun()
                    else:
                        st.error(t("retrain_fail").format(cols=", ".join(required_cols)))
        except Exception as e:
            st.error(f"⚠️ {e}")

    st.divider()

    total_rows = db.market_data_row_count()
    if total_rows == 0:
        st.warning(t("no_data_warning"))
        return

    st.markdown(f"### {t('data_summary')}")
    st.caption(f"{t('records_count')}: {total_rows:,}")

    data = load_market_data_cached(st.session_state.db_version, limit=5000)
    if data.empty:
        st.warning(t("no_data_warning"))
        return

    data_sorted = data.sort_values("record_date")

    tab1, tab2 = st.tabs([t("chart_trend_title"), t("chart_corr_title")])

    with tab1:
        fig = go.Figure()
        colors = {"usd_price": "#4ADE80", "seed_price": "#FFC93C",
                  "petrol_price": "#FB7185", "raw_product_price": "#60A5FA"}
        for col in FEATURE_COLUMNS:
            fig.add_trace(go.Scatter(
                x=data_sorted["record_date"], y=data_sorted[col],
                mode="lines+markers", name=feature_label(col),
                line=dict(color=colors[col], width=2),
            ))
        fig.update_layout(
            template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)", height=420,
            legend=dict(orientation="h", y=-0.2),
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        corr_cols = FEATURE_COLUMNS + ["final_product_price"]
        corr = data_sorted[corr_cols].corr()
        fig2 = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdYlGn",
            aspect="auto",
        )
        fig2.update_layout(
            template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)", height=420,
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

    with st.expander("🔍 " + ("Raw data preview" if st.session_state.lang == "en" else "معاينة البيانات الخام")):
        st.dataframe(data.head(200), use_container_width=True, height=300)


# ==============================================================================
# 9. PAGE: PREDICTION HISTORY & LOGS
# ==============================================================================

def page_history(db: DatabaseHandler):
    st.markdown(f"### {t('history_header')}")

    history = db.get_predictions()
    if history.empty:
        st.info(t("history_empty"))
        return

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(metric_card(t("records_count"), f"{len(history):,}"),
                    unsafe_allow_html=True)
    with col2:
        avg_pred = history["predicted_price"].mean()
        st.markdown(metric_card(t("avg_prediction"),
                                 f"{avg_pred:,.2f} {t('currency_unit')}"),
                    unsafe_allow_html=True)

    st.write("")

    display_df = history.rename(columns={
        "timestamp": t("table_timestamp"),
        "usd_price": feature_label("usd_price"),
        "seed_price": feature_label("seed_price"),
        "petrol_price": feature_label("petrol_price"),
        "raw_product_price": feature_label("raw_product_price"),
        "predicted_price": t("table_predicted"),
        "actual_price": t("table_actual"),
    })
    st.dataframe(display_df, use_container_width=True, height=350)

    col_a, col_b = st.columns(2)
    with col_a:
        csv_bytes = history.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            t("export_csv"), data=csv_bytes,
            file_name=f"ananasa_history_{dt.date.today()}.csv",
            mime="text/csv", use_container_width=True,
        )
    with col_b:
        if st.button(t("clear_history"), use_container_width=True):
            st.session_state.confirm_clear_history = True

    if st.session_state.confirm_clear_history:
        st.warning(t("clear_history_confirm"))
        c1, c2 = st.columns(2)
        if c1.button(t("confirm_action"), key="confirm_clear_hist_yes"):
            db.clear_predictions()
            st.session_state.confirm_clear_history = False
            st.success(t("action_done"))
            st.rerun()
        if c2.button("Cancel" if st.session_state.lang == "en" else "إلغاء",
                      key="confirm_clear_hist_no"):
            st.session_state.confirm_clear_history = False
            st.rerun()

    st.divider()

    # Log actual market outcome against a past prediction
    st.markdown(f"#### {t('add_actual')}")
    ids = history["id"].tolist()
    if ids:
        selected_id = st.selectbox(t("select_row"), ids)
        actual_val = st.number_input(t("actual_price_input"), min_value=0.0, step=0.5)
        if st.button(t("save_actual")):
            db.update_actual_price(int(selected_id), actual_val)
            st.session_state.db_version += 1
            st.success(t("action_done"))
            st.rerun()


# ==============================================================================
# 10. PAGE: SETTINGS & LOCALIZATION
# ==============================================================================

def page_settings(db: DatabaseHandler, model_mgr: ModelManager):
    st.markdown(f"### {t('settings_header')}")

    lang_choice = st.radio(
        t("settings_language"), options=["English", "العربية"],
        index=0 if st.session_state.lang == "en" else 1,
        horizontal=True,
    )
    new_lang = "en" if lang_choice == "English" else "ar"
    if new_lang != st.session_state.lang:
        st.session_state.lang = new_lang
        st.rerun()

    st.divider()
    st.markdown(f"#### {t('settings_danger_zone')}")

    col1, col2 = st.columns(2)

    with col1:
        st.caption(t("reset_db_help"))
        if st.button(t("reset_db"), use_container_width=True):
            st.session_state.confirm_reset_db = True
        if st.session_state.confirm_reset_db:
            st.warning(t("clear_history_confirm"))
            if st.button(t("confirm_action"), key="confirm_reset_db_yes"):
                db.clear_predictions()
                db.clear_market_data()
                st.session_state.db_version += 1
                st.session_state.confirm_reset_db = False
                st.success(t("action_done"))
                st.rerun()

    with col2:
        st.caption(t("reset_model_help"))
        if st.button(t("reset_model"), use_container_width=True):
            st.session_state.confirm_reset_model = True
        if st.session_state.confirm_reset_model:
            st.warning(t("clear_history_confirm"))
            if st.button(t("confirm_action"), key="confirm_reset_model_yes"):
                model_mgr.reset_to_baseline()
                st.session_state.model_version += 1
                st.session_state.confirm_reset_model = False
                st.cache_resource.clear()
                st.success(t("action_done"))
                st.rerun()

    st.divider()
    st.markdown("#### " + ("Model Info" if st.session_state.lang == "en" else "معلومات النموذج"))
    m = model_mgr.metrics
    if m:
        info_cols = st.columns(4)
        info_cols[0].metric("MAE", f"{m.get('mae', 0):.2f}")
        info_cols[1].metric("R²", f"{m.get('r2', 0):.3f}")
        info_cols[2].metric("Rows", f"{m.get('n_rows', 0):,}")
        info_cols[3].metric("Source", m.get("source", "—"))


# ==============================================================================
# 11. MAIN APP ENTRY POINT
# ==============================================================================

def main():
    init_session_state()
    inject_css()

    db = get_db_handler()
    model_mgr = get_model_manager(st.session_state.model_version)

    # ---- Sidebar: language toggle + navigation + model status ----
    with st.sidebar:
        st.markdown("## 🍍 " + (APP_NAME if st.session_state.lang == "en" else APP_NAME_AR))
        lang_toggle = st.radio(
            t("sidebar_language"), ["English", "العربية"],
            index=0 if st.session_state.lang == "en" else 1,
            horizontal=True, key="lang_toggle_sidebar",
        )
        picked_lang = "en" if lang_toggle == "English" else "ar"
        if picked_lang != st.session_state.lang:
            st.session_state.lang = picked_lang
            st.rerun()

        st.divider()
        page = st.radio(
            "Navigation",
            [t("nav_prediction"), t("nav_data"), t("nav_history"), t("nav_settings")],
            label_visibility="collapsed",
        )

        st.divider()
        st.caption(t("sidebar_model_status"))
        st.success(t("model_ready"))

    # ---- Hero header ----
    st.markdown(f"""
        <div class="ananasa-hero">
            <h1>{t('app_title')}</h1>
            <p>{t('app_subtitle')}</p>
        </div>
    """, unsafe_allow_html=True)

    # ---- Route to selected page ----
    if page == t("nav_prediction"):
        page_prediction_hub(db, model_mgr)
    elif page == t("nav_data"):
        page_market_data_manager(db, model_mgr)
    elif page == t("nav_history"):
        page_history(db)
    elif page == t("nav_settings"):
        page_settings(db, model_mgr)

    st.markdown(f'<div class="ananasa-footer">{t("footer_note")}</div>',
                unsafe_allow_html=True)


if __name__ == "__main__":
    main()
