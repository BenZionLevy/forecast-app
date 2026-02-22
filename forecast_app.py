import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from tvDatafeed import TvDatafeed, Interval
import timesfm

st.set_page_config(
    page_title="מעבדת חיזוי מניות ת״א-35",
    layout="wide",
    page_icon="📈"
)

# ==========================================
# עיצוב
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rubik:wght@300;400;500;600;700&display=swap');
    .stApp { background: #f8fafc; }
    html, body, [class*="css"] { font-family: 'Rubik', sans-serif; direction: rtl; }

    .main-header {
        text-align: center;
        padding: 2rem 0 0.5rem 0;
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1d4ed8 0%, #9333ea 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .warning-box {
        background-color: #fff7ed;
        border: 1px solid #fdba74;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        color: #7c2d12;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>📈 מעבדת חיזוי מניות – מדד ת״א-35</h1>", unsafe_allow_html=True)

# ==========================================
# אזהרה למעלה (כפי שביקשת)
# ==========================================
st.markdown("""
<div class='warning-box'>
⚠️ <strong>אזהרה חשובה:</strong> החיזוי מבוסס על מודל בינה מלאכותית ואינו מתחשב בחדשות, דוחות כספיים או אירועים מאקרו-כלכליים.
המערכת לצורכי מחקר בלבד ואינה מהווה ייעוץ השקעות.
</div>
""", unsafe_allow_html=True)

# ==========================================
# טעינת מודל
# ==========================================
@st.cache_resource
def load_ai_model():
    return timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="cpu",
            per_core_batch_size=1,
            horizon_len=128,
            context_len=512,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
        ),
    )

# ==========================================
# מניות מדד ת״א-35 בלבד
# ==========================================
TA35_STOCKS = {
    "לאומי": ("LUMI", "TASE"),
    "פועלים": ("POLI", "TASE"),
    "דיסקונט": ("DSCT", "TASE"),
    "מזרחי טפחות": ("MZTF", "TASE"),
    "אלביט מערכות": ("ESLT", "TASE"),
    "טבע": ("TEVA", "TASE"),
    "נייס": ("NICE", "TASE"),
    "פריגו": ("PRGO", "TASE"),
    "בזק": ("BEZQ", "TASE"),
    "שופרסל": ("SAE", "TASE"),
    "סלקום": ("CEL", "TASE"),
    "דלק קבוצה": ("DLEKG", "TASE"),
    "אמות": ("AMOT", "TASE"),
    "מליסרון": ("MLSR", "TASE"),
    "קבוצת עזריאלי": ("AZRG", "TASE"),
}

# ==========================================
# משיכת נתונים
# ==========================================
@st.cache_data(ttl=600)
def fetch_data_tv(sym_tuple):
    tv = TvDatafeed()

    df = tv.get_hist(
        symbol=sym_tuple[0],
        exchange=sym_tuple[1],
        interval=Interval.in_daily,
        n_bars=1500
    )

    if df is None or df.empty:
        return pd.DataFrame()

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert("Asia/Jerusalem")
    else:
        df.index = df.index.tz_convert("Asia/Jerusalem")

    df.index = df.index.tz_localize(None)

    return df[['close']]

# ==========================================
# בחירת מניה
# ==========================================
st.subheader("שלב 1: בחר מניה ממדד ת״א-35")

asset_name = st.selectbox(
    "בחר מניה:",
    list(TA35_STOCKS.keys())
)

target_tuple = TA35_STOCKS[asset_name]

backtest = st.selectbox(
    "בדיקת אמינות לאחור:",
    {
        "ללא בדיקה (חיזוי עתידי)": 0,
        "לפני שבוע (5 ימי מסחר)": 5,
        "לפני חודש (21 ימי מסחר)": 21,
        "לפני 3 חודשים": 63
    }
)

cutoff = {
    "ללא בדיקה (חיזוי עתידי)": 0,
    "לפני שבוע (5 ימי מסחר)": 5,
    "לפני חודש (21 ימי מסחר)": 21,
    "לפני 3 חודשים": 63
}[backtest]

# ==========================================
# חיזוי
# ==========================================
if st.button("🔮 הפעל חיזוי", type="primary", use_container_width=True):

    with st.spinner("טוען מודל בינה מלאכותית..."):
        model = load_ai_model()

    with st.spinner("מושך נתונים היסטוריים..."):
        df_hist = fetch_data_tv(target_tuple)

    if df_hist.empty or (len(df_hist) - cutoff) < 512:
        st.error("אין מספיק נתונים לצורך חיזוי (נדרשים לפחות 512 ימי מסחר).")
        st.stop()

    if cutoff > 0:
        df_train = df_hist.iloc[:-cutoff]
        df_actual = df_hist.iloc[-cutoff:]
    else:
        df_train = df_hist
        df_actual = pd.DataFrame()

    prices = df_train['close'].values

    with st.spinner("המודל מחשב תחזית..."):
        forecast_results, quantiles_results = model.forecast([prices], freq=[0])

    forecast = forecast_results[0]
    lower = quantiles_results[0, :, 0]
    upper = quantiles_results[0, :, -1]

    last_date = df_train.index[-1]

    future_dates = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1),
        periods=128
    )

    forecast_df = pd.DataFrame({
        "תחזית": forecast,
        "גבול תחתון": lower,
        "גבול עליון": upper
    }, index=future_dates)

    # ==========================================
    # גרף
    # ==========================================
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_train.index[-200:],
        y=df_train['close'].tail(200),
        mode="lines",
        name="היסטוריה",
        line=dict(color="#2563eb", width=2)
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df.index,
        y=forecast_df["גבול עליון"],
        mode="lines",
        line=dict(width=0),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df.index,
        y=forecast_df["גבול תחתון"],
        mode="lines",
        fill="tonexty",
        fillcolor="rgba(245,158,11,0.2)",
        line=dict(width=0),
        name="טווח סביר"
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df.index,
        y=forecast_df["תחזית"],
        mode="lines",
        name="תחזית AI",
        line=dict(color="#f59e0b", width=3, dash="dash")
    ))

    if not df_actual.empty:
        fig.add_trace(go.Scatter(
            x=df_actual.index,
            y=df_actual["close"],
            mode="lines",
            name="מה קרה בפועל",
            line=dict(color="#10b981", width=3)
        ))

        fig.add_vline(
            x=last_date,
            line_dash="dot",
            line_color="#64748b"
        )

        fig.add_annotation(
            x=last_date,
            y=1,
            yref="paper",
            text="נקודת החיתוך",
            showarrow=False
        )

    fig.update_layout(
        title=f"חיזוי מסלול מחיר – {asset_name}",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h")
    )

    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("© מערכת חיזוי מבוססת Google TimesFM | לשימוש מחקרי בלבד")
