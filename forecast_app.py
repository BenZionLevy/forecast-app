import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time

from tvDatafeed import TvDatafeed, Interval
import timesfm

st.set_page_config(page_title="מעבדת חיזוי AI", layout="wide", page_icon="🤖")

# ==========================================
# CSS
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rubik:wght@300;400;500;600;700&display=swap');
    .stApp { background: #f8fafc; }
    html, body, [class*="css"] { font-family: 'Rubik', sans-serif; direction: rtl; }
    .main-header {
        text-align: center; padding: 2rem 0 0.5rem 0; font-size: 2.8rem; font-weight: 800;
        background: linear-gradient(135deg, #4f46e5 0%, #ec4899 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    }
    .sub-header { text-align: center; color: #64748b; margin-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>🤖 מעבדת חיזוי מניות: TimesFM</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Google TimesFM 1.0 - Backtesting וחיזוי עתידי</p>", unsafe_allow_html=True)

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
# משיכת נתונים
# ==========================================
@st.cache_data(ttl=600)
def fetch_data_tv(sym_tuple, interval_str):
    tv = TvDatafeed()
    tv_intervals = {
        "1d": Interval.in_daily,
        "60m": Interval.in_1_hour
    }

    df = tv.get_hist(
        symbol=sym_tuple[0],
        exchange=sym_tuple[1],
        interval=tv_intervals[interval_str],
        n_bars=1500
    )

    if df is None or df.empty:
        return pd.DataFrame()

    # המרה ל-Asia/Jerusalem ואז הסרת timezone (חשוב ל-Plotly!)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert("Asia/Jerusalem")
    else:
        df.index = df.index.tz_convert("Asia/Jerusalem")

    df.index = df.index.tz_localize(None)

    return df[['close']]

# ==========================================
# נכסים
# ==========================================
DEFAULT_TICKERS = {
    "לאומי": ("LUMI", "TASE"),
    "פועלים": ("POLI", "TASE"),
    "מדד ת\"א 35": ("TA35", "TASE"),
    "S&P 500 ETF": ("SPY", "AMEX"),
    'נאסד\"ק 100 ETF': ("QQQ", "NASDAQ"),
    "USD/ILS": ("USDILS", "FX_IDC")
}

col1, col2, col3 = st.columns(3)

with col1:
    asset_name = st.selectbox("בחר נכס:", list(DEFAULT_TICKERS.keys()))
    target_tuple = DEFAULT_TICKERS[asset_name]

with col2:
    interval_choice = st.selectbox("רזולוציה:", ["1d", "60m"])

with col3:
    backtest = st.selectbox("Backtest:", [0, 5, 21, 63, 126])

if interval_choice == "60m" and backtest > 0:
    backtest *= 8

# ==========================================
# חיזוי
# ==========================================
if st.button("🔮 הפעל חיזוי", type="primary", use_container_width=True):

    with st.spinner("טוען מודל..."):
        tfm_model = load_ai_model()

    with st.spinner("מושך נתונים..."):
        df_hist = fetch_data_tv(target_tuple, interval_choice)

    if df_hist.empty or (len(df_hist) - backtest) < 512:
        st.error("אין מספיק נתונים (נדרשים לפחות 512 נרות)")
        st.stop()

    if backtest > 0:
        df_train = df_hist.iloc[:-backtest]
        df_actual = df_hist.iloc[-backtest:]
    else:
        df_train = df_hist
        df_actual = pd.DataFrame()

    prices_array = df_train['close'].values

    with st.spinner("המודל מחשב תחזית..."):
        forecast_results, quantiles_results = tfm_model.forecast([prices_array], freq=[0])

    future_prices = forecast_results[0]
    lower_bound = quantiles_results[0, :, 0]
    upper_bound = quantiles_results[0, :, -1]

    last_train_date = df_train.index[-1]
    last_train_price = df_train['close'].iloc[-1]

    # יצירת ציר זמן עתידי
    if interval_choice == "1d":
        forecast_dates = pd.bdate_range(
            start=last_train_date + pd.Timedelta(days=1),
            periods=128
        )
    else:
        forecast_dates = pd.date_range(
            start=last_train_date + pd.Timedelta(hours=1),
            periods=128,
            freq="H"
        )

    forecast_df = pd.DataFrame({
        "Forecast": future_prices,
        "Lower": lower_bound,
        "Upper": upper_bound
    }, index=forecast_dates)

    # ==========================================
    # גרף
    # ==========================================
    fig = go.Figure()

    display_hist = df_train.tail(200)

    # ענן תחזית
    fig.add_trace(go.Scatter(
        x=forecast_df.index,
        y=forecast_df["Upper"],
        mode="lines",
        line=dict(width=0),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df.index,
        y=forecast_df["Lower"],
        mode="lines",
        fill="tonexty",
        fillcolor="rgba(245,158,11,0.2)",
        line=dict(width=0),
        name="טווח סביר"
    ))

    # תחזית
    fig.add_trace(go.Scatter(
        x=forecast_df.index,
        y=forecast_df["Forecast"],
        mode="lines",
        name="תחזית AI",
        line=dict(color="#f59e0b", width=3, dash="dash")
    ))

    # היסטוריה
    fig.add_trace(go.Scatter(
        x=display_hist.index,
        y=display_hist["close"],
        mode="lines",
        name="היסטוריה",
        line=dict(color="#2563eb", width=2)
    ))

    # מציאות
    if not df_actual.empty:
        fig.add_trace(go.Scatter(
            x=df_actual.index,
            y=df_actual["close"],
            mode="lines",
            name="מה קרה בפועל",
            line=dict(color="#10b981", width=3)
        ))

        # קו חיתוך (בלי annotation פנימי!)
        fig.add_vline(
            x=last_train_date,
            line_width=2,
            line_dash="dot",
            line_color="#94a3b8"
        )

        fig.add_annotation(
            x=last_train_date,
            y=1,
            yref="paper",
            text="נקודת החיתוך (כאן המודל עוור)",
            showarrow=False,
            xanchor="left",
            yanchor="bottom"
        )

    fig.update_layout(
        title=f"חיזוי מסלול מחיר: {asset_name}",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h")
    )

    st.plotly_chart(fig, use_container_width=True)

st.caption("למחקר בלבד. אין באמור המלצה להשקעה.")
