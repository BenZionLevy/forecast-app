import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tvDatafeed import TvDatafeed, Interval
import timesfm
import io
import time

st.set_page_config(page_title="חיזוי מניות AI", layout="wide", page_icon="📈")

# =========================
# טעינת מודל
# =========================
@st.cache_resource(show_spinner=False)
def load_model():
    return timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="cpu",
            per_core_batch_size=1,
            horizon_len=128,
            context_len=1024,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
        ),
    )

# =========================
# נכסים
# =========================
ASSETS = {
    "לאומי": ("LUMI", "TASE"),
    "פועלים": ("POLI", "TASE"),
    "דיסקונט": ("DSCT", "TASE"),
    "מזרחי טפחות": ("MZTF", "TASE"),
    "אלביט מערכות": ("ESLT", "TASE"),
    "טבע": ("TEVA", "TASE"),
    "נייס": ("NICE", "TASE"),
    "בזק": ("BEZQ", "TASE"),
    "דלק קבוצה": ("DLEKG", "TASE"),
    "מדד ת\"א 35": ("TA35", "TASE"),
    "S&P 500 ETF": ("SPY", "AMEX"),
    "נאסד\"ק 100 ETF": ("QQQ", "NASDAQ"),
    "USD/ILS": ("USDILS", "FX_IDC"),
}

# =========================
# משיכת נתונים בטוחה
# =========================
@st.cache_data(ttl=600)
def fetch_data(symbol, interval_str):
    tv = TvDatafeed(username=None, password=None)

    tv_map = {
        "1d": Interval.in_daily,
        "60m": Interval.in_1_hour,
        "15m": Interval.in_15_minute,
        "5m": Interval.in_5_minute,
        "1W": Interval.in_weekly,
    }

    inter = tv_map.get(interval_str, Interval.in_daily)

    df = tv.get_hist(
        symbol=symbol[0],
        exchange=symbol[1],
        interval=inter,
        n_bars=2000,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    # ניקוי נתונים
    df = df[['close']].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    df.index = df.index.tz_convert("Asia/Jerusalem").tz_localize(None)

    return df

# =========================
# חיזוי יציב מספרית
# =========================
def get_forecast(model, ctx_prices, method="שערים", horizon=128):

    ctx_prices = np.array(ctx_prices, dtype=float)
    ctx_prices = ctx_prices[~np.isnan(ctx_prices)]

    if len(ctx_prices) < 1024:
        raise ValueError("פחות מ-1024 נקודות הקשר")

    ctx_prices = ctx_prices[-1024:]

    if "שערים" in method:

        forecast_res, quant_res = model.forecast([ctx_prices], freq=[0])

        fc = forecast_res[0][:horizon]
        lower = quant_res[0, :horizon, 0]
        upper = quant_res[0, :horizon, -1]

        return fc, lower, upper

    else:
        returns = np.diff(ctx_prices) / ctx_prices[:-1]
        returns = np.nan_to_num(returns)

        if len(returns) < 1024:
            raise ValueError("פחות מ-1023 תשואות")

        returns = returns[-1024:]

        forecast_res, quant_res = model.forecast([returns], freq=[0])

        fc_ret = np.clip(forecast_res[0][:horizon], -0.2, 0.2)
        lower_ret = np.clip(quant_res[0, :horizon, 0], -0.2, 0.2)
        upper_ret = np.clip(quant_res[0, :horizon, -1], -0.2, 0.2)

        last_price = ctx_prices[-1]

        fc_price = last_price * np.cumprod(1 + fc_ret)
        lower_price = last_price * np.cumprod(1 + lower_ret)
        upper_price = last_price * np.cumprod(1 + upper_ret)

        return fc_price, lower_price, upper_price

# =========================
# גרף
# =========================
def create_figure(ctx_dates, ctx_prices, fc_dates, fc, lower, upper):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=ctx_dates[-200:],
        y=ctx_prices[-200:],
        mode="lines",
        name="היסטוריה",
        line=dict(color="#2563eb")
    ))

    fig.add_trace(go.Scatter(
        x=fc_dates,
        y=upper,
        mode="lines",
        line=dict(width=0),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=fc_dates,
        y=lower,
        mode="lines",
        fill="tonexty",
        fillcolor="rgba(245,158,11,0.2)",
        line=dict(width=0),
        name="טווח הסתברות"
    ))

    fig.add_trace(go.Scatter(
        x=fc_dates,
        y=fc,
        mode="lines",
        name="תחזית AI",
        line=dict(color="#f59e0b", dash="dash")
    ))

    fig.update_layout(template="plotly_white", hovermode="x unified")
    return fig

# =========================
# ממשק
# =========================
st.title("📈 חיזוי מניות ומדדים – Google TimesFM")

stock = st.selectbox("בחר נכס", list(ASSETS.keys()))
interval = st.selectbox("רזולוציה", ["1d", "60m", "15m", "5m"])

if st.button("🚀 הפעל ניתוח"):

    model = load_model()

    df = fetch_data(ASSETS[stock], interval)

    if df.empty:
        st.error("לא נמצאו נתונים")
        st.stop()

    prices = df['close'].values
    dates = df.index

    try:
        fc, lower, upper = get_forecast(model, prices, method="תשואות")

    except Exception as e:
        st.error(f"שגיאת חיזוי: {e}")
        st.stop()

    last_date = dates[-1]
    fc_dates = pd.date_range(
        start=last_date,
        periods=129,
        freq="D"
    )[1:]

    fig = create_figure(dates, prices, fc_dates, fc, lower, upper)
    st.plotly_chart(fig, use_container_width=True)

    # חישוב MAPE בטוח
    if len(prices) > 150:
        actual = prices[-30:]
        pred = prices[-31:-1]

        denom = np.where(actual == 0, 1e-8, actual)
        mape = np.mean(np.abs((actual - pred) / denom)) * 100

        st.info(f"MAPE משוער (30 תקופות אחרונות): {mape:.2f}%")

st.markdown("---")
st.caption("המערכת לצורכי מחקר בלבד")
