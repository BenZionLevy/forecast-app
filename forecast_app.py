import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from tvDatafeed import TvDatafeed, Interval
import timesfm

st.set_page_config(
    page_title="חיזוי מניות ת״א-35",
    layout="wide",
    page_icon="📈"
)

# =========================
# עיצוב בהיר מקצועי
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Assistant:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Assistant', sans-serif;
    direction: rtl;
}

.stApp {
    background-color: #f4f6f9;
}

.main-title {
    text-align:right;
    font-size:2.2rem;
    font-weight:700;
    margin-bottom:0.3rem;
}

.warning-box {
    background:#fff3cd;
    border:1px solid #ffeeba;
    padding:0.8rem;
    border-radius:8px;
    margin-bottom:1rem;
    font-size:0.9rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>📈 חיזוי מניות – מדד ת״א-35</div>", unsafe_allow_html=True)

st.markdown("""
<div class="warning-box">
⚠️ המערכת לצורכי מחקר בלבד. החיזוי אינו מהווה ייעוץ השקעות.
</div>
""", unsafe_allow_html=True)

# =========================
# מודל
# =========================
@st.cache_resource
def load_model():
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

# =========================
# מניות ת״א-35
# =========================
TA35 = {
    "לאומי": ("LUMI", "TASE"),
    "פועלים": ("POLI", "TASE"),
    "דיסקונט": ("DSCT", "TASE"),
    "מזרחי טפחות": ("MZTF", "TASE"),
    "אלביט מערכות": ("ESLT", "TASE"),
    "טבע": ("TEVA", "TASE"),
    "נייס": ("NICE", "TASE"),
    "בזק": ("BEZQ", "TASE"),
    "דלק קבוצה": ("DLEKG", "TASE"),
}

# =========================
# בחירה עליונה
# =========================
col1, col2 = st.columns(2)

with col1:
    stock = st.selectbox("בחר מניה", list(TA35.keys()))

with col2:
    resolution_label = st.selectbox("רזולוציה", ["יומי", "שעתי"])

interval_choice = "1d" if resolution_label == "יומי" else "60m"

# =========================
# סוג חיזוי
# =========================
mode = st.radio(
    "סוג חיזוי",
    ["חיזוי עתידי", "בדיקה היסטורית"],
    horizontal=True
)

cutoff = 0

if mode == "בדיקה היסטורית":
    if interval_choice == "1d":
        options = {
            "שבוע": 5,
            "חודש": 21,
            "3 חודשים": 63,
            "חצי שנה": 126
        }
    else:
        options = {
            "יום מסחר (8 שעות)": 8,
            "3 ימים": 24,
            "שבוע": 40,
            "חודש": 160
        }

    label = st.selectbox("בחר טווח בדיקה", list(options.keys()))
    cutoff = options[label]

# =========================
# נתונים
# =========================
@st.cache_data(ttl=600)
def fetch_data(symbol, interval):
    tv = TvDatafeed()
    inter = Interval.in_daily if interval == "1d" else Interval.in_1_hour

    df = tv.get_hist(
        symbol=symbol[0],
        exchange=symbol[1],
        interval=inter,
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

# =========================
# הפעלה
# =========================
if st.button("הפעל חיזוי", width="stretch"):

    model = load_model()
    df = fetch_data(TA35[stock], interval_choice)

    if df.empty or (len(df) - cutoff) < 512:
        st.error("אין מספיק נתונים לצורך חיזוי")
        st.stop()

    if cutoff > 0:
        train = df.iloc[:-cutoff]
        actual = df.iloc[-cutoff:]
    else:
        train = df
        actual = pd.DataFrame()

    forecast, quant = model.forecast([train['close'].values], freq=[0])
    forecast = forecast[0]
    lower = quant[0, :, 0]
    upper = quant[0, :, -1]

    last_date = train.index[-1]

    future_dates = (
        pd.bdate_range(start=last_date, periods=129)[1:]
        if interval_choice == "1d"
        else pd.date_range(start=last_date, periods=129, freq="h")[1:]
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=train.index[-200:],
        y=train['close'].tail(200),
        mode="lines",
        name="היסטוריה",
        line=dict(width=2)
    ))

    fig.add_trace(go.Scatter(
        x=future_dates,
        y=upper,
        mode="lines",
        line=dict(width=0),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=future_dates,
        y=lower,
        mode="lines",
        fill="tonexty",
        line=dict(width=0),
        name="טווח הסתברות"
    ))

    fig.add_trace(go.Scatter(
        x=future_dates,
        y=forecast,
        mode="lines",
        name="תחזית AI",
        line=dict(width=3, dash="dash")
    ))

    if not actual.empty:
        fig.add_trace(go.Scatter(
            x=actual.index,
            y=actual['close'],
            mode="lines",
            name="מה קרה בפועל",
            line=dict(width=3)
        ))

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h"),
        margin=dict(l=10, r=10, t=40, b=10)
    )

    st.plotly_chart(fig, width="stretch")
