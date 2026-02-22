import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from tvDatafeed import TvDatafeed, Interval
import timesfm

st.set_page_config(
    page_title="חיזוי מניות ת״א-35",
    layout="wide",
    page_icon="📊"
)

# =============================
# עיצוב ברוקר כהה
# =============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Heebo:wght@300;400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Heebo', sans-serif;
    direction: rtl;
}

.stApp {
    background-color: #0f172a;
    color: white;
}

h1, h2, h3, h4 {
    text-align: right;
}

.section-box {
    background-color: #1e293b;
    padding: 1.2rem;
    border-radius: 12px;
    margin-bottom: 1.2rem;
}

.warning-box {
    background-color: #7c2d12;
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    font-size: 0.9rem;
}

button[kind="primary"] {
    background-color: #2563eb !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>📊 חיזוי מניות – מדד ת״א-35</h1>", unsafe_allow_html=True)

# =============================
# אזהרה עליונה
# =============================
st.markdown("""
<div class="warning-box">
⚠️ המערכת לצורכי מחקר בלבד. החיזוי אינו מתחשב בחדשות או אירועים כלכליים ואינו מהווה ייעוץ השקעות.
</div>
""", unsafe_allow_html=True)

# =============================
# טעינת מודל
# =============================
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

# =============================
# מניות ת״א-35
# =============================
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

# =============================
# בחירה עליונה – נקי וברור
# =============================
col1, col2 = st.columns(2)

with col1:
    stock = st.selectbox("בחר מניה", list(TA35.keys()))

with col2:
    resolution = st.selectbox("רזולוציה", {
        "יומי": "1d",
        "שעתי": "60m"
    })

interval_choice = {
    "יומי": "1d",
    "שעתי": "60m"
}[resolution]

# =============================
# הפרדה בין עתידי להיסטורי
# =============================
mode = st.radio(
    "סוג החיזוי",
    ["🔮 חיזוי עתידי", "📈 בדיקה היסטורית (Backtest)"],
    horizontal=True
)

if mode == "📈 בדיקה היסטורית (Backtest)":
    st.markdown('<div class="section-box">', unsafe_allow_html=True)

    if interval_choice == "1d":
        back_options = {
            "שבוע אחורה": 5,
            "חודש אחורה": 21,
            "3 חודשים": 63,
            "חצי שנה": 126
        }
    else:
        back_options = {
            "יום מסחר אחורה (8 שעות)": 8,
            "3 ימים": 24,
            "שבוע": 40,
            "חודש": 160
        }

    back_label = st.selectbox("בחר טווח בדיקה", list(back_options.keys()))
    cutoff = back_options[back_label]

    st.markdown('</div>', unsafe_allow_html=True)

else:
    cutoff = 0

# =============================
# משיכת נתונים
# =============================
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

# =============================
# הפעלת חיזוי
# =============================
if st.button("הפעל חיזוי", use_container_width=True):

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
        pd.bdate_range(start=last_date, periods=128)[1:]
        if interval_choice == "1d"
        else pd.date_range(start=last_date, periods=128, freq="H")[1:]
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=train.index[-200:],
        y=train['close'].tail(200),
        mode="lines",
        name="היסטוריה",
        line=dict(color="#3b82f6", width=2)
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
        fillcolor="rgba(251,191,36,0.15)",
        line=dict(width=0),
        name="טווח הסתברות"
    ))

    fig.add_trace(go.Scatter(
        x=future_dates,
        y=forecast,
        mode="lines",
        name="תחזית AI",
        line=dict(color="#fbbf24", width=3, dash="dash")
    ))

    if not actual.empty:
        fig.add_trace(go.Scatter(
            x=actual.index,
            y=actual['close'],
            mode="lines",
            name="מה קרה בפועל",
            line=dict(color="#22c55e", width=3)
        ))

    fig.update_layout(
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="h"),
        margin=dict(l=10, r=10, t=40, b=10)
    )

    st.plotly_chart(fig, use_container_width=True)
