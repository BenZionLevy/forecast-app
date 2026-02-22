import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tvDatafeed import TvDatafeed, Interval
import timesfm

st.set_page_config(
    page_title="חיזוי מניות AI",
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

st.markdown("<div class='main-title'>📈 חיזוי מניות ומדדים (Google TimesFM)</div>", unsafe_allow_html=True)

st.markdown("""
<div class="warning-box">
⚠️ המערכת נועדה לצורכי מחקר סטטיסטי בלבד. מודל החיזוי אינו מהווה ייעוץ השקעות.
</div>
""", unsafe_allow_html=True)

# =========================
# מודל AI
# =========================
@st.cache_resource(show_spinner=False)
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
# נכסים לבחירה (כולל מאקרו)
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
    'נאסד"ק 100 ETF': ("QQQ", "NASDAQ"), 
    "USD/ILS (דולר-שקל)": ("USDILS", "FX_IDC")
}

# =========================
# הגדרות ממשק משתמש
# =========================
col1, col2 = st.columns(2)

with col1:
    stock = st.selectbox("בחר נכס פיננסי", list(ASSETS.keys()))

with col2:
    int_map = {
        "5 דקות": "5m", 
        "15 דקות": "15m", 
        "30 דקות": "30m", 
        "שעתי (60m)": "60m", 
        "יומי (1d)": "1d", 
        "שבועי (1W)": "1W"
    }
    resolution_label = st.selectbox("רזולוציית זמן", list(int_map.keys()), index=4)
    interval_choice = int_map[resolution_label]

mode = st.radio(
    "סוג חיזוי",
    ["חיזוי עתידי (מהיום והלאה)", "בדיקה היסטורית (Backtesting)"],
    horizontal=True
)

cutoff = 0

if mode == "בדיקה היסטורית (Backtesting)":
    st.info("💡 בחר כמה נרות (תצפיות) להסתיר מהמודל כדי לבחון את הדיוק שלו מול מה שקרה בפועל.")
    cutoff = st.number_input("כמה נרות לחזור אחורה אל תוך העבר?", min_value=1, max_value=128, value=30)

# =========================
# משיכת נתונים
# =========================
@st.cache_data(ttl=600, show_spinner=False)
def fetch_data(symbol, interval_str):
    tv = TvDatafeed()
    tv_intervals = {
        "5m": Interval.in_5_minute,
        "15m": Interval.in_15_minute,
        "30m": Interval.in_30_minute,
        "60m": Interval.in_1_hour,
        "1d": Interval.in_daily,
        "1W": Interval.in_weekly
    }
    inter = tv_intervals.get(interval_str, Interval.in_daily)

    df = tv.get_hist(symbol=symbol[0], exchange=symbol[1], interval=inter, n_bars=1500)

    if df is None or df.empty:
        return pd.DataFrame()

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert("Asia/Jerusalem")
    else:
        df.index = df.index.tz_convert("Asia/Jerusalem")

    # הסרת אזור הזמן מונעת באגים בהצגת הגרף ב-Plotly
    df.index = df.index.tz_localize(None) 
    return df[['close']]

# =========================
# הפעלה ועיבוד
# =========================
if st.button("🚀 הפעל חיזוי AI עכשיו", type="primary", use_container_width=True):

    with st.spinner("טוען מודל ומושך נתונים מ-TradingView..."):
        model = load_model()
        df = fetch_data(ASSETS[stock], interval_choice)

    if df.empty or (len(df) - cutoff) < 512:
        st.error("❌ אין מספיק נתונים לצורך חיזוי. המודל דורש מינימום 512 נרות היסטוריים פנויים.")
        st.stop()

    if cutoff > 0:
        train = df.iloc[:-cutoff]
        actual = df.iloc[-cutoff:]
    else:
        train = df
        actual = pd.DataFrame()

    with st.spinner("ה-AI מנתח תבניות היסטוריות ומחשב תחזית לעתיד..."):
        # חשוב: פרמטר freq=[0] נדרש בגרסה החדשה של TimesFM
        forecast, quant = model.forecast([train['close'].values], freq=[0])
        forecast = forecast[0]
        lower = quant[0, :, 0]
        upper = quant[0, :, -1]

    # יצירת צירי זמן לחיזוי
    last_date = train.index[-1]
    last_price = train['close'].iloc[-1]

    # יצירת תאריכים עתידיים בהתאם לרזולוציה
    if interval_choice == "1d":
        future_dates = pd.bdate_range(start=last_date, periods=129)[1:]
    elif interval_choice == "1W":
        future_dates = pd.date_range(start=last_date, periods=129, freq="W")[1:]
    else:
        freq_str = interval_choice.replace('m', 'min')
        future_dates = pd.date_range(start=last_date, periods=129, freq=freq_str)[1:]

    # חיבור הנקודות כדי למנוע נתק בגרף
    conn_dates = [last_date] + list(future_dates)
    conn_forecast = [last_price] + list(forecast)
    conn_lower = [last_price] + list(lower)
    conn_upper = [last_price] + list(upper)

    # =========================
    # ויזואליזציה (גרפים)
    # =========================
    fig = go.Figure()

    # קו היסטוריה
    fig.add_trace(go.Scatter(
        x=train.index[-200:],
        y=train['close'].tail(200),
        mode="lines",
        name="היסטוריה (בסיס לחיזוי)",
        line=dict(color='#2563eb', width=2)
    ))

    # גבול עליון (שקוף) לענן ההסתברות
    fig.add_trace(go.Scatter(
        x=conn_dates,
        y=conn_upper,
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))

    # גבול תחתון ומילוי ענן ההסתברות
    fig.add_trace(go.Scatter(
        x=conn_dates,
        y=conn_lower,
        mode="lines",
        fill="tonexty",
        fillcolor="rgba(245, 158, 11, 0.2)",
        line=dict(width=0),
        name="טווח הסתברות (AI)"
    ))

    # קו תחזית AI מרכזי
    fig.add_trace(go.Scatter(
        x=conn_dates,
        y=conn_forecast,
        mode="lines",
        name="תחזית AI עתידית",
        line=dict(color='#f59e0b', width=2.5, dash="dash")
    ))

    # אם אנחנו במצב בדיקה לאחור - נוסיף את מה שקרה באמת
    if not actual.empty:
        conn_act_dates = [last_date] + list(actual.index)
        conn_act_prices = [last_price] + list(actual['close'])
        
        fig.add_trace(go.Scatter(
            x=conn_act_dates,
            y=conn_act_prices,
            mode="lines",
            name="מה קרה בפועל (המציאות)",
            line=dict(color='#10b981', width=3)
        ))
        
        # קו מקווקו אנכי המסמן את נקודת העיוורון של המודל
        fig.add_vline(x=last_date, line_width=2, line_dash="dot", line_color="#94a3b8", annotation_text="נקודת עיוורון", annotation_position="top left")

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=40, b=10)
    )

    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # בדיקת ביצועים (Metrics)
    # =========================
    if not actual.empty:
        # ניקח את התחזיות רק לאורך התקופה שיש לנו עליה מציאות (cutoff)
        pred_for_actual = forecast[:cutoff]
        actual_vals = actual['close'].values

        # חישוב אחוז שגיאה ממוצע (MAPE)
        mape = np.mean(np.abs((actual_vals - pred_for_actual) / actual_vals)) * 100

        # בדיקת מגמה (האם שניהם עלו או שניהם ירדו ביחס לנקודת ההתחלה)
        actual_direction = actual_vals[-1] - last_price
        pred_direction = pred_for_actual[-1] - last_price
        
        is_trend_correct = (actual_direction > 0 and pred_direction > 0) or (actual_direction < 0 and pred_direction < 0)
        trend_text = "✅ הצלחה (חזה את הכיוון)" if is_trend_correct else "❌ כישלון (טעה בכיוון המגמה)"

        st.markdown("### 📊 תוצאות מבחן המציאות (Backtest)")
        c1, c2, c3 = st.columns(3)
        c1.metric("סטייה ממוצעת מהמציאות (MAPE)", f"{mape:.2f}%")
        c2.metric("זיהוי מגמה", trend_text)
        c3.info("💡 **MAPE** נמוך יותר = המודל היה מדויק וקרוב יותר לקו הירוק. \n\n **זיהוי מגמה** בודק אם המודל צדק לפחות בשאלה האם הנכס יעלה או ירד בסוף התקופה.")
