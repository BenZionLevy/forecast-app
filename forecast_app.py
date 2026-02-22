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
# עיצוב בהיר מקצועי (מתוקן לימין)
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
    text-align: right;
    direction: rtl;
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
    ["חיזוי עתידי (מהיום והלאה)", "בדיקה היסטורית בודדת למחקר ממוקד"],
    horizontal=True
)

cutoff = 0

if mode == "בדיקה היסטורית בודדת למחקר ממוקד":
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

    with st.spinner("ה-AI מנתח תבניות היסטוריות ומחשב תחזית..."):
        forecast, quant = model.forecast([train['close'].values], freq=[0])
        forecast = forecast[0]
        lower = quant[0, :, 0]
        upper = quant[0, :, -1]

    last_date = train.index[-1]
    last_price = train['close'].iloc[-1]

    if interval_choice == "1d":
        future_dates = pd.bdate_range(start=last_date, periods=129)[1:]
    elif interval_choice == "1W":
        future_dates = pd.date_range(start=last_date, periods=129, freq="W")[1:]
    else:
        freq_str = interval_choice.replace('m', 'min')
        future_dates = pd.date_range(start=last_date, periods=129, freq=freq_str)[1:]

    conn_dates = [last_date] + list(future_dates)
    conn_forecast = [last_price] + list(forecast)
    conn_lower = [last_price] + list(lower)
    conn_upper = [last_price] + list(upper)

    # =========================
    # ויזואליזציה (גרף מרכזי)
    # =========================
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=train.index[-200:], y=train['close'].tail(200),
        mode="lines", name="היסטוריה (בסיס לחיזוי)", line=dict(color='#2563eb', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=conn_dates, y=conn_upper,
        mode="lines", line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))

    fig.add_trace(go.Scatter(
        x=conn_dates, y=conn_lower,
        mode="lines", fill="tonexty", fillcolor="rgba(245, 158, 11, 0.2)",
        line=dict(width=0), name="טווח הסתברות (AI)"
    ))

    fig.add_trace(go.Scatter(
        x=conn_dates, y=conn_forecast,
        mode="lines", name="תחזית AI", line=dict(color='#f59e0b', width=2.5, dash="dash")
    ))

    if not actual.empty:
        conn_act_dates = [last_date] + list(actual.index)
        conn_act_prices = [last_price] + list(actual['close'])
        
        fig.add_trace(go.Scatter(
            x=conn_act_dates, y=conn_act_prices,
            mode="lines", name="מה קרה בפועל (המציאות)", line=dict(color='#10b981', width=3)
        ))
        
        fig.add_vline(x=last_date, line_width=2, line_dash="dot", line_color="#94a3b8")
        fig.add_annotation(x=last_date, y=1.05, yref="paper", text="נקודת עיוורון", showarrow=False, font=dict(color="#94a3b8", size=12), xanchor="center")

    fig.update_layout(
        template="plotly_white", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=40, b=10)
    )

    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # בדיקת ביצועים לבדיקה בודדת
    # =========================
    if not actual.empty:
        pred_for_actual = forecast[:cutoff]
        actual_vals = actual['close'].values

        mape = np.mean(np.abs((actual_vals - pred_for_actual) / actual_vals)) * 100
        actual_direction = actual_vals[-1] - last_price
        pred_direction = pred_for_actual[-1] - last_price
        
        is_trend_correct = (actual_direction > 0 and pred_direction > 0) or (actual_direction < 0 and pred_direction < 0)
        trend_text = "✅ הצלחה (חזה נכון)" if is_trend_correct else "❌ כישלון (טעה בכיוון)"

        st.markdown("### 📊 תוצאות מבחן המציאות שהרצת")
        c1, c2, c3 = st.columns(3)
        c1.metric("סטייה ממוצעת מהמציאות (MAPE)", f"{mape:.2f}%")
        c2.metric("זיהוי מגמה", trend_text)
        c3.info("💡 **MAPE** נמוך יותר = המודל היה מדויק. **זיהוי מגמה** בודק אם המודל חזה נכון אם הנכס יעלה או ירד בסוף התקופה.")

    # =========================
    # טבלת אמינות אוטומטית (מופיעה רק במצב חיזוי עתידי)
    # =========================
    elif cutoff == 0:
        st.divider()
        st.markdown("### 🔬 טבלת אמינות היסטורית (Backtesting אוטומטי)")
        st.info("המערכת בוחנת כעת כיצד המודל היה מתפקד אם היינו מריצים אותו בנקודות זמן שונות בעבר. (הפעולה עשויה לקחת כדקה)")
        
        prices_full = df['close'].values
        
        # הגדרת הטווחי זמן לבדיקה בהתאם לרזולוציה
        if interval_choice == "1d":
            test_cutoffs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 21, 126]
            test_labels = ["1 ימים", "2 ימים", "3 ימים", "4 ימים", "5 ימים", "6 ימים", "7 ימים", "8 ימים", "9 ימים", "10 ימים", "חודש (21 ימים)", "חצי שנה (126 ימים)"]
        else:
            test_cutoffs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100]
            test_labels = [f"{c} נרות אחורה" for c in test_cutoffs]

        results_list = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, (c, label) in enumerate(zip(test_cutoffs, test_labels)):
            if len(prices_full) - c >= 512:
                status_text.text(f"בודק אמינות: חוזר {label} אחורה...")
                
                # יצירת הנתונים המוסתרים
                ctx_prices = prices_full[:-c]
                actual_hidden = prices_full[-c:]
                last_known_val = prices_full[-(c + 1)]
                
                try:
                    # מריצים את החיזוי על העבר המדומה
                    test_forecast, _ = model.forecast([ctx_prices], freq=[0])
                    test_pred = test_forecast[0][:c]
                    
                    # חישוב הסטייה
                    test_mape = np.mean(np.abs((actual_hidden - test_pred) / actual_hidden)) * 100
                    
                    # בדיקת כיוון המגמה
                    act_dir = actual_hidden[-1] - last_known_val
                    pred_dir = test_pred[-1] - last_known_val
                    is_correct = (act_dir > 0 and pred_dir > 0) or (act_dir < 0 and pred_dir < 0)
                    
                    results_list.append({
                        "טווח זמן שנבדק": label,
                        "סטייה ממוצעת מהמציאות (MAPE)": test_mape,
                        "זיהוי כיוון המגמה": "✅ קלע לכיוון" if is_correct else "❌ טעה בכיוון"
                    })
                except:
                    pass
                    
            progress_bar.progress((i + 1) / len(test_cutoffs))
            
        status_text.empty()
        progress_bar.empty()
        
        if results_list:
            res_df = pd.DataFrame(results_list)
            
            # חישוב אחוז ההצלחה הכללי (Win Rate)
            correct_count = sum(1 for r in results_list if "✅" in r["זיהוי כיוון המגמה"])
            win_rate = (correct_count / len(results_list)) * 100
            
            # עיצוב הטבלה
            def style_trend(val):
                if "✅" in val: return 'color: #047857; font-weight: bold;'
                if "❌" in val: return 'color: #b91c1c;'
                return ''
                
            styled_df = res_df.style.format({"סטייה ממוצעת מהמציאות (MAPE)": "{:.2f}%"}).map(style_trend, subset=["זיהוי כיוון המגמה"])
            
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            if win_rate > 60:
                st.success(f"🏆 **ציון אמינות כללי למודל על נכס זה: {win_rate:.0f}% הצלחה בזיהוי כיוון.** (המודל נחשב כאמין יחסית למניה זו).")
            elif win_rate < 40:
                st.error(f"⚠️ **ציון אמינות כללי למודל על נכס זה: {win_rate:.0f}% הצלחה בזיהוי כיוון.** (לא מומלץ להסתמך על החיזוי העתידי במקרה הזה).")
            else:
                st.warning(f"⚖️ **ציון אמינות כללי למודל על נכס זה: {win_rate:.0f}% הצלחה בזיהוי כיוון.** (תוצאה בינונית - כדאי לשלב כלים נוספים).")

st.divider()
st.markdown("""
<div style='text-align: center; color: #64748b; font-size: 0.85rem; padding-top: 1rem; padding-bottom: 2rem; direction: rtl;'>
    מודל החיזוי מופעל באמצעות Google TimesFM 1.0. האתר לצורכי מחקר, ועל אחריות המשתמש.<br>
    לשיתופי פעולה ניתן לפנות ליוצר במייל: <a href="mailto:147590@gmail.com" style="color: #3b82f6; text-decoration: none;" dir="ltr">147590@gmail.com</a>
</div>
""", unsafe_allow_html=True)
