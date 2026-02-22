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
# עיצוב בהיר מקצועי (מיושר לימין)
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
# טעינת מודל AI (נשמר בזיכרון)
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
# נכסים לבחירה
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

# =========================
# פונקציות משיכה ויצירת גרפים
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
    
    if df is None or df.empty: return pd.DataFrame()
    if df.index.tz is None: df.index = df.index.tz_localize("UTC").tz_convert("Asia/Jerusalem")
    else: df.index = df.index.tz_convert("Asia/Jerusalem")
    df.index = df.index.tz_localize(None) 
    return df[['close']]

# פונקציה לייצור הגרף (מונעת שכפול קוד ומשמשת גם לגרף הראשי וגם לחלון הצף)
def create_forecast_figure(data_dict):
    ctx_dates, ctx_prices = data_dict['ctx_dates'], data_dict['ctx_prices']
    actual_dates, actual_prices = data_dict['actual_dates'], data_dict['actual_prices']
    fcst_dates, fcst_prices = data_dict['fcst_dates'], data_dict['fcst_prices']
    fcst_lower, fcst_upper = data_dict['fcst_lower'], data_dict['fcst_upper']
    c_val = data_dict['c_val']
    
    last_date = ctx_dates[-1]
    last_price = ctx_prices[-1]
    
    conn_dates = [last_date] + list(fcst_dates)
    conn_fcst = [last_price] + list(fcst_prices)
    conn_lower = [last_price] + list(fcst_lower)
    conn_upper = [last_price] + list(fcst_upper)
    
    fig = go.Figure()
    
    # היסטוריה
    fig.add_trace(go.Scatter(x=ctx_dates[-200:], y=ctx_prices[-200:], mode="lines", name="היסטוריה (בסיס לחיזוי)", line=dict(color='#2563eb', width=2)))
    # גבול עליון לענן
    fig.add_trace(go.Scatter(x=conn_dates, y=conn_upper, mode="lines", line=dict(width=0), showlegend=False, hoverinfo='skip'))
    # גבול תחתון לענן (ממלא שטח למעלה)
    fig.add_trace(go.Scatter(x=conn_dates, y=conn_lower, mode="lines", fill="tonexty", fillcolor="rgba(245, 158, 11, 0.2)", line=dict(width=0), name="טווח הסתברות (AI)"))
    # קו התחזית
    fig.add_trace(go.Scatter(x=conn_dates, y=conn_fcst, mode="lines", name="תחזית AI", line=dict(color='#f59e0b', width=2.5, dash="dash")))

    if c_val > 0: # תוספת מציאות בבדיקת Backtest
        conn_act_dates = [last_date] + list(actual_dates)
        conn_act_prices = [last_price] + list(actual_prices)
        fig.add_trace(go.Scatter(x=conn_act_dates, y=conn_act_prices, mode="lines", name="מה קרה בפועל (המציאות)", line=dict(color='#10b981', width=3)))
        
        # קו הפרדה (נקודת עיוורון)
        fig.add_vline(x=str(last_date), line_width=2, line_dash="dot", line_color="#94a3b8")
        fig.add_annotation(x=str(last_date), y=1.05, yref="paper", text="נקודת עיוורון", showarrow=False, font=dict(color="#94a3b8", size=12), xanchor="center")

    fig.update_layout(
        template="plotly_white", 
        hovermode="x unified", 
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), 
        margin=dict(l=10, r=10, t=40, b=10)
    )
    # אילוץ צפיפות גדולה יותר של תאריכים בציר ה-X עם זווית נוחה לקריאה
    fig.update_xaxes(nticks=25, tickangle=-45)

    return fig

# חלון צף להצגת גרף כשלוחצים על שורה בטבלה ההיסטורית
@st.dialog("📊 גרף בדיקת עבר - מודל חיזוי מול מציאות", width="large")
def show_chart_dialog(c_idx):
    data = st.session_state['backtest_data'][c_idx]
    fig = create_forecast_figure(data)
    st.plotly_chart(fig, use_container_width=True)

# =========================
# הפעלת הלולאה המרכזית
# =========================
if st.button("🚀 הפעל ניתוח AI מקיף", type="primary", use_container_width=True):

    with st.spinner("טוען מודל ומושך נתונים מ-TradingView..."):
        model = load_model()
        df = fetch_data(ASSETS[stock], interval_choice)

    if df.empty or len(df) < 600:
        st.error("❌ אין מספיק נתונים עבור הנכס הזה (דרושים לפחות 600 תצפיות לעבודה תקינה).")
        st.stop()

    # הגדרת תקופות זמן בהתאם לרזולוציה
    if interval_choice == "1d":
        unit = "ימי מסחר"
        test_cutoffs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 21, 63, 126]
        test_labels = ["חיזוי עתידי אמיתי (היום והלאה)"] + [f"{c} {unit} אחורה" for c in test_cutoffs[1:11]] + ["חודש (21 ימים) אחורה", "3 חודשים (63 ימים) אחורה", "חצי שנה (126 ימים) אחורה"]
    else:
        unit = "תקופות זמן"
        test_cutoffs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100]
        test_labels = ["חיזוי עתידי אמיתי (היום והלאה)"] + [f"{c} {unit} אחורה" for c in test_cutoffs[1:]]

    st.session_state['test_cutoffs'] = test_cutoffs
    st.session_state['backtest_data'] = {}
    results_list = []

    prices_full = df['close'].values
    dates_full = df.index

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, (c, label) in enumerate(zip(test_cutoffs, test_labels)):
        status_text.text(f"מחשב מודל עבור: {label}...")
        
        if len(prices_full) - c >= 512:
            if c > 0:
                ctx_prices = prices_full[:-c]
                ctx_dates = dates_full[:-c]
                actual_prices = prices_full[-c:]
                actual_dates = dates_full[-c:]
            else:
                ctx_prices = prices_full
                ctx_dates = dates_full
                actual_prices = []
                actual_dates = []

            last_date = ctx_dates[-1]
            last_price = ctx_prices[-1]

            try:
                forecast_res, quant_res = model.forecast([ctx_prices], freq=[0])
                fcst_prices = forecast_res[0]
                fcst_lower = quant_res[0, :, 0]
                fcst_upper = quant_res[0, :, -1]

                # יצירת ציר זמן עתידי מדויק לפי רזולוציה
                if interval_choice == "1d": fcst_dates = pd.bdate_range(start=last_date, periods=129)[1:]
                elif interval_choice == "1W": fcst_dates = pd.date_range(start=last_date, periods=129, freq="W")[1:]
                else:
                    freq_str = interval_choice.replace('m', 'min')
                    fcst_dates = pd.date_range(start=last_date, periods=129, freq=freq_str)[1:]

                # חישוב אחוזי הצלחה לתקופות עבר
                if c > 0:
                    pred_for_actual = fcst_prices[:c]
                    mape = np.mean(np.abs((actual_prices - pred_for_actual) / actual_prices)) * 100
                    act_dir = actual_prices[-1] - last_price
                    pred_dir = pred_for_actual[-1] - last_price
                    is_correct = (act_dir > 0 and pred_dir > 0) or (act_dir < 0 and pred_dir < 0)
                    
                    trend_str = "✅ קלע לכיוון" if is_correct else "❌ טעה בכיוון"
                    mape_str = f"{mape:.2f}%"
                else:
                    trend_str = "🔮 עתיד"
                    mape_str = "---"
                    is_correct = None

                # נוסיף לטבלה רק את שורות ה-Backtest (העתיד מוצג בגרף נפרד למעלה)
                if c > 0:
                    results_list.append({
                        "נקודת התחלה": label,
                        "סטייה מהמציאות (MAPE)": mape_str,
                        "זיהוי כיוון מגמה": trend_str,
                        "_c_val": c,
                        "_is_correct": is_correct
                    })

                # שמירת הנתונים לטובת ציור הגרף (עתידי וחלונות צפים)
                st.session_state['backtest_data'][c] = {
                    'ctx_dates': ctx_dates, 'ctx_prices': ctx_prices,
                    'actual_dates': actual_dates, 'actual_prices': actual_prices,
                    'fcst_dates': fcst_dates, 'fcst_prices': fcst_prices,
                    'fcst_lower': fcst_lower, 'fcst_upper': fcst_upper,
                    'c_val': c, 'label': label
                }

            except Exception as e:
                pass 
                
        progress_bar.progress((i + 1) / len(test_cutoffs))

    status_text.empty()
    progress_bar.empty()

    if results_list:
        st.session_state['results_df'] = pd.DataFrame(results_list)
        st.session_state['run_done'] = True

# =========================
# תצוגת התוצאות (גרף עתידי ואז טבלה)
# =========================
if st.session_state.get('run_done'):
    
    # 1. הצגת גרף החיזוי העתידי (האמיתי) בגדול למעלה
    st.markdown("### 📈 תחזית עתידית (מהיום והלאה)")
    future_data = st.session_state['backtest_data'][0] # אינדקס 0 זה ההווה
    fig_future = create_forecast_figure(future_data)
    st.plotly_chart(fig_future, use_container_width=True)
    
    st.divider()

    # 2. הצגת טבלת האמינות (Backtesting) ממתחת
    df_res = st.session_state['results_df']

    correct_count = sum(1 for x in df_res['_is_correct'] if x == True)
    total_tests = sum(1 for x in df_res['_is_correct'] if x is not None)
    win_rate = (correct_count / total_tests) * 100 if total_tests > 0 else 0

    display_df = df_res.drop(columns=['_c_val', '_is_correct'])

    def style_trend(val):
        if "✅" in str(val): return 'color: #047857; font-weight: bold;'
        if "❌" in str(val): return 'color: #b91c1c;'
        return ''

    styled_df = display_df.style.map(style_trend, subset=["זיהוי כיוון מגמה"])

    st.markdown("### 🔬 מבחני אמינות למודל (Backtesting)")
    st.markdown("**לחץ על שורה בטבלה כדי לפתוח את החיזוי מול המציאות בגרף מפורט** 👇")

    event = st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True,
        selection_mode="single-row",
        on_select="rerun",
        key="backtest_table"
    )

    if len(event.selection.rows) > 0:
        selected_row_idx = event.selection.rows[0]
        selected_c = df_res.iloc[selected_row_idx]['_c_val']
        show_chart_dialog(selected_c)

    if total_tests > 0:
        if win_rate >= 60:
            st.success(f"🏆 **ציון אמינות כללי:** {win_rate:.0f}% הצלחה בזיהוי המגמה. (נחשב למודל יציב ואמין עבור הנכס הזה)")
        elif win_rate <= 40:
            st.error(f"⚠️ **ציון אמינות כללי:** {win_rate:.0f}% הצלחה בזיהוי המגמה. (המודל מתקשה לקרוא את הנכס הזה, לא מומלץ להסתמך עליו כאן)")
        else:
            st.warning(f"⚖️ **ציון אמינות כללי:** {win_rate:.0f}% הצלחה בזיהוי המגמה. (תוצאה בינונית - כדאי לשלב כלים נוספים בהחלטה)")

st.divider()
st.markdown("""
<div style='text-align: center; color: #64748b; font-size: 0.85rem; padding-top: 1rem; padding-bottom: 2rem; direction: rtl;'>
    מודל החיזוי מופעל באמצעות Google TimesFM 1.0. האתר לצורכי מחקר, ועל אחריות המשתמש.<br>
    לשיתופי פעולה ניתן לפנות ליוצר במייל: <a href="mailto:147590@gmail.com" style="color: #3b82f6; text-decoration: none;" dir="ltr">147590@gmail.com</a>
</div>
""", unsafe_allow_html=True)
