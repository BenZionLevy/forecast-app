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
# עיצוב בהיר מקצועי (אכיפת RTL)
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Assistant:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Assistant', sans-serif;
    direction: rtl;
    text-align: right;
}

div[data-testid="stMarkdownContainer"] {
    direction: rtl;
    text-align: right;
}

div[data-testid="stAlert"] {
    direction: rtl;
    text-align: right;
}

.stApp {
    background-color: #f4f6f9;
}

.main-title {
    text-align: right;
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}

.warning-box {
    background: #fff3cd;
    border: 1px solid #ffeeba;
    padding: 0.8rem;
    border-radius: 8px;
    margin-bottom: 1rem;
    font-size: 0.9rem;
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
            context_len=1024,
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
    mode = st.radio(
        "סוג חיזוי",
        ["חיזוי עתידי רגיל", "בדיקה היסטורית (Backtest)", "חיזוי רב-שכבתי (Multi-Timeframe)"],
        horizontal=False
    )

interval_choice = "1d"
cutoff = 0

if mode != "חיזוי רב-שכבתי (Multi-Timeframe)":
    int_map = {"5 דקות": "5m", "15 דקות": "15m", "30 דקות": "30m", "שעתי (60m)": "60m", "יומי (1d)": "1d", "שבועי (1W)": "1W"}
    resolution_label = st.selectbox("רזולוציית זמן (עבור חיזוי רגיל/היסטורי)", list(int_map.keys()), index=4)
    interval_choice = int_map[resolution_label]

if mode == "בדיקה היסטורית (Backtest)":
    st.info("💡 בחר כמה תצפיות להסתיר מהמודל כדי לבחון את הדיוק שלו מול מה שקרה בפועל.")
    cutoff = st.number_input("כמה נרות לחזור אחורה אל תוך העבר?", min_value=1, max_value=128, value=30)
elif mode == "חיזוי רב-שכבתי (Multi-Timeframe)":
    st.info("🧬 **מצב מחקר מתקדם:** מציג את ההצטלבות בין המגמה הקצרה לארוכה. הגרפים נחתכו בצורה חכמה כדי שיהיה אפשר לראות את כולם מקרוב.")

# =========================
# מנוע תאריכים מותאם לבורסה
# =========================
def generate_israel_trading_dates(start_date, periods, tf):
    """
    מייצר תאריכים עתידיים תוך דילוג על שישי-שבת.
    ברזולוציה תוך יומית, מדלג על שעות הלילה ומתמקד ב-10:00 עד 17:00.
    """
    dates = []
    curr = start_date
    
    # הגדרת קפיצת הזמן (Step)
    if tf == "60m": step = pd.Timedelta(hours=1)
    elif tf == "30m": step = pd.Timedelta(minutes=30)
    elif tf == "15m": step = pd.Timedelta(minutes=15)
    elif tf == "5m": step = pd.Timedelta(minutes=5)
    elif tf == "1W": step = pd.Timedelta(weeks=1)
    else: step = pd.Timedelta(days=1)
    
    while len(dates) < periods:
        curr += step
        
        # שבועי פשוט קופץ קדימה
        if tf == "1W":
            dates.append(curr)
            continue
            
        # ימי מסחר בישראל: ראשון(6), שני(0), שלישי(1), רביעי(2), חמישי(3)
        is_trading_day = curr.weekday() in [6, 0, 1, 2, 3]
        
        if tf == "1d":
            if is_trading_day:
                dates.append(curr)
        else: # רזולוציה תוך יומית (שעות/דקות)
            is_trading_hour = 10 <= curr.hour < 17
            if is_trading_day and is_trading_hour:
                dates.append(curr)
                
    return dates

# =========================
# פונקציות משיכה ויצירת גרפים
# =========================
@st.cache_data(ttl=600, show_spinner=False)
def fetch_data(symbol, interval_str):
    tv = TvDatafeed()
    tv_intervals = {"5m": Interval.in_5_minute, "15m": Interval.in_15_minute, "30m": Interval.in_30_minute, "60m": Interval.in_1_hour, "1d": Interval.in_daily, "1W": Interval.in_weekly}
    inter = tv_intervals.get(interval_str, Interval.in_daily)
    df = tv.get_hist(symbol=symbol[0], exchange=symbol[1], interval=inter, n_bars=4000)
    
    if df is None or df.empty: return pd.DataFrame()
    if df.index.tz is None: df.index = df.index.tz_localize("UTC").tz_convert("Asia/Jerusalem")
    else: df.index = df.index.tz_convert("Asia/Jerusalem")
    df.index = df.index.tz_localize(None) 
    return df[['close']]

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
    
    fig.add_trace(go.Scatter(x=ctx_dates[-200:], y=ctx_prices[-200:], mode="lines", name="היסטוריה (בסיס לחיזוי)", line=dict(color='#2563eb', width=2)))
    fig.add_trace(go.Scatter(x=conn_dates, y=conn_upper, mode="lines", line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=conn_dates, y=conn_lower, mode="lines", fill="tonexty", fillcolor="rgba(245, 158, 11, 0.2)", line=dict(width=0), name="טווח הסתברות (AI)"))
    fig.add_trace(go.Scatter(x=conn_dates, y=conn_fcst, mode="lines", name="תחזית AI", line=dict(color='#f59e0b', width=2.5, dash="dash")))

    if c_val > 0:
        conn_act_dates = [last_date] + list(actual_dates)
        conn_act_prices = [last_price] + list(actual_prices)
        fig.add_trace(go.Scatter(x=conn_act_dates, y=conn_act_prices, mode="lines", name="מה קרה בפועל (המציאות)", line=dict(color='#10b981', width=3)))
        fig.add_vline(x=str(last_date), line_width=2, line_dash="dot", line_color="#94a3b8")
        fig.add_annotation(x=str(last_date), y=1.05, yref="paper", text="נקודת עיוורון", showarrow=False, font=dict(color="#94a3b8", size=12), xanchor="center")

    fig.update_layout(template="plotly_white", hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), margin=dict(l=10, r=10, t=40, b=80))
    fig.update_xaxes(nticks=25, tickangle=-45, automargin=True)

    return fig

@st.dialog("📊 גרף בדיקת עבר - מודל חיזוי מול מציאות", width="large")
def show_chart_dialog(c_idx):
    data = st.session_state['backtest_data'][c_idx]
    fig = create_forecast_figure(data)
    st.plotly_chart(fig, use_container_width=True)

# =========================
# הפעלת הלולאה והחישובים
# =========================
if st.button("🚀 הפעל ניתוח AI מקיף", type="primary", use_container_width=True):

    with st.spinner("טוען מודל ומושך נתונים מקסימליים מ-TradingView..."):
        model = load_model()
        
    # מסלול 1: חיזוי רב-שכבתי (Multi-Timeframe)
    if mode == "חיזוי רב-שכבתי (Multi-Timeframe)":
        
        tfs = {"1d": ("יומי", "#f59e0b"), "60m": ("שעתי", "#8b5cf6"), "15m": ("15 דקות", "#ef4444")}
        
        fig_mtf = go.Figure()
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # כדי לצייר היסטוריה אחידה ויפה ברקע, נשתמש בגרף השעתי כבסיס
        bg_df = fetch_data(ASSETS[stock], "60m")
        if not bg_df.empty:
            # נמשוך רק את 150 השעות האחרונות להיסטוריה כדי להתמקד בעתיד
            fig_mtf.add_trace(go.Scatter(x=bg_df.index[-150:], y=bg_df['close'].tail(150), mode="lines", name="היסטוריה קרובה", line=dict(color='#cbd5e1', width=1.5)))

        for i, (tf, (name, color)) in enumerate(tfs.items()):
            status_text.text(f"מנתח שכבת זמן: {name}...")
            df = fetch_data(ASSETS[stock], tf)
            
            if df.empty or len(df) < 512:
                continue
                
            prices_full = df['close'].values
            ctx_prices = prices_full[-1024:] if len(prices_full) > 1024 else prices_full
            last_date = df.index[-1]
            last_price = ctx_prices[-1]
            
            try:
                forecast_res, _ = model.forecast([ctx_prices], freq=[0])
                
                # כאן אנחנו חותכים את התחזיות הארוכות כדי שכולן יתיישבו יפה על המסך
                if tf == "1d": 
                    draw_periods = 25  # מציג בערך חודש קדימה
                elif tf == "60m": 
                    draw_periods = 80  # מציג בערך 10 ימי מסחר קדימה
                else: 
                    draw_periods = 128 # מציג בערך 3-4 ימי מסחר קדימה
                
                fcst_prices = forecast_res[0][:draw_periods]
                fcst_dates = generate_israel_trading_dates(last_date, draw_periods, tf)
                
                conn_dates = [last_date] + list(fcst_dates)
                conn_fcst = [last_price] + list(fcst_prices)
                
                fig_mtf.add_trace(go.Scatter(x=conn_dates, y=conn_fcst, mode="lines", name=f"תחזית {name}", line=dict(color=color, width=2.5)))
                
            except Exception as e:
                pass
                
            progress_bar.progress((i + 1) / len(tfs))
            
        status_text.empty()
        progress_bar.empty()
        
        fig_mtf.update_layout(
            template="plotly_white", hovermode="x unified", title_x=0.5,
            title=f"תצוגה רב-שכבתית: הצטלבות מגמות ({stock})",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), 
            margin=dict(l=10, r=10, t=40, b=80) 
        )
        fig_mtf.update_xaxes(nticks=25, tickangle=-45, automargin=True)
        
        st.markdown("### 🧬 תרשים רב-שכבתי (Multi-Timeframe)")
        st.plotly_chart(fig_mtf, use_container_width=True)
        
        st.info("💡 הקו הכתום קוצר כדי שתוכל לעשות 'זום-אין' ולראות בבירור את התנועה העדינה של הגרף השעתי וה-15 דקות.")

    # מסלול 2: חיזוי רגיל או Backtesting יחיד
    else:
        df = fetch_data(ASSETS[stock], interval_choice)

        if df.empty or len(df) < 1200:
            st.error("❌ אין מספיק נתונים עבור הנכס הזה (דרושים לפחות 1200 תצפיות לעבודה תקינה). נסה רזולוציית זמן קצרה יותר.")
            st.stop()

        if interval_choice == "1d":
            unit = "ימי מסחר"
            test_cutoffs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 21, 63, 126]
            test_labels = ["חיזוי עתידי אמיתי (היום והלאה)"] + [f"{c} {unit} אחורה" for c in test_cutoffs[1:11]] + ["חודש (21 ימים) אחורה", "3 חודשים (63 ימים) אחורה", "חצי שנה (126 ימים) אחורה"]
        else:
            unit = "תקופות זמן"
            test_cutoffs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100]
            test_labels = ["חיזוי עתידי אמיתי (היום והלאה)"] + [f"{c} {unit} אחורה" for c in test_cutoffs[1:]]

        if mode == "בדיקה היסטורית (Backtest)":
            test_cutoffs = [cutoff]
            test_labels = [f"בדיקה ספציפית ({cutoff} תצפיות אחורה)"]

        st.session_state['test_cutoffs'] = test_cutoffs
        st.session_state['backtest_data'] = {}
        results_list = []

        prices_full = df['close'].values
        dates_full = df.index

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, (c, label) in enumerate(zip(test_cutoffs, test_labels)):
            status_text.text(f"מחשב מודל עבור: {label}...")
            
            if len(prices_full) - c >= 1024:
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

                    # שימוש בפונקציית התאריכים הישראלית החדשה
                    fcst_dates = generate_israel_trading_dates(last_date, 128, interval_choice)

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

                    if c > 0:
                        results_list.append({
                            "פעולה": "📊 הצג גרף",
                            "נקודת התחלה (בדיקת עבר)": label,
                            "סטייה ממוצעת מהמציאות (MAPE)": mape_str,
                            "זיהוי כיוון מגמה": trend_str,
                            "_c_val": c,
                            "_is_correct": is_correct
                        })

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

        if results_list or mode == "חיזוי עתידי רגיל":
            st.session_state['results_df'] = pd.DataFrame(results_list)
            st.session_state['run_done'] = True
            st.session_state['run_mode'] = mode

# =========================
# תצוגת התוצאות
# =========================
if st.session_state.get('run_done') and st.session_state.get('run_mode') != "חיזוי רב-שכבתי (Multi-Timeframe)":
    
    if st.session_state['run_mode'] == "חיזוי עתידי רגיל":
        st.markdown("### 📈 תחזית עתידית (מהיום והלאה)")
        future_data = st.session_state['backtest_data'][0]
        fig_future = create_forecast_figure(future_data)
        st.plotly_chart(fig_future, use_container_width=True)
        st.divider()
    elif st.session_state['run_mode'] == "בדיקה היסטורית (Backtest)":
        st.markdown("### 📈 בדיקה היסטורית בודדת")
        first_key = list(st.session_state['backtest_data'].keys())[0]
        single_test_data = st.session_state['backtest_data'][first_key]
        fig_single = create_forecast_figure(single_test_data)
        st.plotly_chart(fig_single, use_container_width=True)
        st.divider()

    df_res = st.session_state.get('results_df', pd.DataFrame())

    if not df_res.empty:
        correct_count = sum(1 for x in df_res['_is_correct'] if x == True)
        total_tests = sum(1 for x in df_res['_is_correct'] if x is not None)
        win_rate = (correct_count / total_tests) * 100 if total_tests > 0 else 0

        display_df = df_res.drop(columns=['_c_val', '_is_correct'])

        def style_trend(val):
            if "✅" in str(val): return 'color: #047857; font-weight: bold;'
            if "❌" in str(val): return 'color: #b91c1c;'
            return ''

        styled_df = display_df.style.map(style_trend, subset=["זיהוי כיוון מגמה"])

        if st.session_state['run_mode'] == "חיזוי עתידי רגיל":
            st.markdown("### 🔬 מבחני אמינות אוטומטיים למודל")
        else:
            st.markdown("### 🔬 תוצאות הבדיקה שהגדרת")

        st.info("💡 **הוראות:** לחץ על שורה בטבלה כדי לפתוח את הגרף שלה ולראות את החיזוי מול המציאות.")

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

        if total_tests > 1:
            if win_rate >= 60:
                st.success(f"🏆 **ציון אמינות כללי:** {win_rate:.0f}% הצלחה בזיהוי המגמה. (נחשב למודל יציב ואמין עבור הנכס הזה)")
            elif win_rate <= 40:
                st.error(f"⚠️ **ציון אמינות כללי:** {win_rate:.0f}% הצלחה בזיהוי המגמה. (המודל מתקשה לקרוא את הנכס הזה, לא מומלץ להסתמך עליו כאן)")
            else:
                st.warning(f"⚖️ **ציון אמינות כללי:** {win_rate:.0f}% הצלחה בזיהוי המגמה. (תוצאה בינונית - כדאי לשלב כלים נוספים בהחלטה)")

        with st.expander("❓ איך מחושבת 'הסטייה מהמציאות' (MAPE)?"):
            st.markdown("""
            **MAPE (Mean Absolute Percentage Error)** הוא מדד סטטיסטי שמראה בכמה אחוזים המודל "פספס" בממוצע.
            
            **דוגמה פשוטה:**
            אם המניה סגרה בפועל במחיר של **100 שקלים**, אבל המודל חזה שהיא תגיע ל-**105 שקלים**, הסטייה היא של **5%**.
            המדד לוקח את כל הסטיות היומיות לאורך התקופה שנבדקה, ומציג את הממוצע שלהן.
            
            * **סטייה נמוכה (למשל 1%-3%):** המודל היה מדויק מאוד וקרוב לקו המציאות.
            * **סטייה גבוהה (למשל מעל 10%):** המודל התקשה לחזות את התנודתיות, או שהתרחש אירוע בלתי צפוי בשוק.
            """)

st.divider()
st.markdown("""
<div style='text-align: center; color: #64748b; font-size: 0.85rem; padding-top: 1rem; padding-bottom: 2rem; direction: rtl;'>
    מודל החיזוי מופעל באמצעות Google TimesFM 1.0. האתר לצורכי מחקר, ועל אחריות המשתמש.<br>
    לשיתופי פעולה ניתן לפנות ליוצר במייל: <a href="mailto:147590@gmail.com" style="color: #3b82f6; text-decoration: none;" dir="ltr">147590@gmail.com</a>
</div>
""", unsafe_allow_html=True)
