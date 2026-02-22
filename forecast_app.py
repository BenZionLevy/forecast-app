import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tvDatafeed import TvDatafeed, Interval
import timesfm
import io

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

div[data-testid="stMarkdownContainer"], div[data-testid="stAlert"] {
    direction: rtl;
    text-align: right;
}

.stApp { background-color: #f4f6f9; }

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

.table-header {
    font-weight: bold;
    color: #475569;
    padding-bottom: 10px;
    border-bottom: 2px solid #cbd5e1;
    margin-bottom: 10px;
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
# נכסים לבחירה וקישורי Yahoo
# =========================
ASSETS = {
    "לאומי": ("LUMI", "TASE"), "פועלים": ("POLI", "TASE"), "דיסקונט": ("DSCT", "TASE"),
    "מזרחי טפחות": ("MZTF", "TASE"), "אלביט מערכות": ("ESLT", "TASE"), "טבע": ("TEVA", "TASE"),
    "נייס": ("NICE", "TASE"), "בזק": ("BEZQ", "TASE"), "דלק קבוצה": ("DLEKG", "TASE"),
    "מדד ת\"א 35": ("TA35", "TASE"), "S&P 500 ETF": ("SPY", "AMEX"), 
    'נאסד"ק 100 ETF': ("QQQ", "NASDAQ"), "USD/ILS (דולר-שקל)": ("USDILS", "FX_IDC")
}

YAHOO_LINKS = {
    "לאומי": "https://finance.yahoo.com/quote/LUMI.TA",
    "פועלים": "https://finance.yahoo.com/quote/POLI.TA",
    "דיסקונט": "https://finance.yahoo.com/quote/DSCT.TA",
    "מזרחי טפחות": "https://finance.yahoo.com/quote/MZTF.TA",
    "אלביט מערכות": "https://finance.yahoo.com/quote/ESLT.TA",
    "טבע": "https://finance.yahoo.com/quote/TEVA.TA",
    "נייס": "https://finance.yahoo.com/quote/NICE.TA",
    "בזק": "https://finance.yahoo.com/quote/BEZQ.TA",
    "דלק קבוצה": "https://finance.yahoo.com/quote/DLEKG.TA",
    "מדד ת\"א 35": "https://finance.yahoo.com/quote/^TA35",
    "S&P 500 ETF": "https://finance.yahoo.com/quote/SPY",
    'נאסד"ק 100 ETF': "https://finance.yahoo.com/quote/QQQ",
    "USD/ILS (דולר-שקל)": "https://finance.yahoo.com/quote/ILS=X"
}

# =========================
# הגדרות ממשק משתמש
# =========================
col1, col2 = st.columns(2)

with col1:
    stock = st.selectbox("בחר נכס פיננסי", list(ASSETS.keys()))

with col2:
    mode = st.radio(
        "סוג ניתוח",
        ["חיזוי רגיל (עתיד + מבחני עבר)", "חיזוי רב-שכבתי כפול (Multi-Timeframe)"],
        horizontal=False
    )

interval_choice = "1d"
calc_method = "שערים גולמיים"

if mode == "חיזוי רגיל (עתיד + מבחני עבר)":
    c_res, c_meth = st.columns(2)
    with c_res:
        int_map = {"5 דקות": "5m", "15 דקות": "15m", "30 דקות": "30m", "שעתי (60m)": "60m", "יומי (1d)": "1d", "שבועי (1W)": "1W"}
        resolution_label = st.selectbox("רזולוציית זמן:", list(int_map.keys()), index=4)
        interval_choice = int_map[resolution_label]
    with c_meth:
        calc_method = st.radio("שיטת חישוב:", ["שערים גולמיים", "תשואות באחוזים (מומלץ)"])
else:
    st.info("🧬 **מצב מחקר מתקדם:** המערכת תריץ במקביל גם שיטת שערים (קו רציף) וגם שיטת תשואות (קו מקווקו) על 3 רזולוציות זמן שונות באותו גרף.")

# =========================
# פונקציות ליבה (תאריכים, משיכה, וחיזוי)
# =========================
def generate_israel_trading_dates(start_date, periods, tf):
    dates = []
    curr = start_date
    if tf == "60m": step = pd.Timedelta(hours=1)
    elif tf == "30m": step = pd.Timedelta(minutes=30)
    elif tf == "15m": step = pd.Timedelta(minutes=15)
    elif tf == "5m": step = pd.Timedelta(minutes=5)
    elif tf == "1W": step = pd.Timedelta(weeks=1)
    else: step = pd.Timedelta(days=1)
    
    while len(dates) < periods:
        curr += step
        if tf == "1W":
            dates.append(curr)
            continue
        weekday = curr.weekday()
        if tf == "1d":
            if weekday in [0, 1, 2, 3, 4]: dates.append(curr)
        else:
            if weekday in [0, 1, 2, 3]:
                if 10 <= curr.hour < 17: dates.append(curr)
            elif weekday == 4:
                if 10 <= curr.hour < 14: dates.append(curr)
    return dates

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

def get_forecast(model, ctx_prices, method="שערים גולמיים", horizon=128):
    """
    מבצע חיזוי ומתמודד אוטומטית עם המרת תשואות למחירים במידת הצורך.
    """
    if "שערים" in method:
        forecast_res, quant_res = model.forecast([ctx_prices], freq=[0])
        return forecast_res[0][:horizon], quant_res[0, :horizon, 0], quant_res[0, :horizon, -1]
    else:
        # שיטת תשואות: חישוב אחוזי שינוי
        returns = np.diff(ctx_prices) / ctx_prices[:-1]
        # במקרים נדירים של חלוקה באפס או שגיאה, מחליפים ב-0
        returns = np.nan_to_num(returns)
        
        forecast_res, quant_res = model.forecast([returns], freq=[0])
        fcst_ret = forecast_res[0][:horizon]
        lower_ret = quant_res[0, :horizon, 0]
        upper_ret = quant_res[0, :horizon, -1]
        
        # שחזור התשואות בחזרה למחיר (ריבית דריבית)
        last_price = ctx_prices[-1]
        fcst_prices = last_price * np.cumprod(1 + fcst_ret)
        fcst_lower = last_price * np.cumprod(1 + lower_ret)
        fcst_upper = last_price * np.cumprod(1 + upper_ret)
        
        return fcst_prices, fcst_lower, fcst_upper

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
    fig.add_trace(go.Scatter(x=ctx_dates[-200:], y=ctx_prices[-200:], mode="lines", name="היסטוריה (בסיס)", line=dict(color='#2563eb', width=2)))
    fig.add_trace(go.Scatter(x=conn_dates, y=conn_upper, mode="lines", line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=conn_dates, y=conn_lower, mode="lines", fill="tonexty", fillcolor="rgba(245, 158, 11, 0.2)", line=dict(width=0), name="טווח הסתברות"))
    fig.add_trace(go.Scatter(x=conn_dates, y=conn_fcst, mode="lines", name="תחזית AI", line=dict(color='#f59e0b', width=2.5, dash="dash")))

    if c_val > 0:
        conn_act_dates = [last_date] + list(actual_dates)
        conn_act_prices = [last_price] + list(actual_prices)
        fig.add_trace(go.Scatter(x=conn_act_dates, y=conn_act_prices, mode="lines", name="מציאות בפועל", line=dict(color='#10b981', width=3)))
        fig.add_vline(x=str(last_date), line_width=2, line_dash="dot", line_color="#94a3b8")
        fig.add_annotation(x=str(last_date), y=1.05, yref="paper", text="נקודת עיוורון", showarrow=False, font=dict(color="#94a3b8", size=12), xanchor="center")

    fig.update_layout(template="plotly_white", hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), margin=dict(l=10, r=10, t=40, b=80))
    fig.update_xaxes(nticks=25, tickangle=-45, automargin=True)
    return fig

@st.dialog("📊 גרף מפורט - חיזוי מול מציאות", width="large")
def show_chart_dialog(c_idx):
    data = st.session_state['backtest_data'][c_idx]
    fig = create_forecast_figure(data)
    st.plotly_chart(fig, use_container_width=True)

def generate_excel(data_dict, stock_name):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        link_df = pd.DataFrame({"נכס פיננסי": [stock_name], "קישור לאימות (Yahoo Finance)": [YAHOO_LINKS.get(stock_name, "אין נתון")]})
        link_df.to_excel(writer, index=False, sheet_name="מידע וקישורים")
        
        for sheet_name, df in data_dict.items():
            export_df = df.copy()
            export_df.reset_index(inplace=True)
            export_df.columns = ["תאריך ושעה", "שער סגירה"]
            export_df.to_excel(writer, index=False, sheet_name=sheet_name)
    return output.getvalue()

# =========================
# הפעלת הלולאה והחישובים
# =========================
if st.button("🚀 הפעל ניתוח AI מקיף", type="primary", use_container_width=True):

    with st.spinner("טוען מודל ומושך נתונים מ-TradingView..."):
        model = load_model()
    
    st.session_state['selected_stock'] = stock
    st.session_state['raw_data_export'] = {}
        
    if mode == "חיזוי רב-שכבתי כפול (Multi-Timeframe)":
        tfs = {"1d": ("יומי", "#f59e0b"), "60m": ("שעתי", "#8b5cf6"), "15m": ("15 דקות", "#ef4444")}
        methods = ["שערים", "תשואות"]
        
        fig_mtf = go.Figure()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        bg_df = fetch_data(ASSETS[stock], "60m")
        if not bg_df.empty:
            fig_mtf.add_trace(go.Scatter(x=bg_df.index[-150:], y=bg_df['close'].tail(150), mode="lines", name="היסטוריה קרובה (שעתי)", line=dict(color='#cbd5e1', width=1.5)))

        total_steps = len(tfs) * len(methods)
        current_step = 0

        for tf, (name, color) in tfs.items():
            df = fetch_data(ASSETS[stock], tf)
            if df.empty or len(df) < 512: 
                current_step += 2
                continue
            
            st.session_state['raw_data_export'][f"נתוני_{name}"] = df
            prices_full = df['close'].values
            ctx_prices = prices_full[-1024:] if len(prices_full) > 1024 else prices_full
            last_date = df.index[-1]
            last_price = ctx_prices[-1]
            
            if tf == "1d": draw_periods = 25
            elif tf == "60m": draw_periods = 80
            else: draw_periods = 128
            
            fcst_dates = generate_israel_trading_dates(last_date, draw_periods, tf)
            conn_dates = [last_date] + list(fcst_dates)
            
            for meth in methods:
                status_text.text(f"מנתח שכבת זמן: {name} | שיטה: {meth}...")
                try:
                    fcst_prices, _, _ = get_forecast(model, ctx_prices, method=meth, horizon=draw_periods)
                    conn_fcst = [last_price] + list(fcst_prices)
                    
                    dash_style = "solid" if meth == "שערים" else "dot"
                    opac = 1.0 if meth == "שערים" else 0.7
                    
                    fig_mtf.add_trace(go.Scatter(
                        x=conn_dates, y=conn_fcst, mode="lines", 
                        name=f"תחזית {name} ({meth})", 
                        line=dict(color=color, width=2.5, dash=dash_style),
                        opacity=opac
                    ))
                except Exception as e: pass
                
                current_step += 1
                progress_bar.progress(current_step / total_steps)
            
        status_text.empty()
        progress_bar.empty()
        
        fig_mtf.update_layout(
            template="plotly_white", hovermode="x unified", title_x=0.5,
            title=f"תצוגה רב-שכבתית כפולה: מבוסס שערים ומבוסס תשואות ({stock})",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), 
            margin=dict(l=10, r=10, t=40, b=80) 
        )
        fig_mtf.update_xaxes(nticks=25, tickangle=-45, automargin=True)
        
        st.markdown("### 🧬 תרשים רב-שכבתי כפול (Multi-Timeframe)")
        st.plotly_chart(fig_mtf, use_container_width=True)
        st.session_state['run_done'] = True
        st.session_state['run_mode'] = mode

    else:
        df = fetch_data(ASSETS[stock], interval_choice)
        
        if df.empty or len(df) < 1200:
            st.error("❌ אין מספיק נתונים עבור הנכס הזה. נסה רזולוציית זמן קצרה יותר.")
            st.stop()
            
        st.session_state['raw_data_export']["נתונים_גולמיים"] = df

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
            status_text.text(f"מחשב מודל (שיטה: {calc_method}) עבור: {label}...")
            
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
                    # שימוש בפונקציה החכמה החדשה לחיזוי
                    fcst_prices, fcst_lower, fcst_upper = get_forecast(model, ctx_prices, method=calc_method, horizon=128)
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
                            "label": label,
                            "mape": mape_str,
                            "trend": trend_str,
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
                except Exception as e: pass 
                    
            progress_bar.progress((i + 1) / len(test_cutoffs))

        status_text.empty()
        progress_bar.empty()

        if results_list or mode == "חיזוי עתידי רגיל":
            st.session_state['results_df'] = pd.DataFrame(results_list)
            st.session_state['run_done'] = True
            st.session_state['run_mode'] = mode

# =========================
# תצוגת התוצאות (לחיזוי רגיל בלבד)
# =========================
if st.session_state.get('run_done') and st.session_state.get('run_mode') == "חיזוי רגיל (עתיד + מבחני עבר)":
    
    st.markdown("### 📈 תחזית עתידית (מהיום והלאה)")
    future_data = st.session_state['backtest_data'][0]
    fig_future = create_forecast_figure(future_data)
    st.plotly_chart(fig_future, use_container_width=True)
    
    st.divider()
    
    df_res = st.session_state.get('results_df', pd.DataFrame())

    if not df_res.empty:
        correct_count = sum(1 for x in df_res['_is_correct'] if x == True)
        total_tests = sum(1 for x in df_res['_is_correct'] if x is not None)
        win_rate = (correct_count / total_tests) * 100 if total_tests > 0 else 0

        st.markdown("### 🔬 מבחני אמינות אוטומטיים למודל")
        st.info("💡 המערכת חזרה אחורה בזמן ובדקה אם התחזיות שלה אכן התממשו במציאות. **לחץ על לחצן 'הצג' בכל שורה כדי לראות את הגרף!**")

        col_h1, col_h2, col_h3, col_h4 = st.columns([2, 2, 2, 1])
        col_h1.markdown("<div class='table-header'>נקודת התחלה (בדיקת עבר)</div>", unsafe_allow_html=True)
        col_h2.markdown("<div class='table-header'>סטייה מהמציאות (MAPE)</div>", unsafe_allow_html=True)
        col_h3.markdown("<div class='table-header'>זיהוי כיוון מגמה</div>", unsafe_allow_html=True)
        col_h4.markdown("<div class='table-header'>פעולה</div>", unsafe_allow_html=True)
        
        for index, row in df_res.iterrows():
            c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
            c1.write(row['label'])
            c2.write(row['mape'])
            
            trend = row['trend']
            if "✅" in trend: c3.markdown(f"<span style='color: #047857; font-weight: bold;'>{trend}</span>", unsafe_allow_html=True)
            else: c3.markdown(f"<span style='color: #b91c1c; font-weight: bold;'>{trend}</span>", unsafe_allow_html=True)
            
            if c4.button("📊 הצג", key=f"btn_show_{row['_c_val']}"):
                show_chart_dialog(row['_c_val'])
                
            st.markdown("<hr style='margin: 0.2rem 0; opacity: 0.2;'>", unsafe_allow_html=True)

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
            """)

# =========================
# כפתור הורדת אקסל (מופיע בסוף כל הרצה)
# =========================
if st.session_state.get('run_done'):
    st.divider()
    st.markdown("### 📥 בדיקת נתונים גולמיים")
    st.info("כדי להבטיח שקיפות מלאה, באפשרותך להוריד את קובץ הנתונים הגולמי שעליו התבסס המודל לאימות מול מקורות חיצוניים. הקובץ כולל גיליון עם קישור ישיר למניה באתר Yahoo Finance.")
    
    excel_file = generate_excel(st.session_state['raw_data_export'], st.session_state['selected_stock'])
    st.download_button(
        label="💾 הורד קובץ נתונים (Excel)",
        data=excel_file,
        file_name=f"{st.session_state['selected_stock']}_RawData.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

st.divider()
st.markdown("""
<div style='text-align: center; color: #64748b; font-size: 0.85rem; padding-top: 1rem; padding-bottom: 2rem; direction: rtl;'>
    מודל החיזוי מופעל באמצעות Google TimesFM 1.0. האתר לצורכי מחקר, ועל אחריות המשתמש.<br>
    לשיתופי פעולה ניתן לפנות ליוצר במייל: <a href="mailto:147590@gmail.com" style="color: #3b82f6; text-decoration: none;" dir="ltr">147590@gmail.com</a>
</div>
""", unsafe_allow_html=True)
