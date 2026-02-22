import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tvDatafeed import TvDatafeed, Interval
import timesfm
import io

st.set_page_config(
    page_title="מעבדת מאקרו - AI",
    layout="wide",
    page_icon="🔬"
)

# =========================
# עיצוב מותאם
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Assistant:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Assistant', sans-serif; direction: rtl; text-align: right; }
div[data-testid="stMarkdownContainer"], div[data-testid="stAlert"] { direction: rtl; text-align: right; }
.stApp { background-color: #f0f4f8; }
.main-title { text-align: right; font-size: 2.4rem; font-weight: 800; margin-bottom: 0.2rem; color: #1e293b; }
.sub-title { text-align: right; font-size: 1.1rem; color: #475569; margin-bottom: 1.5rem; }
.heavy-warning { background: #fee2e2; border: 1px solid #fca5a5; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; color: #991b1b; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🔬 מעבדת מאקרו וחיזוי כמותי (TimesFM)</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>מערכת מחקר כבדה: ניתוח מקביל של נכסי בסיס והשפעות מאקרו-כלכליות</div>", unsafe_allow_html=True)

st.markdown("""
<div class="heavy-warning">
⚡ זהירות: מצב מחקר מתקדם מופעל. המערכת מעבדת במקביל אלפי נתונים על מספר נכסים ומותחת את גבולות הזיכרון (RAM) של השרת. ייתכנו קריסות או זמני טעינה ארוכים.
</div>
""", unsafe_allow_html=True)

# =========================
# מודל AI מורחב (2048 זיכרון)
# =========================
@st.cache_resource(show_spinner=False)
def load_model():
    return timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="cpu",
            per_core_batch_size=3, # מריץ 3 נכסים במקביל
            horizon_len=128,
            context_len=2048, # הקשר היסטורי כפול!
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
        ),
    )

ASSETS = {
    "לאומי": ("LUMI", "TASE"), "פועלים": ("POLI", "TASE"), "דיסקונט": ("DSCT", "TASE"),
    "מזרחי טפחות": ("MZTF", "TASE"), "אלביט מערכות": ("ESLT", "TASE"), "טבע": ("TEVA", "TASE"),
    "נייס": ("NICE", "TASE"), "בזק": ("BEZQ", "TASE"), "דלק קבוצה": ("DLEKG", "TASE")
}

MACRO_ASSETS = {
    "S&P 500 ETF": ("SPY", "AMEX"), 
    "USD/ILS (דולר-שקל)": ("USDILS", "FX_IDC")
}

YAHOO_LINKS = {
    "לאומי": "https://finance.yahoo.com/quote/LUMI.TA", "פועלים": "https://finance.yahoo.com/quote/POLI.TA",
    "S&P 500 ETF": "https://finance.yahoo.com/quote/SPY", "USD/ILS (דולר-שקל)": "https://finance.yahoo.com/quote/ILS=X"
}

st.markdown("### ⚙️ הגדרות ניתוח")
col1, col2 = st.columns(2)
with col1: stock = st.selectbox("בחר מניית מטרה (Target):", list(ASSETS.keys()))
with col2:
    int_map = {"15 דקות": "15m", "שעתי (60m)": "60m", "יומי (1d)": "1d", "שבועי (1W)": "1W"}
    interval_choice = int_map[st.selectbox("רזולוציית זמן:", list(int_map.keys()), index=2)]

# =========================
# פונקציות תאריכים ומשיכה
# =========================
def generate_dates(start_date, periods, tf):
    dates, curr = [], start_date
    step = pd.Timedelta(hours=1) if tf=="60m" else pd.Timedelta(minutes=15) if tf=="15m" else pd.Timedelta(days=1)
    if tf == "1W": step = pd.Timedelta(weeks=1)
    
    while len(dates) < periods:
        curr += step
        if tf == "1W":
            dates.append(curr); continue
        weekday = curr.weekday()
        if tf == "1d" and weekday < 5: dates.append(curr)
        elif tf in ["60m", "15m"]:
            if weekday < 4 and 10 <= curr.hour < 17: dates.append(curr)
            elif weekday == 4 and 10 <= curr.hour < 14: dates.append(curr)
    return dates

@st.cache_data(ttl=600, show_spinner=False)
def fetch_data(symbol, interval_str):
    tv = TvDatafeed()
    inter = Interval.in_15_minute if interval_str=="15m" else Interval.in_1_hour if interval_str=="60m" else Interval.in_daily if interval_str=="1d" else Interval.in_weekly
    df = tv.get_hist(symbol=symbol[0], exchange=symbol[1], interval=inter, n_bars=4500)
    if df is None or df.empty: return pd.DataFrame()
    df.index = df.index.tz_convert("Asia/Jerusalem") if df.index.tz else df.index.tz_localize("UTC").tz_convert("Asia/Jerusalem")
    df.index = df.index.tz_localize(None) 
    return df[['close']]

def generate_macro_excel(target_name, dfs_dict):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        links = pd.DataFrame({"נכס": list(dfs_dict.keys()), "קישור Yahoo": [YAHOO_LINKS.get(k, "N/A") for k in dfs_dict.keys()]})
        links.to_excel(writer, index=False, sheet_name="מקורות ואימות")
        
        for name, df in dfs_dict.items():
            exp_df = df.copy().reset_index()
            exp_df.columns = ["תאריך ושעה", "שער סגירה"]
            exp_df.to_excel(writer, index=False, sheet_name=name[:30]) # מוגבל 30 תווים לשם גיליון
    return output.getvalue()

# =========================
# הרצת ה"מפלצת"
# =========================
st.divider()
if st.button("🚀 הפעל סימולציית מאקרו כבדה", type="primary", use_container_width=True):
    
    with st.spinner("מוזג נתונים: מושך מניית מטרה + נתוני S&P 500 + נתוני מטח במקביל..."):
        model = load_model()
        
        # משיכת נתונים משולבת
        df_target = fetch_data(ASSETS[stock], interval_choice)
        df_spy = fetch_data(MACRO_ASSETS["S&P 500 ETF"], interval_choice)
        df_usd = fetch_data(MACRO_ASSETS["USD/ILS (דולר-שקל)"], interval_choice)
        
        if df_target.empty or len(df_target) < 1500:
            st.error("❌ חסרים נתונים לנכס המטרה למודל הכבד הזה.")
            st.stop()
            
        st.session_state['dfs_dict'] = {stock: df_target, "S&P 500": df_spy, "USD-ILS": df_usd}

    with st.spinner("AI עובד: מריץ חיזוי מתקדם על כלל הנכסים באצווה אחת (Batch Processing)..."):
        
        # הכנת מערכים למודל
        t_vals = df_target['close'].values[-2048:]
        s_vals = df_spy['close'].values[-2048:] if not df_spy.empty else t_vals
        u_vals = df_usd['close'].values[-2048:] if not df_usd.empty else t_vals
        
        last_date = df_target.index[-1]
        
        try:
            # הפעלת המודל על מערך של נכסים יחד!
            forecasts, quants = model.forecast([t_vals, s_vals, u_vals], freq=[0, 0, 0])
            
            # חילוץ התוצאות לכל נכס
            fcst_target = forecasts[0]
            fcst_spy = forecasts[1]
            fcst_usd = forecasts[2]
            
            fcst_dates = generate_dates(last_date, 128, interval_choice)
            conn_dates = [last_date] + list(fcst_dates)
            
        except Exception as e:
            st.error(f"🚨 השרת קרס מעומס זיכרון (OOM) או שגיאת חישוב! השגיאה המדויקת: {e}")
            st.stop()

    with st.spinner("מנרמל תצוגה (אחוזי שינוי) ומצייר מפת קורלציה עתידית..."):
        # כדי להציג את כולם יחד, נהפוך את המחירים לאחוזי שינוי מנקודת ההווה (האפס)
        t_base = t_vals[-1]
        s_base = s_vals[-1]
        u_base = u_vals[-1]
        
        norm_t_hist = ((t_vals[-200:] - t_vals[-200]) / t_vals[-200]) * 100
        norm_t_fcst = ((np.insert(fcst_target, 0, t_base) - t_base) / t_base) * 100
        
        norm_s_fcst = ((np.insert(fcst_spy, 0, s_base) - s_base) / s_base) * 100
        norm_u_fcst = ((np.insert(fcst_usd, 0, u_base) - u_base) / u_base) * 100
        
        hist_dates = df_target.index[-200:]
        
        fig = go.Figure()
        
        # היסטוריית המניה (מנורמלת)
        fig.add_trace(go.Scatter(x=hist_dates, y=norm_t_hist, mode="lines", name=f"{stock} (היסטוריה)", line=dict(color='#94a3b8', width=2)))
        
        # תחזיות עתיד מנורמלות!
        fig.add_trace(go.Scatter(x=conn_dates, y=norm_t_fcst, mode="lines", name=f"תחזית {stock}", line=dict(color='#2563eb', width=3.5)))
        fig.add_trace(go.Scatter(x=conn_dates, y=norm_s_fcst, mode="lines", name="תחזית S&P 500", line=dict(color='#10b981', width=2.5, dash='dash')))
        fig.add_trace(go.Scatter(x=conn_dates, y=norm_u_fcst, mode="lines", name="תחזית דולר-שקל", line=dict(color='#f59e0b', width=2.5, dash='dot')))
        
        fig.add_vline(x=str(last_date), line_width=2, line_dash="solid", line_color="#475569")
        fig.add_annotation(x=str(last_date), y=0, text="כאן מתחיל העתיד", showarrow=False, xanchor="right", yanchor="bottom", textangle=-90)

        fig.update_layout(
            title=f"מפת קורלציה עתידית: לאן השוק הולך? (תצוגה מנורמלת באחוזים %)",
            title_x=0.5, template="plotly_white", hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), 
            margin=dict(l=10, r=10, t=40, b=80),
            yaxis_title="שינוי באחוזים (%)"
        )
        fig.update_xaxes(nticks=25, tickangle=-45, automargin=True)
        
        st.markdown("### 🌐 סימולציית מאקרו מקבילה")
        st.info("💡 **איך קוראים את הגרף?** כל הנכסים אופסו ל-0% בנקודת הזמן של היום. כעת ניתן לראות האם ה-AI צופה שהמניה תעלה בזמן שהדולר יורד, ואיך היא מתנהגת ביחס לשוק האמריקאי הכללי (S&P 500).")
        st.plotly_chart(fig, use_container_width=True)
        
        st.session_state['run_done'] = True
        st.session_state['target_stock'] = stock

if st.session_state.get('run_done'):
    st.divider()
    st.markdown("### 📥 ייצוא נתוני מעבדה")
    st.info("מכיוון שהרצנו מודל כבד ששואב מספר נכסים מקבילים, קובץ האקסל שתוריד כעת מכיל גיליונות נפרדים לכל נכס. תוכל להשתמש בו כדי לחשב קורלציות (Correlation) או מדדי אלפא/ביתא במודלים העצמאיים שלך.")
    
    excel_file = generate_macro_excel(st.session_state['target_stock'], st.session_state['dfs_dict'])
    st.download_button(
        label="💾 הורד קובץ נתוני מאקרו מלא (Excel)",
        data=excel_file,
        file_name=f"Macro_Lab_{st.session_state['target_stock']}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

st.divider()
st.markdown("""
<div style='text-align: center; color: #64748b; font-size: 0.85rem; padding-top: 1rem; padding-bottom: 2rem; direction: rtl;'>
    מודל החיזוי מופעל באמצעות Google TimesFM 1.0 בקיבולת מקסימלית. האתר לצורכי מחקר, ועל אחריות המשתמש.<br>
    לשיתופי פעולה ניתן לפנות ליוצר במייל: <a href="mailto:147590@gmail.com" style="color: #3b82f6; text-decoration: none;" dir="ltr">147590@gmail.com</a>
</div>
""", unsafe_allow_html=True)
