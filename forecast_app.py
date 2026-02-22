import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time

# ייבוא ספריות החיזוי והנתונים
from tvDatafeed import TvDatafeed, Interval
import timesfm

st.set_page_config(page_title="מעבדת חיזוי AI", layout="wide", page_icon="🤖")

# ==========================================
# עיצוב CSS מותאם אישית
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rubik:wght@300;400;500;600;700&display=swap');
    .stApp { background: #f8fafc; }
    html, body, [class*="css"] { font-family: 'Rubik', sans-serif; direction: rtl; }
    .main-header {
        text-align: center; padding: 2rem 0 0.5rem 0; font-size: 2.8rem; font-weight: 800;
        background: linear-gradient(135deg, #4f46e5 0%, #ec4899 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin-bottom: 0;
    }
    .sub-header { text-align: center; color: #64748b; font-size: 1.1rem; font-weight: 400; margin-bottom: 2.5rem; }
    .section-title {
        font-size: 1.35rem; font-weight: 700; color: #1e293b; margin-top: 1rem; margin-bottom: 1.5rem;
        text-align: right; direction: rtl; display: flex; align-items: center; gap: 0.5rem;
    }
    .section-title::after { content: ""; flex: 1; height: 2px; background: #e2e8f0; margin-right: 15px; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>🤖 מעבדת חיזוי מניות: TimesFM</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>מודל בינה מלאכותית מבית Google לחקר מגמות וביצוע Backtesting בנכסים פיננסיים</p>", unsafe_allow_html=True)

# ==========================================
# טעינת מודל ה-AI (מעודכן ל-API החדש של גוגל)
# ==========================================
@st.cache_resource(show_spinner=False)
def load_ai_model():
    tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="cpu",
            per_core_batch_size=1, # מותאם לשרת חלש
            horizon_len=128,
            context_len=512,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
        ),
    )
    return tfm

# ==========================================
# משיכת נתונים
# ==========================================
@st.cache_data(ttl=600, show_spinner=False)
def fetch_data_tv(sym_tuple, interval_str):
    try:
        tv = TvDatafeed()
        tv_intervals = {"1d": Interval.in_daily, "60m": Interval.in_1_hour, "15m": Interval.in_15_minute}
        inter = tv_intervals.get(interval_str, Interval.in_daily)
        
        # מושכים 1500 נרות כדי שיהיה לנו מספיק גם ל-Backtesting עמוק וגם ל-512 נרות חובה
        df = tv.get_hist(symbol=sym_tuple[0], exchange=sym_tuple[1], interval=inter, n_bars=1500)
        
        if df is None or df.empty: return pd.DataFrame()
        
        if df.index.tz is None: 
            df.index = df.index.tz_localize("UTC").tz_convert("Asia/Jerusalem")
        else: 
            df.index = df.index.tz_convert("Asia/Jerusalem")
            
        return df[['close']]
    except Exception as e:
        return pd.DataFrame()

# ==========================================
# הגדרות משתמש
# ==========================================
DEFAULT_TICKERS = {
    "לאומי": ("LUMI", "TASE"), 
    "פועלים": ("POLI", "TASE"), 
    "מדד ת\"א 35": ("TA35", "TASE"), 
    "S&P 500 ETF": ("SPY", "AMEX"), 
    'נאסד"ק 100 ETF': ("QQQ", "NASDAQ"), 
    "USD/ILS": ("USDILS", "FX_IDC")
}

st.markdown("<div class='section-title'>⚙️ שלב 1: הגדרות מודל ונקודת זמן</div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    ticker_names = list(DEFAULT_TICKERS.keys())
    asset_name = st.selectbox("בחר נכס:", ticker_names, index=0)
    target_tuple = DEFAULT_TICKERS[asset_name]

with col2:
    int_map = {"יומי (1d)": "1d", "שעתי (60m)": "60m"}
    interval_choice = int_map[st.selectbox("רזולוציית זמן:", list(int_map.keys()), index=0)]

with col3:
    # הגדרות Backtesting
    backtest_options = {
        "ללא (חיזוי להיום אל תוך העתיד)": 0, 
        "לפני שבוע (5 נרות)": 5, 
        "לפני חודש (21 נרות)": 21, 
        "לפני 3 חודשים (63 נרות)": 63, 
        "לפני חצי שנה (126 נרות)": 126
    }
    backtest_choice = st.selectbox("בדיקת אמינות לאחור (Backtesting):", list(backtest_options.keys()), index=0)
    cutoff_bars = backtest_options[backtest_choice]
    
    # התאמה בסיסית אם בחרנו שעתי (נכפיל ב-8 שעות מסחר בערך)
    if interval_choice == "60m" and cutoff_bars > 0:
        cutoff_bars = cutoff_bars * 8

# ==========================================
# ביצוע החיזוי
# ==========================================
st.divider()

if st.button("🔮 הפעל מודל חיזוי AI עכשיו", type="primary", use_container_width=True):
    
    with st.spinner("טוען את מודל הבינה המלאכותית (TimesFM)... (זה עשוי לקחת מספר דקות בפעם הראשונה)"):
        try:
            tfm_model = load_ai_model()
        except Exception as e:
            st.error(f"שגיאה בטעינת המודל. ייתכן שהשרת עמוס מדי או חסר זיכרון: {e}")
            st.stop()
            
    with st.spinner(f"מושך נתונים היסטוריים עבור {asset_name}..."):
        df_hist = fetch_data_tv(target_tuple, interval_choice)
        
    if df_hist.empty or (len(df_hist) - cutoff_bars) < 512:
        st.error(f"❌ אין מספיק נתונים. המודל דורש 512 נרות היסטוריים מעבר לנקודת החיתוך שבחרת.")
        st.stop()
        
    # פיצול הנתונים לפי בחירת ה-Backtesting
    if cutoff_bars > 0:
        df_train = df_hist.iloc[:-cutoff_bars]  # הנתונים שהמודל "רואה"
        df_actual = df_hist.iloc[-cutoff_bars:] # מה שבאמת קרה ונסתיר מהמודל
    else:
        df_train = df_hist
        df_actual = pd.DataFrame()

    with st.spinner("המודל מנתח את התבניות ההיסטוריות ובונה תחזית (כולל טווח סביר)..."):
        prices_array = df_train['close'].values
        
        try:
            # הוספנו את הגדרת ה-freq כפי שדורשת הגרסה החדשה
            forecast_results, quantiles_results = tfm_model.forecast([prices_array], freq=[0])
            
            future_prices = forecast_results[0] 
            lower_bound = quantiles_results[0, :, 0]
            upper_bound = quantiles_results[0, :, -1]
            
        except Exception as e:
            st.error(f"שגיאה בתהליך החיזוי. ייתכן קריסת זיכרון (OOM): {e}")
            st.stop()
            
    # יצירת ציר זמן עתידי
    last_train_date = df_train.index[-1]
    last_train_price = df_train['close'].iloc[-1]
    
    if interval_choice == "1d":
        forecast_dates = pd.bdate_range(start=last_train_date + pd.Timedelta(days=1), periods=128)
    else:
        forecast_dates = pd.date_range(start=last_train_date + pd.Timedelta(hours=1), periods=128, freq='H')
        
    forecast_df = pd.DataFrame({
        "Date": forecast_dates, 
        "Forecast": future_prices,
        "Lower": lower_bound,
        "Upper": upper_bound
    })
    forecast_df.set_index("Date", inplace=True)

    # ==========================================
    # ציור הגרף המשולב
    # ==========================================
    st.markdown("<div class='section-title'>📈 תוצאות החיזוי ומבחן מציאות</div>", unsafe_allow_html=True)
    
    fig = go.Figure()
    
    # הצגת ההיסטוריה שהמודל למד (נציג רק 200 אחרונים כדי שיהיה נוח בעין)
    display_hist = df_train.tail(200)
    
    # חיבור הקווים
    connect_dates = [last_train_date] + list(forecast_df.index)
    connect_prices = [last_train_price] + list(forecast_df['Forecast'])
    connect_lower = [last_train_price] + list(forecast_df['Lower'])
    connect_upper = [last_train_price] + list(forecast_df['Upper'])
    
    # טווח סביר עליון ותחתון (ענן)
    fig.add_trace(go.Scatter(
        x=connect_dates, y=connect_upper, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=connect_dates, y=connect_lower, mode='lines', fill='tonexty', fillcolor='rgba(245, 158, 11, 0.2)', line=dict(width=0), name='טווח סביר (AI)'
    ))
    
    # קו התחזית (כתום מקווקו)
    fig.add_trace(go.Scatter(
        x=connect_dates, y=connect_prices, mode='lines', name='תחזית AI', line=dict(color='#f59e0b', width=2.5, dash='dash')
    ))

    # קו ההיסטוריה (כחול)
    fig.add_trace(go.Scatter(
        x=display_hist.index, y=display_hist['close'], mode='lines', name='היסטוריה (בסיס לחיזוי)', line=dict(color='#2563eb', width=2)
    ))

    # הקו המציאותי (ירוק זוהר)
    if not df_actual.empty:
        actual_dates = [last_train_date] + list(df_actual.index)
        actual_prices = [last_train_price] + list(df_actual['close'])
        
        fig.add_trace(go.Scatter(
            x=actual_dates, y=actual_prices, mode='lines', name='מה קרה בפועל? (המציאות)', line=dict(color='#10b981', width=3)
        ))
        
        fig.add_vline(x=last_train_date.isoformat(), line_width=2, line_dash="dot", line_color="#94a3b8", annotation_text="נקודת החיתוך (כאן המודל עוור)", annotation_position="top left")
    
    title_text = f"חיזוי מסלול מחיר: {asset_name}"
    if cutoff_bars > 0:
        title_text += f" (בדיקה לאחור - חזרנו בזמן {cutoff_bars} נרות)"
        
    fig.update_layout(
        title=title_text,
        title_x=0.5,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    if cutoff_bars > 0:
        st.info("🔍 **איך קוראים את מבחן המציאות?** הקו הכחול מסתיים בנקודת הזמן שבחרנו 'לחזור' אליה. משם, המודל צייר את הקו הכתום (התחזית), והמציאות ציירה את הקו הירוק. עד כמה הם קרובים? (שים לב שקו ירוק שנשאר בתוך 'הענן' הכתום נחשב לתחזית מוצלחת סטטיסטית).")
    else:
        st.caption("⚠️ השטח הכתום המקווקו מייצג את הטווח הסביר שבו המניה צפויה לנוע. החיזוי אינו מתחשב בחדשות או נתוני מאקרו ואינו מהווה המלצה.")

st.divider()
st.markdown("""
<div style='text-align: center; color: #64748b; font-size: 0.85rem; padding-top: 1rem; padding-bottom: 2rem; direction: rtl;'>
    מודל החיזוי מופעל באמצעות Google TimesFM 1.0. האתר לצורכי מחקר, ועל אחריות המשתמש.<br>
    לשיתופי פעולה ניתן לפנות ליוצר במייל: <a href="mailto:147590@gmail.com" style="color: #3b82f6; text-decoration: none;" dir="ltr">147590@gmail.com</a>
</div>
""", unsafe_allow_html=True)
