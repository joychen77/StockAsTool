import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings('ignore')

st.set_page_config(page_title="股票歷史波段分析工具", page_icon="📈", layout="wide")

st.title("📈 股票歷史波段回撤與漲幅分析系統")

# 側邊欄：使用者互動控制項
st.sidebar.header("⚙️ 參數設定")
symbols_input = st.sidebar.text_input("輸入股票代碼 (多檔以逗號隔開)", value="0050.TW, 00631L.TW").strip().upper()
start_year = st.sidebar.slider("歷史資料起始年份", min_value=2008, max_value=2024, value=2012)
pull_back_pct = st.sidebar.slider("波段結算拉回門檻 (%)", min_value=2.0, max_value=15.0, value=5.0, step=0.5)
pull_back_limit = pull_back_pct / 100.0

@st.cache_data(ttl=3600)
def fetch_multi_data(symbols_list, start_date):
    downloaded = {}
    for symbol in symbols_list:
        df = yf.download(symbol, start=start_date, auto_adjust=False, progress=False)
        if not df.empty:
            series = df['Close'][symbol] if isinstance(df.columns, pd.MultiIndex) else df['Close']
            series = series.dropna()
            
            # 清洗壞點
            returns = series.pct_change().abs()
            clean_series = series.copy()
            clean_series[returns > 0.20] = np.nan
            clean_series = clean_series.ffill().bfill()
            
            # 切除開局壞點
            med_price = clean_series.median()
            if clean_series.iloc[:30].median() > med_price * 2.0:
                valid_mask = clean_series <= (med_price * 2.0)
                first_valid_idx = valid_mask.idxmax() if valid_mask.any() else None
                if first_valid_idx:
                    clean_series = clean_series.loc[first_valid_idx:]
            downloaded[symbol] = clean_series
    return pd.DataFrame(downloaded)

def analyze_drawdown_buckets(df_prices):
    bucket_results = {}
    for name in df_prices.columns:
        series = df_prices[name].dropna()
        cummax = series.cummax()
        drawdown = (series - cummax) / cummax
        total_days = len(drawdown)

        buckets = [
            round((drawdown == 0).sum() / total_days * 100, 2),
            round(((drawdown < 0) & (drawdown > -0.05)).sum() / total_days * 100, 2),
            round(((drawdown <= -0.05) & (drawdown > -0.10)).sum() / total_days * 100, 2),
            round(((drawdown <= -0.10) & (drawdown > -0.15)).sum() / total_days * 100, 2),
            round(((drawdown <= -0.15) & (drawdown > -0.20)).sum() / total_days * 100, 2),
            round(((drawdown <= -0.20) & (drawdown > -0.25)).sum() / total_days * 100, 2),
            round(((drawdown <= -0.25) & (drawdown > -0.30)).sum() / total_days * 100, 2),
            round(((drawdown <= -0.30) & (drawdown > -0.35)).sum() / total_days * 100, 2),
            round(((drawdown <= -0.35) & (drawdown > -0.40)).sum() / total_days * 100, 2),
            round((drawdown <= -0.40).sum() / total_days * 100, 2)
        ]
        bucket_results[name] = buckets

    cols = ['創新高(0%)', '0%~-5%', '-5%~-10%', '-10%~-15%', '-15%~-20%',
            '-20%~-25%', '-25%~-30%', '-30%~-35%', '-35%~-40%', '<-40%']
    return pd.DataFrame(bucket_results, index=cols).T

def analyze_rally_buckets(df_prices, limit):
    bucket_results, stats_results = {}, {}
    for name in df_prices.columns:
        series = df_prices[name].dropna()
        prices = series.values
        swing_gains = []
        if len(prices) < 2: continue

        in_swing, p_low, p_peak = False, prices[0], prices[0]
        for p in prices:
            if not in_swing:
                p_low, p_peak, in_swing = p, p, True
            else:
                if p > p_peak:
                    p_peak = p
                elif (p - p_peak) / p_peak <= -limit:
                    gain = (p_peak - p_low) / p_low
                    if gain >= limit: swing_gains.append(gain)
                    p_low, p_peak = p, p

        if in_swing:
            gain = (p_peak - p_low) / p_low
            if gain >= limit: swing_gains.append(gain)

        swing_gains = np.array(swing_gains)
        total_swings = len(swing_gains)

        if total_swings > 0:
            bucket_results[name] = [
                round(((swing_gains >= 0.05) & (swing_gains < 0.10)).sum() / total_swings * 100, 2),
                round(((swing_gains >= 0.10) & (swing_gains < 0.15)).sum() / total_swings * 100, 2),
                round(((swing_gains >= 0.15) & (swing_gains < 0.20)).sum() / total_swings * 100, 2),
                round(((swing_gains >= 0.20) & (swing_gains < 0.25)).sum() / total_swings * 100, 2),
                round(((swing_gains >= 0.25) & (swing_gains < 0.30)).sum() / total_swings * 100, 2),
                round(((swing_gains >= 0.30) & (swing_gains < 0.40)).sum() / total_swings * 100, 2),
                round(((swing_gains >= 0.40) & (swing_gains < 0.50)).sum() / total_swings * 100, 2),
                round((swing_gains >= 0.50).sum() / total_swings * 100, 2)
            ]
            stats_results[name] = [
                total_swings,
                f"+{round(swing_gains.mean() * 100, 2)}%",
                f"+{round(np.median(swing_gains) * 100, 2)}%",
                f"+{round(swing_gains.max() * 100, 2)}%"
            ]
        else:
            bucket_results[name] = [0]*8
            stats_results[name] = [0, "0%", "0%", "0%"]

    cols_b = ['+5%~+10%', '+10%~+15%', '+15%~+20%', '+20%~+25%', '+25%~+30%', '+30%~+40%', '+40%~+50%', '>+50%']
    cols_s = ['歷史波段總次數', '平均波段漲幅', '中位數波段漲幅', '歷史最大波段漲幅']
    return pd.DataFrame(bucket_results, index=cols_b).T, pd.DataFrame(stats_results, index=cols_s).T

# 頁面主體渲染
if symbols_input:
    target_symbols = [s.strip() for s in symbols_input.split(",") if s.strip()]
    with st.spinner("下載並計算數據中..."):
        df_prices = fetch_multi_data(target_symbols, f"{start_year}-01-01")

    if not df_prices.empty:
        df_dd = analyze_drawdown_buckets(df_prices)
        df_rally_b, df_rally_s = analyze_rally_buckets(df_prices, pull_back_limit)

        # 1. 買點分析區
        st.subheader("🔵 【買點分析】歷史波段回檔區間天數佔比 (%)")
        fig_dd_heat = px.imshow(df_dd, text_auto=True, aspect="auto", color_continuous_scale="Blues")
        st.plotly_chart(fig_dd_heat, use_container_width=True)

        # 2. 賣點分析區
        st.subheader(f"🔴 【賣點分析】未拉回 {pull_back_pct}% 前，波段漲幅落點區間佔比 (%)")
        fig_rally_heat = px.imshow(df_rally_b, text_auto=True, aspect="auto", color_continuous_scale="Reds")
        st.plotly_chart(fig_rally_heat, use_container_width=True)

        st.markdown("#### 📋 波段統計摘要")
        st.dataframe(df_rally_s, use_container_width=True)

        # 3. 回撤曲線對照
        st.subheader("📉 【多標的比較】歷史波段回撤對照曲線 (%)")
        fig_curve = go.Figure()
        for col in df_prices.columns:
            s = df_prices[col].dropna()
            dd_curve = (s - s.cummax()) / s.cummax() * 100
            fig_curve.add_trace(go.Scatter(x=dd_curve.index, y=dd_curve, mode='lines', name=col))
        fig_curve.update_layout(hovermode='x unified', template='plotly_white')
        st.plotly_chart(fig_curve, use_container_width=True)
