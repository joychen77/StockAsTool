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

# ==========================================
# 側邊欄：使用者互動控制項
# ==========================================
st.sidebar.header("⚙️ 參數設定")

symbols_input = st.sidebar.text_input(
    "輸入股票代碼 (多檔以逗號隔開)", 
    value="0050, 00662, 00631L, 00685L, 00865B, 00933B"
).strip().upper()

auto_adjust_option = st.sidebar.radio(
    "股價計算模式 (配息還原)",
    options=["未還原收盤價 (純資本利得)", "還原權息股價 (含息總報酬)"],
    index=0,
    help="• 未還原收盤價：適合不配息的槓桿商品 (如 00631L) 或只看票面價格波動。\n• 還原權息股價：適合高股息 ETF (如 0056, 00878) 或長期存股總報酬分析。"
)
is_auto_adjust = True if "含息總報酬" in auto_adjust_option else False

start_year = st.sidebar.slider("歷史資料起始年份", min_value=2008, max_value=2024, value=2012)
pull_back_pct = st.sidebar.slider("波段結算拉回門檻 (%)", min_value=2.0, max_value=15.0, value=5.0, step=0.5)
pull_back_limit = pull_back_pct / 100.0

st.sidebar.markdown("---")
st.sidebar.caption("💡 **智慧補全提示**：輸入 `2330` 或 `00933B` 等純代碼時，系統會自動嘗試上市 (.TW) 與上櫃 (.TWO)。")

# ==========================================
# 核心資料處理與演算法函式
# ==========================================
def smart_download_ticker(symbol, start_date, auto_adjust):
    candidates = []
    if symbol.endswith(".TW") or symbol.endswith(".TWO"):
        candidates = [symbol]
    else:
        candidates = [f"{symbol}.TW", f"{symbol}.TWO", symbol]

    for ticker in candidates:
        try:
            df = yf.download(ticker, start=start_date, auto_adjust=auto_adjust, progress=False)
            if not df.empty and len(df) > 30:
                if isinstance(df.columns, pd.MultiIndex):
                    target_col = 'Close'
                    series = df[target_col][ticker] if ticker in df[target_col].columns else df[target_col].iloc[:, 0]
                else:
                    series = df['Close']
                
                series = series.dropna()
                if len(series) > 30:
                    return ticker, series
        except Exception:
            continue

    return None, None

@st.cache_data(ttl=3600)
def fetch_multi_data(symbols_list, start_date, auto_adjust):
    downloaded = {}
    failed_symbols = []

    for symbol in symbols_list:
        valid_ticker, series = smart_download_ticker(symbol, start_date, auto_adjust)
        
        if series is not None:
            returns = series.pct_change().abs()
            clean_series = series.copy()
            clean_series[returns > 0.20] = np.nan
            clean_series = clean_series.ffill().bfill()
            
            med_price = clean_series.median()
            if clean_series.iloc[:30].median() > med_price * 2.0:
                valid_mask = clean_series <= (med_price * 2.0)
                first_valid_idx = valid_mask.idxmax() if valid_mask.any() else None
                if first_valid_idx:
                    clean_series = clean_series.loc[first_valid_idx:]
                    
            downloaded[valid_ticker] = clean_series
        else:
            failed_symbols.append(symbol)

    return pd.DataFrame(downloaded), failed_symbols

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

def extract_swings(series, limit):
    """擷取波段數值與波段軌跡點 (起點 Low 與頂點 Peak)"""
    prices = series.values
    dates = series.index
    swing_gains = []
    swing_points = []
    
    if len(prices) < 2:
        return np.array([]), []

    in_swing = False
    p_low, p_low_date = prices[0], dates[0]
    p_peak, p_peak_date = prices[0], dates[0]

    for p, d in zip(prices, dates):
        if not in_swing:
            p_low, p_low_date = p, d
            p_peak, p_peak_date = p, d
            in_swing = True
        else:
            if p > p_peak:
                p_peak, p_peak_date = p, d
            elif (p - p_peak) / p_peak <= -limit:
                gain = (p_peak - p_low) / p_low
                if gain >= limit:
                    swing_gains.append(gain)
                    swing_points.append({
                        'low_date': p_low_date, 'low_price': p_low,
                        'peak_date': p_peak_date, 'peak_price': p_peak,
                        'gain': gain
                    })
                p_low, p_low_date = p, d
                p_peak, p_peak_date = p, d

    if in_swing:
        gain = (p_peak - p_low) / p_low
        if gain >= limit:
            swing_gains.append(gain)
            swing_points.append({
                'low_date': p_low_date, 'low_price': p_low,
                'peak_date': p_peak_date, 'peak_price': p_peak,
                'gain': gain
            })

    return np.array(swing_gains), swing_points

def analyze_rally_buckets(df_prices, limit):
    bucket_results, stats_results, swings_dict = {}, {}, {}
    for name in df_prices.columns:
        series = df_prices[name].dropna()
        swing_gains, swing_points = extract_swings(series, limit)
        swings_dict[name] = swing_points
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
    return pd.DataFrame(bucket_results, index=cols_b).T, pd.DataFrame(stats_results, index=cols_s).T, swings_dict

# ==========================================
# 頁面主體渲染
# ==========================================
if symbols_input:
    target_symbols = [s.strip() for s in symbols_input.split(",") if s.strip()]
    with st.spinner("下載並計算數據中..."):
        df_prices, failed_tickers = fetch_multi_data(target_symbols, f"{start_year}-01-01", is_auto_adjust)

    if failed_tickers:
        st.warning(f"⚠️ 以下代碼查無數據：`{', '.join(failed_tickers)}`")

    if not df_prices.empty:
        high_dividend_symbols = [col for col in df_prices.columns if any(k in col for k in ['0056', '00878', '00919', '00918', '00713'])]
        if high_dividend_symbols and not is_auto_adjust:
            st.info(f"💡 偵測到高股息相關標的 (`{', '.join(high_dividend_symbols)}`)，建議可於左側開關切換為 **「還原權息股價」** 評估真實總報酬！")

        df_dd = analyze_drawdown_buckets(df_prices)
        df_rally_b, df_rally_s, swings_dict = analyze_rally_buckets(df_prices, pull_back_limit)

        # ------------------------------------------
        # 1. 當前水位即時指針與買點警示卡片 (新增)
        # ------------------------------------------
        st.subheader("🎯 【當前水位即時指針】進場甜蜜點警示")
        cols = st.columns(len(df_prices.columns))
        
        for idx, col_name in enumerate(df_prices.columns):
            series = df_prices[col_name].dropna()
            latest_price = round(series.iloc[-1], 2)
            cummax_price = round(series.cummax().iloc[-1], 2)
            curr_dd = round((latest_price - cummax_price) / cummax_price * 100, 2)
            
            # 計算目前回撤深度在歷史上的累積出現天數比例 (比當前回撤更深的天數佔比)
            all_dd = (series - series.cummax()) / series.cummax() * 100
            rare_pct = round((all_dd <= curr_dd).sum() / len(all_dd) * 100, 1)
            
            with cols[idx]:
                st.markdown(f"#### **{col_name}**")
                st.metric("最新股價", f"${latest_price}", f"距高點 {curr_dd}%")
                
                # 燈號判定
                if curr_dd <= -20.0:
                    st.error(f"🚨 **極度罕見買點**\n\n歷史僅 **{rare_pct}%** 天數比現在更便宜！")
                elif curr_dd <= -10.0:
                    st.success(f"💎 **波段進場甜蜜點**\n\n歷史僅 **{rare_pct}%** 天數處於此回檔深度！")
                elif curr_dd <= -5.0:
                    st.warning(f"👀 **拉回觀察區**\n\n歷史有 **{rare_pct}%** 天數比現在深。")
                else:
                    st.info(f"☁️ **常態強勢區**\n\n距高點小於 5%，無顯著回檔。")

        st.markdown("---")

        # ------------------------------------------
        # 2. 買點分析區 (熱力圖)
        # ------------------------------------------
        st.subheader("🔵 【買點分析】歷史波段回檔區間天數佔比 (%)")
        fig_dd_heat = px.imshow(
            df_dd, 
            text_auto=True, 
            aspect="auto", 
            color_continuous_scale="Blues",
            labels=dict(x="回檔深度區間", y="標的代碼", color="天數佔比(%)")
        )
        st.plotly_chart(fig_dd_heat, use_container_width=True)

        # ------------------------------------------
        # 3. 賣點分析區 (熱力圖與摘要)
        # ------------------------------------------
        st.subheader(f"🔴 【賣點分析】未拉回 {pull_back_pct}% 前，波段漲幅落點區間佔比 (%)")
        fig_rally_heat = px.imshow(
            df_rally_b, 
            text_auto=True, 
            aspect="auto", 
            color_continuous_scale="Reds",
            labels=dict(x="波段漲幅區間", y="標的代碼", color="發生比例(%)")
        )
        st.plotly_chart(fig_rally_heat, use_container_width=True)

        st.markdown("#### 📋 波段統計摘要")
        st.dataframe(df_rally_s, use_container_width=True)

        # ------------------------------------------
        # 4. 動態疊加走勢與波段標註圖 (新增)
        # ------------------------------------------
        st.subheader(f"🔍 【波段軌跡驗證】價格走勢與波段標註圖 (拉回門檻: {pull_back_pct}%)")
        selected_ticker = st.selectbox("選擇要檢視波段軌跡的標的：", df_prices.columns)
        
        if selected_ticker:
            s_price = df_prices[selected_ticker].dropna()
            points = swings_dict.get(selected_ticker, [])
            
            fig_overlay = go.Figure()
            # 股價主曲線
            fig_overlay.add_trace(go.Scatter(x=s_price.index, y=s_price.values, mode='lines', name='收盤價', line=dict(color='#888888', width=1.5)))
            
            # 標記起漲點 (Low) 與頂點 (Peak)
            if points:
                low_dates = [p['low_date'] for p in points]
                low_prices = [p['low_price'] for p in points]
                peak_dates = [p['peak_date'] for p in points]
                peak_prices = [p['peak_price'] for p in points]
                gains_text = [f"起漲點<br>價格: {p['low_price']:.2f}" for p in points]
                peaks_text = [f"頂點<br>價格: {p['peak_price']:.2f}<br>波段漲幅: +{p['gain']*100:.1f}%" for p in points]
                
                # 綠點：起漲點
                fig_overlay.add_trace(go.Scatter(
                    x=low_dates, y=low_prices, mode='markers',
                    name='波段起點 (Low)', marker=dict(color='green', size=7, symbol='circle'),
                    hovertext=gains_text, hoverinfo='text+x'
                ))
                
                # 紅點：波段頂點
                fig_overlay.add_trace(go.Scatter(
                    x=peak_dates, y=peak_prices, mode='markers',
                    name='波段頂點 (Peak)', marker=dict(color='red', size=7, symbol='triangle-up'),
                    hovertext=peaks_text, hoverinfo='text+x'
                ))
            
            fig_overlay.update_layout(
                xaxis_title="日期", yaxis_title="價格",
                hovermode='closest', template='plotly_white',
                height=500
            )
            st.plotly_chart(fig_overlay, use_container_width=True)

        # ------------------------------------------
        # 5. 回撤曲線對照
        # ------------------------------------------
        st.subheader("📉 【多標的比較】歷史波段回撤對照曲線 (%)")
        fig_curve = go.Figure()
        for col in df_prices.columns:
            s = df_prices[col].dropna()
            dd_curve = (s - s.cummax()) / s.cummax() * 100
            fig_curve.add_trace(go.Scatter(x=dd_curve.index, y=dd_curve, mode='lines', name=col))
        fig_curve.update_layout(
            xaxis_title="日期",
            yaxis_title="距歷史高點回檔比例 (%)",
            hovermode='x unified', 
            template='plotly_white'
        )
        st.plotly_chart(fig_curve, use_container_width=True)
