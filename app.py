import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings('ignore')

st.set_page_config(page_title="股票歷史波段與交易策略分析系統", page_icon="📈", layout="wide")

st.title("📈 股票歷史波段回撤、漲幅與實戰策略分析系統")

# ==========================================
# 側邊欄：使用者互動控制項
# ==========================================
st.sidebar.header("⚙️ 核心參數設定")

symbols_input = st.sidebar.text_input(
    "輸入股票代碼 (多檔以逗號隔開)", 
    value="0050, 00662, 00631L, 00685L, 00865B, 00933B"
).strip().upper()

auto_adjust_option = st.sidebar.radio(
    "股價計算模式 (配息還原)",
    options=["未還原收盤價 (純資本利得)", "還原權息股價 (含息總報酬)"],
    index=0,
    help="• 未還原收盤價：適合槓桿商品 (如 00631L)。\n• 還原權息股價：適合高股息 ETF (如 0056, 00878)。"
)
is_auto_adjust = True if "含息總報酬" in auto_adjust_option else False

start_year = st.sidebar.slider("歷史資料起始年份", min_value=2008, max_value=2024, value=2012)
pull_back_pct = st.sidebar.slider("波段結算拉回門檻 (%)", min_value=2.0, max_value=15.0, value=5.0, step=0.5)
pull_back_limit = pull_back_pct / 100.0

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 即時指針：波段定義模式")
swing_mode = st.sidebar.selectbox(
    "選擇卡片呈現的高低點定義：",
    options=[
        "近期短線波段 (自訂近期交易日)",
        "演算法進行中波段 (拉回門檻判定)",
        "歷史全局天花板 (全期間最高/低)"
    ],
    index=0
)

rolling_days = 60
if "近期短線波段" in swing_mode:
    rolling_days = st.sidebar.slider("近期追蹤天數 (交易日)", min_value=20, max_value=180, value=60, step=10)

st.sidebar.markdown("---")
st.sidebar.subheader("🪜 金字塔加碼策略參數")
num_tiers = st.sidebar.number_input("設定分批加碼總階段數", min_value=1, max_value=6, value=3, step=1)

tiers_config = []
total_weight = 0

default_pcts = [-5.0, -10.0, -15.0, -20.0, -25.0, -30.0]
default_weights = [20, 30, 50, 20, 20, 20]

for i in range(int(num_tiers)):
    col_t1, col_t2 = st.sidebar.columns(2)
    def_p = default_pcts[i] if i < len(default_pcts) else -5.0 * (i + 1)
    def_w = default_weights[i] if i < len(default_weights) else 100 // int(num_tiers)
    
    with col_t1:
        t_pct = st.number_input(f"第 {i+1} 階回檔門檻 (%)", min_value=-60.0, max_value=-1.0, value=def_p, step=1.0, key=f"tier_pct_{i}")
    with col_t2:
        t_w = st.number_input(f"第 {i+1} 階資金權重 (%)", min_value=1, max_value=100, value=def_w, step=5, key=f"tier_w_{i}")
    
    total_weight += t_w
    tiers_config.append((t_pct / 100.0, t_w / 100.0))

tiers_config = sorted(tiers_config, key=lambda x: x[0], reverse=True)

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
    prices = series.values
    dates = series.index
    swing_gains, swing_durations, swing_points = [], [], []
    
    if len(prices) < 2:
        return np.array([]), np.array([]), [], {}

    in_swing = False
    p_low, p_low_date, p_low_idx = prices[0], dates[0], 0
    p_peak, p_peak_date, p_peak_idx = prices[0], dates[0], 0

    for idx, (p, d) in enumerate(zip(prices, dates)):
        if not in_swing:
            p_low, p_low_date, p_low_idx = p, d, idx
            p_peak, p_peak_date, p_peak_idx = p, d, idx
            in_swing = True
        else:
            if p > p_peak:
                p_peak, p_peak_date, p_peak_idx = p, d, idx
            elif (p - p_peak) / p_peak <= -limit:
                gain = (p_peak - p_low) / p_low
                duration = p_peak_idx - p_low_idx
                if gain >= limit:
                    swing_gains.append(gain)
                    swing_durations.append(max(1, duration))
                    swing_points.append({
                        'low_date': p_low_date, 'low_price': p_low,
                        'peak_date': p_peak_date, 'peak_price': p_peak,
                        'gain': gain, 'duration': max(1, duration)
                    })
                p_low, p_low_date, p_low_idx = p, d, idx
                p_peak, p_peak_date, p_peak_idx = p, d, idx

    current_active_swing = {
        'low_date': p_low_date, 'low_price': p_low,
        'peak_date': p_peak_date, 'peak_price': p_peak
    }

    if in_swing:
        gain = (p_peak - p_low) / p_low
        duration = p_peak_idx - p_low_idx
        if gain >= limit:
            swing_gains.append(gain)
            swing_durations.append(max(1, duration))
            swing_points.append({
                'low_date': p_low_date, 'low_price': p_low,
                'peak_date': p_peak_date, 'peak_price': p_peak,
                'gain': gain, 'duration': max(1, duration)
            })

    return np.array(swing_gains), np.array(swing_durations), swing_points, current_active_swing

def analyze_rally_buckets(df_prices, limit):
    bucket_results, stats_results, swings_dict, active_swings_dict = {}, {}, {}, {}
    for name in df_prices.columns:
        series = df_prices[name].dropna()
        swing_gains, swing_durations, swing_points, active_swing = extract_swings(series, limit)
        swings_dict[name] = swing_points
        active_swings_dict[name] = active_swing
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
            
            avg_gain = swing_gains.mean() * 100
            avg_dur = swing_durations.mean()
            velocity = avg_gain / avg_dur if avg_dur > 0 else 0
            
            stats_results[name] = [
                total_swings,
                f"+{round(avg_gain, 2)}%",
                f"+{round(np.median(swing_gains) * 100, 2)}%",
                f"+{round(swing_gains.max() * 100, 2)}%",
                f"{round(avg_dur, 1)} 天",
                f"+{round(velocity, 2)}% / 天"
            ]
        else:
            bucket_results[name] = [0]*8
            stats_results[name] = [0, "0%", "0%", "0%", "0 天", "0% / 天"]

    cols_b = ['+5%~+10%', '+10%~+15%', '+15%~+20%', '+20%~+25%', '+25%~+30%', '+30%~+40%', '+40%~+50%', '>+50%']
    cols_s = ['歷史波段總次數', '平均波段漲幅', '中位數波段漲幅', '歷史最大波段漲幅', '平均歷時天數', '日均推進速度']
    return pd.DataFrame(bucket_results, index=cols_b).T, pd.DataFrame(stats_results, index=cols_s).T, swings_dict, active_swings_dict

def calculate_time_under_water(df_prices):
    tuw_thresholds = [-0.05, -0.10, -0.15, -0.20, -0.25, -0.30]
    records = []
    
    for name in df_prices.columns:
        series = df_prices[name].dropna()
        cummax = series.cummax()
        drawdown = (series - cummax) / cummax
        
        row_data = {}
        for th in tuw_thresholds:
            th_pct = int(abs(th) * 100)
            in_dd = False
            start_idx = 0
            durations = []
            
            for i in range(len(series)):
                if not in_dd:
                    if drawdown.iloc[i] <= th:
                        in_dd = True
                        start_idx = i
                else:
                    if drawdown.iloc[i] == 0:
                        in_dd = False
                        durations.append(i - start_idx)
            
            if in_dd:
                durations.append(len(series) - 1 - start_idx)
                
            if durations:
                avg_days = round(np.mean(durations), 1)
                max_days = int(np.max(durations))
                row_data[f"≤-{th_pct}% 平均解套"] = f"{avg_days} 天"
                row_data[f"≤-{th_pct}% 最長套牢"] = f"{max_days} 天"
            else:
                row_data[f"≤-{th_pct}% 平均解套"] = "未曾跌破"
                row_data[f"≤-{th_pct}% 最長套牢"] = "未曾跌破"
                
        records.append(pd.Series(row_data, name=name))
        
    return pd.DataFrame(records)

def simulate_pyramid_strategy(series, tiers):
    cummax = series.cummax()
    drawdown = (series - cummax) / cummax
    
    rounds = []
    in_round = False
    triggered_tiers = set()
    purchases = []
    
    for i in range(len(series)):
        p = series.iloc[i]
        dd = drawdown.iloc[i]
        
        if dd == 0:
            if in_round and purchases:
                total_w = sum(w for _, w in purchases)
                if total_w > 0:
                    avg_cost = sum(price * w for price, w in purchases) / total_w
                    gain_pct = (p - avg_cost) / avg_cost * 100
                    rounds.append({
                        'executed_tiers': len(purchases),
                        'avg_cost': avg_cost,
                        'exit_price': p,
                        'gain_pct': gain_pct
                    })
            in_round = False
            triggered_tiers = set()
            purchases = []
        else:
            in_round = True
            for tier_idx, (th, weight) in enumerate(tiers):
                if dd <= th and tier_idx not in triggered_tiers:
                    triggered_tiers.add(tier_idx)
                    purchases.append((p, weight))

    if not rounds:
        return {
            "觸發總輪數": "0 輪",
            "平均每輪加權報酬": "0.0%",
            "策略勝率": "0.0%",
            "各階觸發分佈": "未曾觸發門檻"
        }
        
    total_rounds = len(rounds)
    wins = sum(1 for r in rounds if r['gain_pct'] > 0)
    avg_gain = np.mean([r['gain_pct'] for r in rounds])
    win_rate = (wins / total_rounds) * 100
    
    tier_counts = {}
    for r in rounds:
        tier_counts[r['executed_tiers']] = tier_counts.get(r['executed_tiers'], 0) + 1
    tier_dist_str = ", ".join([f"加至第{k}階: {v}次" for k, v in sorted(tier_counts.items())])
    
    return {
        "觸發總輪數": f"{total_rounds} 輪",
        "平均每輪加權報酬": f"+{round(avg_gain, 2)}%",
        "策略勝率": f"{round(win_rate, 1)}%",
        "各階觸發分佈": tier_dist_str
    }

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
        df_rally_b, df_rally_s, swings_dict, active_swings_dict = analyze_rally_buckets(df_prices, pull_back_limit)

        # 1. 當前水位即時指針
        st.subheader("🎯 【當前水位即時指針】波段高低點與進場警示")
        st.caption(f"📌 目前高低點計算基準：**{swing_mode}**" + (f" (最近 {rolling_days} 個交易日)" if "近期短線波段" in swing_mode else ""))
        
        cols = st.columns(len(df_prices.columns))
        
        for idx, col_name in enumerate(df_prices.columns):
            series = df_prices[col_name].dropna()
            latest_price = round(series.iloc[-1], 2)
            latest_date = series.index[-1].strftime('%Y-%m-%d')
            
            if "近期短線波段" in swing_mode:
                recent_series = series.iloc[-rolling_days:] if len(series) >= rolling_days else series
                peak_p = round(recent_series.max(), 2)
                peak_d = recent_series.idxmax().strftime('%Y-%m-%d')
                low_p = round(recent_series.min(), 2)
                low_d = recent_series.idxmin().strftime('%Y-%m-%d')
            elif "演算法進行中波段" in swing_mode:
                active_info = active_swings_dict.get(col_name, {})
                peak_p = round(active_info.get('peak_price', series.max()), 2)
                peak_d = active_info.get('peak_date', series.index[0]).strftime('%Y-%m-%d')
                low_p = round(active_info.get('low_price', series.min()), 2)
                low_d = active_info.get('low_date', series.index[0]).strftime('%Y-%m-%d')
            else:
                peak_p = round(series.max(), 2)
                peak_d = series.idxmax().strftime('%Y-%m-%d')
                low_p = round(series.min(), 2)
                low_d = series.idxmin().strftime('%Y-%m-%d')
            
            curr_dd = round((latest_price - peak_p) / peak_p * 100, 2)
            all_dd = (series - series.cummax()) / series.cummax() * 100
            rare_pct = round((all_dd <= curr_dd).sum() / len(all_dd) * 100, 1)
            
            with cols[idx]:
                st.markdown(f"### **{col_name}**")
                st.metric("最新股價", f"${latest_price}", f"距波段高點 {curr_dd}%")
                st.caption(f"📅 報價日期：`{latest_date}`")
                
                st.markdown(
                    f"""
                    <div style="background-color: #f8f9fa; padding: 10px; border-radius: 8px; border: 1px solid #e9ecef; margin-bottom: 10px; font-size: 13px;">
                        <b>🔴 波段高點：</b> ${peak_p}<br>
                        <span style="color: gray; font-size: 11px;">(高點日: {peak_d})</span><br>
                        <b>🟢 波段低點：</b> ${low_p}<br>
                        <span style="color: gray; font-size: 11px;">(低點日: {low_d})</span>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                if curr_dd <= -20.0:
                    st.error(f"🚨 **極度罕見買點**\n\n歷史僅 **{rare_pct}%** 天數比現在更深！")
                elif curr_dd <= -10.0:
                    st.success(f"💎 **波段進場甜蜜點**\n\n歷史僅 **{rare_pct}%** 天數處於此回檔深度！")
                elif curr_dd <= -5.0:
                    st.warning(f"👀 **拉回觀察區**\n\n歷史有 **{rare_pct}%** 天數比現在深。")
                else:
                    st.info(f"☁️ **常態強勢區**\n\n距波段高點 < 5%，無顯著回檔。")

        st.markdown("---")

        # 2. 買點分析區 (熱力圖)
        st.subheader("🔵 【買點分析】歷史波段回檔區間天數佔比 (%)")
        fig_dd_heat = px.imshow(
            df_dd, 
            text_auto=True, 
            aspect="auto", 
            color_continuous_scale="Blues",
            labels=dict(x="回檔深度區間", y="標的代碼", color="天數佔比(%)")
        )
        st.plotly_chart(fig_dd_heat, use_container_width=True)

        # 3. 賣點分析區
        st.subheader(f"🔴 【賣點分析】未拉回 {pull_back_pct}% 前，波段漲幅落點區間佔比 (%)")
        fig_rally_heat = px.imshow(
            df_rally_b, 
            text_auto=True, 
            aspect="auto", 
            color_continuous_scale="Reds",
            labels=dict(x="波段漲幅區間", y="標的代碼", color="發生比例(%)")
        )
        st.plotly_chart(fig_rally_heat, use_container_width=True)

        st.markdown("#### 📋 波段統計摘要 (含歷時天數與日均推進速度)")
        st.dataframe(df_rally_s, use_container_width=True)

        st.markdown("---")

        # 4. 回撤解套時間統計 (Time Under Water)
        st.subheader("⏳ 【時間成本分析】回撤深度平均解套天數與最長套牢期 (TUW)")
        st.caption("統計當股價跌破特定深度後，重新創歷史新高所需的交易日數。")
        df_tuw = calculate_time_under_water(df_prices)
        st.dataframe(df_tuw, use_container_width=True)

        st.markdown("---")

        # 5. 金字塔分批加碼策略回測
        st.subheader(f"🪜 【資金配置實戰】自訂 {num_tiers} 階段金字塔加碼策略回測")
        if total_weight != 100:
            st.warning(f"⚠️ 目前各階資金權重加總為 **{total_weight}%**，建議調整各階加總至 100% 以利精準試算。")

        pyramid_results = {}
        for col_name in df_prices.columns:
            s = df_prices[col_name].dropna()
            pyramid_results[col_name] = simulate_pyramid_strategy(s, tiers_config)
            
        df_pyramid = pd.DataFrame(pyramid_results).T
        st.dataframe(df_pyramid, use_container_width=True)

        st.markdown("---")

        # 6. 動態疊加走勢與波段標註圖
        st.subheader(f"🔍 【波段軌跡驗證】價格走勢與波段標註圖 (拉回門檻: {pull_back_pct}%)")
        selected_ticker = st.selectbox("選擇要檢視波段軌跡的標的：", df_prices.columns)
        
        if selected_ticker:
            s_price = df_prices[selected_ticker].dropna()
            points = swings_dict.get(selected_ticker, [])
            
            fig_overlay = go.Figure()
            fig_overlay.add_trace(go.Scatter(x=s_price.index, y=s_price.values, mode='lines', name='收盤價', line=dict(color='#888888', width=1.5)))
            
            if points:
                low_dates = [p['low_date'] for p in points]
                low_prices = [p['low_price'] for p in points]
                peak_dates = [p['peak_date'] for p in points]
                peak_prices = [p['peak_price'] for p in points]
                gains_text = [f"起漲點<br>價格: {p['low_price']:.2f}" for p in points]
                peaks_text = [f"頂點<br>價格: {p['peak_price']:.2f}<br>波段漲幅: +{p['gain']*100:.1f}%<br>歷時: {p['duration']}天" for p in points]
                
                fig_overlay.add_trace(go.Scatter(
                    x=low_dates, y=low_prices, mode='markers',
                    name='波段起點 (Low)', marker=dict(color='green', size=7, symbol='circle'),
                    hovertext=gains_text, hoverinfo='text+x'
                ))
                
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

        # 7. 回撤曲線對照
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
