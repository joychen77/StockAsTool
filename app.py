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

    # 修復：統一欄位名稱，避免 Pandas 合併時產生多餘的 None 欄位
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
