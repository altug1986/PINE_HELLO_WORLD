"""Fixed-capital audit on closed trade PnL in calendar slots."""
from collections import defaultdict
from datetime import datetime, timezone
from math import sqrt
from statistics import mean, stdev
from zoneinfo import ZoneInfo


def percentile(values, q):
    ranked = sorted(values)
    i = q * (len(ranked) - 1)
    low = int(i)
    return ranked[low] + (ranked[min(low + 1, len(ranked) - 1)] - ranked[low]) * (i - low)


def fixed_capital(trades, capital, start, end, zone):
    if capital <= 0:
        raise ValueError("Sabit sermaye metriği için başlangıç sermayesi pozitif olmalı.")
    tz = ZoneInfo(zone)
    start_at = datetime.fromtimestamp(start / 1000, timezone.utc).astimezone(tz)
    end_at = datetime.fromtimestamp(end / 1000, timezone.utc).astimezone(tz)
    daily, weekly = defaultdict(float), defaultdict(float)
    for t in trades:
        day = datetime.fromtimestamp(t["exit_time"] / 1000, timezone.utc).astimezone(tz).date().toordinal()
        daily[day] += t["pnl"]
        weekly[(day - 1) // 7] += t["pnl"]
    days = [daily[k] for k in range(start_at.date().toordinal(), end_at.date().toordinal() + 1)]
    weeks = [weekly[k] for k in range((start_at.date().toordinal() - 1) // 7, (end_at.date().toordinal() - 1) // 7 + 1)]
    returns = [d / capital for d in days]
    average = mean(returns)
    sigma = stdev(returns) if len(returns) > 1 else 0
    downside = sqrt(mean(min(0, r) ** 2 for r in returns))
    total = sum(t["pnl"] for t in trades)
    drawdowns = {}
    max_underwater = 0
    for minute in (10, 60):
        step = minute * 60_000
        slots = defaultdict(float)
        for t in trades:
            slots[(t["exit_time"] + step - 1) // step] += t["pnl"]
        equity = peak = 0.0
        depth = []
        underwater = 0
        first = start // step
        last = (end + step - 1) // step
        if last - first > 2_000_000:
            raise ValueError("Drawdown örnekleme aralığı çok uzun.")
        for slot in range(first, last + 1):
            equity += slots[slot]
            peak = max(peak, equity)
            dd = max(0, peak - equity) * 100 / capital
            depth.append(dd)
            underwater = underwater + minute if dd > 1e-10 else 0
            if minute == 10:
                max_underwater = max(max_underwater, underwater)
        drawdowns[f"{minute}m"] = {
            **{f"p{p}": percentile(depth, p / 100) for p in (95, 98, 99, 100)},
            "ulcer": sqrt(mean(x * x for x in depth)),
        }
    positive = sum(max(0, t["pnl"]) for t in trades)
    negative = sum(max(0, -t["pnl"]) for t in trades)
    max_dd = drawdowns["10m"]["p100"]

    def cv(values):
        avg = mean(values)
        return stdev(values) / abs(avg) if len(values) > 1 and abs(avg) > 1e-12 else None

    return {
        "basis": "fixed_initial_capital_closed_pnl",
        "net_pnl": total, "closed_legs": len(trades), "profit_factor": positive / negative if negative else None,
        "sharpe_daily_365_rf0": average / sigma * sqrt(365) if sigma else None,
        "sortino_daily_365_rf0": average / downside * sqrt(365) if downside else None,
        "calmar_linear_365": total * 100 * 365 / (capital * len(days) * max_dd) if max_dd else None,
        "daily_cv": cv(days), "weekly_cv": cv(weeks),
        "negative_weeks": sum(v < 0 for v in weeks), "positive_weeks": sum(v > 0 for v in weeks),
        "closed_pnl_drawdown_percent": drawdowns, "longest_underwater_minutes_10m": max_underwater,
        "calendar_timezone": zone,
    }
