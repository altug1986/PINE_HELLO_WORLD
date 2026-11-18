#!/usr/bin/env python3
"""CYBORG command-line backtest and TradingView closed-trade parity checker."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from fractions import Fraction
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

from market_data import minutes, normalize_feed, timestamp
from metrics import fixed_capital
from parity import compare, load_reference, write_trades

HERE = Path(__file__).resolve().parent


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def symbol_info(symbol, values):
    prefix, separator, ticker = symbol.partition(":")
    if not separator:
        raise ValueError(f"Borsa:sembol biçimi gerekli: {symbol}")
    mintick = float(values["mintick"])
    mincontract = float(values["mincontract"])
    pointvalue = float(values.get("pointvalue", 1))
    if any(not math.isfinite(x) or x <= 0 for x in (mintick, mincontract, pointvalue)):
        raise ValueError(f"Geçersiz tick, kontrat adımı veya nokta değeri: {symbol}")
    tick = Fraction(str(values["mintick"]))
    zone = values.get("timezone", "UTC")
    ZoneInfo(zone)
    return {
        "tickerid": symbol, "main_tickerid": symbol, "ticker": ticker, "prefix": prefix, "root": ticker,
        "timezone": zone, "type": values.get("type", "crypto"), "session": "24x7",
        "currency": values["currency"], "basecurrency": values.get("basecurrency", ""),
        "mintick": mintick, "mincontract": mincontract, "pointvalue": pointvalue,
        "volumetype": values.get("volumetype", "base"),
        "pricescale": tick.denominator, "minmove": tick.numerator,
    }


def prepare(config, base, temp):
    pine = (base / config["pine"]).resolve(strict=True)
    tf = minutes(config["timeframe"])
    symbol = config["symbol"]
    zone = config.get("chart_timezone", "Europe/Istanbul")
    ZoneInfo(zone)
    start = timestamp(config["start"], zone)
    end = timestamp(config["end"], zone)
    step = tf * 60_000
    if end <= start or start % step or end % step:
        raise ValueError("Başlangıç ve bitiş sıralı, grafik zaman dilimine hizalı bar açılışları olmalı.")
    symbols = {name: symbol_info(name, data) for name, data in config["symbols"].items()}
    feeds = {}
    feed_info = {}
    for i, feed in enumerate(config["feeds"]):
        name = feed["symbol"]
        timeframe = str(minutes(feed["timeframe"]))
        if name not in symbols:
            raise ValueError(f"Sembol ayarları eksik: {name}")
        files = feeds.setdefault(name, {})
        if timeframe in files:
            raise ValueError(f"Tekrarlı OHLC kaynağı: {name}@{timeframe}")
        target = temp / f"feed-{i}.csv"
        # Causal history before the main test start can seed secondary requests.
        first = timestamp(feed["history_start"], feed.get("timezone", "UTC")) if "history_start" in feed else None
        last = end + step - minutes(timeframe) * 60_000
        if first is not None and first > start:
            raise ValueError("history_start ana başlangıçtan geç olamaz.")
        info = normalize_feed(feed, base, target, first, last, config.get("allow_gaps", False))
        files[timeframe] = str(target)
        feed_info[f"{name}@{timeframe}"] = {**info, "normalized_sha256": file_hash(target)}
    if symbol not in feeds:
        raise ValueError(f"Ana OHLC verilmedi: {symbol}")
    return {
        "pine": str(pine), "main": {"symbol": symbol, "timeframe": str(tf)},
        "start": start, "end": end, "feeds": feeds, "symbols": symbols,
        "volume": {name: {tf: feed_info[f"{name}@{tf}"]["has_volume"] for tf in series} for name, series in feeds.items()},
        "inputs": config.get("inputs", {}), "properties": config.get("properties", {}),
        "allowGaps": config.get("allow_gaps", False), "chartTimezone": zone,
    }, feed_info, file_hash(pine)


def compare_reference(trades, path, zone):
    reference, open_count, properties = load_reference(path, zone)
    return {**compare(trades, reference), "reference_open_legs": open_count,
            "reference_properties": properties, "reference_sha256": file_hash(Path(path)),
            "reference_timezone": zone}


def run(config_path, output, reference, reference_zone):
    config_path = config_path.resolve()
    base = config_path.parent
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    with TemporaryDirectory(prefix="cyborg-backtest-") as tmp:
        temp = Path(tmp)
        manifest, feed_info, pine_hash = prepare(config, base, temp)
        write_json(temp / "manifest.json", manifest)
        subprocess.run(["node", str(HERE / "run-pine.mjs"), str(temp / "manifest.json"),
                        str(temp / "equity.csv"), str(temp / "run.json")], check=True)
        raw = json.loads((temp / "run.json").read_text(encoding="utf-8"))
        trades = raw.pop("trades")
        audit = compare_reference(trades, reference, reference_zone or manifest["chartTimezone"]) if reference else None
        flag = audit["status"] if audit else "no_tradingview_reference"
        if audit and flag == "matched_closed_trades" and raw["warnings"]:
            flag = "matched_closed_trades_with_engine_warnings"
        metrics = fixed_capital(trades, raw["initial_capital"], manifest["start"],
                                manifest["end"] + minutes(config["timeframe"]) * 60_000,
                                manifest["chartTimezone"])
        report = {
            "status": flag, "engine": "PineTS 0.13.1 + CYBORG venue routing v1", "pine_sha256": pine_hash,
            "main": manifest["main"], "start_bar_open_ms": manifest["start"], "end_bar_open_ms": manifest["end"],
            "data_audit": feed_info, "inputs_overridden": manifest["inputs"], "properties_overridden": manifest["properties"],
            **raw, "metrics": metrics, "tradingview_closed_trade_comparison": audit,
        }
        output.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(temp / "equity.csv", output / "equity.csv")
        write_trades(output / "trades.csv", trades)
        write_json(output / "report.json", report)
        print(json.dumps({"status": flag, "bars": raw["bars"], "trades": len(trades),
                          "net_pnl": metrics["net_pnl"], "report": str(output / "report.json")}, ensure_ascii=False))
    return 2 if audit and audit["status"] != "matched_closed_trades" else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cli = commands.add_parser("run")
    cli.add_argument("config", type=Path)
    cli.add_argument("--out", type=Path, required=True, help="Yeni çıktı dizini")
    cli.add_argument("--tv", type=Path, help="TradingView XLSX/CSV")
    cli.add_argument("--tv-timezone", help="Exportun grafik saat dilimi")
    cli = commands.add_parser("compare")
    cli.add_argument("trades", type=Path, help="Motorun trades.csv dosyası")
    cli.add_argument("--tv", type=Path, required=True)
    cli.add_argument("--tv-timezone", default="Europe/Istanbul")
    cli.add_argument("--out", type=Path, required=True)
    cli = commands.add_parser("inspect-tv")
    cli.add_argument("tv", type=Path)
    cli.add_argument("--tv-timezone", default="Europe/Istanbul")
    args = parser.parse_args()
    if args.command == "run":
        return run(args.config, args.out, args.tv, args.tv_timezone)
    if args.command == "compare":
        trades, _, _ = load_reference(args.trades, "UTC")
        audit = compare_reference(trades, args.tv, args.tv_timezone)
        with args.out.open("x", encoding="utf-8") as stream:
            json.dump(audit, stream, ensure_ascii=False, indent=2)
        print(audit["status"])
        return 0 if audit["status"] == "matched_closed_trades" else 2
    trades, open_count, properties = load_reference(args.tv, args.tv_timezone)
    print(json.dumps({"closed_legs": len(trades), "open_legs": open_count, "properties": properties,
                      "first_entry_ms": min((t["entry_time"] for t in trades), default=None),
                      "last_exit_ms": max((t["exit_time"] for t in trades), default=None)},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, IndexError, TypeError, subprocess.CalledProcessError) as error:
        print(f"CYBORG: {error}", file=sys.stderr)
        sys.exit(1)
