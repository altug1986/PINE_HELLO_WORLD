#!/usr/bin/env python3
"""Fast repeatable offline Rust backtest runner with a fail-closed TV trade parity gate.

Private OHLC, Pine and TradingView exports remain local. A matching normalized
trade list proves ONLY that the selected fields matched, not universal TV parity.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
COMPARATOR = Path(__file__).with_name("compare_trades.py")
spec = importlib.util.spec_from_file_location("tv_trade_compare", COMPARATOR)
compare_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare_mod)


def file_fingerprint(path, cache, force=False):
    path = Path(path).resolve()
    st = path.stat()
    key = str(path)
    previous = cache.get(key, {})
    if not force and previous.get("bytes") == st.st_size and previous.get("mtime_ns") == st.st_mtime_ns:
        return previous
    sha = hashlib.sha256()
    first_line = b""
    newlines = 0
    with path.open("rb") as f:
        first_line = f.readline().lstrip(b"\xef\xbb\xbf").strip()
        f.seek(0)
        last_byte = b""
        while True:
            chunk = f.read(4 * 1024 * 1024)
            if not chunk:
                break
            sha.update(chunk)
            newlines += chunk.count(b"\n")
            last_byte = chunk[-1:]
    return {
        "path": key, "sha256": sha.hexdigest(), "bytes": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "lines": newlines + (1 if last_byte != b"\n" and st.st_size else 0),
        "header": first_line.decode("utf-8", "replace"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ohlc", required=True, help="Normalized CSV: timestamp,open,high,low,close")
    ap.add_argument("--signals", required=True, help="Normalized CSV: timestamp,action,fraction")
    ap.add_argument("--tv", help="Normalized TV List of Trades CSV (optional)")
    ap.add_argument("--pine", help="Pine source for provenance hashing ONLY; not interpreted")
    ap.add_argument("--out", default="results/fast", help="Output prefix")
    ap.add_argument("--engine", default=str(ROOT / "target/release/tv-rust-backtester"))
    ap.add_argument("--build", action="store_true", help="Compile release engine once")
    ap.add_argument("--rehash", action="store_true", help="Rehash inputs regardless of metadata cache")
    ap.add_argument("--initial", type=float, default=2500)
    ap.add_argument("--cash-per-leg", type=float, default=250)
    ap.add_argument("--commission-percent", type=float, default=0.04)
    ap.add_argument("--slippage-ticks", type=float, default=1)
    ap.add_argument("--tick-size", type=float, default=0.001)
    ap.add_argument("--max-legs", type=int, default=2)
    ap.add_argument("--on-close", action="store_true")
    ap.add_argument("--abs", type=float, default=1e-6)
    ap.add_argument("--rel", type=float, default=1e-6)
    x = ap.parse_args()

    prefix = Path(x.out).resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    cache_file = prefix.parent / ".fast_sync_fingerprints.json"
    try:
        cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    except (ValueError, OSError):
        cache = {}

    sources = {}
    started = time.perf_counter()
    required = {"ohlc": "timestamp,open,high,low,close",
                "signals": "timestamp,action,fraction"}
    for label in ("ohlc", "signals", "tv", "pine"):
        fname = getattr(x, label)
        if fname:
            meta = file_fingerprint(fname, cache, force=x.rehash)
            if label in required and meta["header"] != required[label]:
                ap.error(f"{label} must have header {required[label]!r}; got {meta['header']!r}")
            sources[label] = meta
            cache[str(Path(fname).resolve())] = meta
    cache_file.write_text(json.dumps(cache, indent=2) + "\n")
    prep_seconds = time.perf_counter() - started

    if x.build:
        subprocess.run(["cargo", "build", "--release"], cwd=ROOT, check=True)
    binary = Path(x.engine)
    if not binary.is_file():
        ap.error(f"No compiled engine at {binary}. Run with --build once (Rust toolchain required).")

    argv = [
        str(binary), "--ohlc", str(Path(x.ohlc).resolve()),
        "--signals", str(Path(x.signals).resolve()),
        "--out", str(prefix),
        "--initial", str(x.initial), "--cash-per-leg", str(x.cash_per_leg),
        "--commission-percent", str(x.commission_percent),
        "--slippage-ticks", str(x.slippage_ticks), "--tick-size", str(x.tick_size),
        "--max-legs", str(x.max_legs),
    ]
    if x.on_close:
        argv.append("--on-close")

    started = time.perf_counter()
    process = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
    runtime = time.perf_counter() - started
    if process.returncode:
        sys.stderr.write(process.stderr or process.stdout)
        return process.returncode

    summary = json.loads(Path(str(prefix) + "_summary.json").read_text())
    result = {
        "status": "NO_REFERENCE" if not x.tv else "PARITY_FAIL",
        "tv_full_parity_claim": False,
        "selected_fields_parity": None,
        "first_mismatch": None,
        "engine_seconds": round(runtime, 6),
        "preparation_seconds": round(prep_seconds, 6),
        "observed_bars_per_second": round(summary["bars"]/runtime, 2) if runtime > 0 else None,
        "engine_summary": summary,
        "sources": sources,
        "runtime_config": {
            "initial": x.initial, "cash_per_leg": x.cash_per_leg,
            "commission_percent": x.commission_percent, "slippage_ticks": x.slippage_ticks,
            "tick_size": x.tick_size, "max_legs": x.max_legs, "on_close": x.on_close,
        }
    }
    if x.tv:
        try:
            tv = compare_mod.read(x.tv)
            engine = compare_mod.read(str(prefix) + "_trades.csv")
            mismatch = compare_mod.compare(tv, engine, x.abs, x.rel)
        except (ValueError, OSError) as exc:
            mismatch = str(exc)
        result["selected_fields_parity"] = mismatch is None
        result["first_mismatch"] = mismatch
        result["status"] = "PASS_SELECTED_FIELDS_ONLY" if mismatch is None else "PARITY_FAIL"

    result_path = Path(str(prefix) + "_sync.json")
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "status": result["status"],
        "engine_seconds": result["engine_seconds"],
        "bars_per_second": result["observed_bars_per_second"],
        "closed_trades": summary["closed_trades"],
        "first_mismatch": result["first_mismatch"],
        "report": str(result_path),
    }, ensure_ascii=False, indent=2))
    return 2 if result["status"] == "PARITY_FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
