"""Executable end-to-end regression with independently calculated fill/PnL."""
import csv
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
START = 1704067200000
SCRIPT = """//@version={version}
strategy("Smoke", initial_capital=500, commission_type=strategy.commission.percent, commission_value=0.04, slippage=1)
if bar_index == 1
    strategy.entry("L", strategy.long, qty=2)
if bar_index == 4
    strategy.close("L")
"""


def main():
    with TemporaryDirectory(prefix="cyborg-selftest-") as folder:
        d = Path(folder)
        with (d / "ohlc.csv").open("w") as file:
            writer = csv.writer(file)
            writer.writerow(["time", "open", "high", "low", "close", "volume"])
            for i in range(30):
                writer.writerow([START + i * 60000, 100 + i, 103 + i, 99 + i, 102 + i, 100])
        with (d / "ref.csv").open("w") as file:
            writer = csv.writer(file)
            writer.writerow(["Trade #", "Type", "Date and time", "Signal", "Price USDT", "Position size (qty)", "Net P&L USDT"])
            writer.writerow([1, "Exit long", "2024-01-01 00:05", "L", 104.99, 2, 5.79])
            writer.writerow([1, "Entry long", "2024-01-01 00:02", "L", 102.01, 2, 5.79])
        config = {
            "pine": "script.pine", "symbol": "BINANCE:TEST", "timeframe": "1",
            "start": START, "end": START + 29 * 60000, "chart_timezone": "UTC",
            "symbols": {"BINANCE:TEST": {"mintick": .01, "mincontract": .0001, "currency": "USDT"}},
            "feeds": [{"symbol": "BINANCE:TEST", "timeframe": "1", "path": "ohlc.csv"}],
        }

        def run(tag, tv=True, expected=0):
            (d / "config.json").write_text(json.dumps(config))
            args = ["--tv", str(d / "ref.csv")] if tv else []
            completed = subprocess.run([sys.executable, str(HERE / "backtest.py"), "run", str(d / "config.json"),
                                        "--out", str(d / tag), *args], capture_output=True, text=True)
            assert completed.returncode == expected, completed.stderr or completed.stdout
            return json.loads((d / tag / "report.json").read_text()) if expected != 1 else completed.stderr

        for version in (5, 6):
            (d / "script.pine").write_text(SCRIPT.format(version=version))
            report = run(f"v{version}")
            assert report["status"] == "matched_closed_trades", report
            assert abs(report["engine_net"] - 5.7944) < 1e-9
        (d / "ref.csv").write_text((d / "ref.csv").read_text().replace("104.99", "104.98"))
        assert run("different_fill", expected=2)["status"] == "mismatch"

        htf = 's = request.security(syminfo.tickerid, "5", close[1], lookahead=barmerge.lookahead_on)\n'
        stem = SCRIPT.format(version=6)
        mtf_source = stem.replace("if bar_index == 1", htf + "if bar_index == 5 and s == 106")
        (d / "script.pine").write_text(mtf_source.replace("bar_index == 4", "bar_index == 9"))
        mtf = run("mtf", tv=False)
        assert "BINANCE:TEST@5" in mtf["feeds"], mtf
        assert mtf["metrics"]["closed_legs"] == 1, mtf

        (d / "script.pine").write_text(mtf_source.replace("syminfo.tickerid", '"BINANCE:MISSING"'))
        error = run("missing", tv=False, expected=1)
        assert "BINANCE:MISSING@5" in error, error
        (d / "script.pine").write_text(mtf_source.replace("syminfo.tickerid", '"OTHER:TEST"'))
        assert "OTHER:TEST@5" in run("wrong_venue", tv=False, expected=1)
        (d / "script.pine").write_text(stem.replace("slippage=1", "slippage=1, use_bar_magnifier=true"))
        assert "Bar Magnifier" in run("magnifier", tv=False, expected=1)
    print("Pine v5/v6, komisyon/kayma, emir eşleşmesi/farkı, HTF kapanışı, eksik seri ve Bar Magnifier kontrolleri geçti.")


if __name__ == "__main__":
    main()
