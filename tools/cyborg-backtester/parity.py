"""Read TradingView closed trades and compare each entry/exit pair."""
import csv
from collections import defaultdict
from math import isclose
from pathlib import Path
from unicodedata import normalize, combining

from market_data import input_rows, number, timestamp


def norm(value):
    text = normalize("NFKD", str(value or "").casefold().replace("ı", "i"))
    return "".join(c for c in text if c.isalnum() and not combining(c))


def read_trade_rows(rows, zone):
    header = [norm(x) for x in next(rows, ())]
    if {"entrytimems", "exittimems", "side", "entryprice", "exitprice", "qty", "pnl"} <= set(header):
        col = {v: header.index(v) for v in header}
        return [
            dict(side=str(row[col["side"]]).casefold(), entry_time=int(row[col["entrytimems"]]),
                 exit_time=int(row[col["exittimems"]]), entry_price=number(row[col["entryprice"]]),
                 exit_price=number(row[col["exitprice"]]), qty=abs(number(row[col["qty"]])),
                 pnl=number(row[col["pnl"]]))
            for row in rows if row and row[col["side"]]
        ], 0

    def find(*names):
        for i, key in enumerate(header):
            if any(key.startswith(name) for name in names):
                return i
        raise ValueError(f"TradingView işlem sütunu eksik: {names[0]}")

    id_col = find("tradeno", "tradenumber", "islemnumarasi", "trade")
    kind_col = find("type", "tip")
    date_col = find("dateandtime", "datetime", "tarihvesaat")
    price_col = find("price", "fiyat")
    qty_col = find("positionsizeqty", "boyutmiktar", "positionqty", "qty", "contracts")
    pnl_col = find("netpnl", "netpl", "netkz", "profit")
    grouped = defaultdict(dict)
    for row in rows:
        if not row or row[id_col] in ("", None):
            continue
        kind = norm(row[kind_col])
        entry = "entry" in kind or "giris" in kind
        exit_ = "exit" in kind or "cikis" in kind
        if entry == exit_:
            raise ValueError(f"Tanımsız TradingView işlem tipi: {row[kind_col]}")
        leg = "entry" if entry else "exit"
        group = grouped[str(row[id_col])]
        if leg in group:
            raise ValueError(f"TradingView işlem numarasında tekrarlı {leg}: {row[id_col]}")
        if "long" in kind or "uzun" in kind:
            side = "long"
        elif "short" in kind or "kisa" in kind:
            side = "short"
        else:
            raise ValueError(f"TradingView pozisyon yönü belirsiz: {row[kind_col]}")
        group[leg] = (row, side)
    trades = []
    open_count = 0
    for trade_id, legs in grouped.items():
        if "entry" not in legs:
            raise ValueError(f"TradingView giriş satırı yok: {trade_id}")
        if "exit" not in legs:
            open_count += 1
            continue
        entry, side = legs["entry"]
        exit_, exit_side = legs["exit"]
        if norm(exit_[date_col]) in {"open", "acik"}:
            open_count += 1
            continue
        opened = timestamp(entry[date_col], zone)
        closed = timestamp(exit_[date_col], zone)
        if side != exit_side or opened > closed:
            raise ValueError(f"TradingView işlem yönü veya zaman sırası tutarsız: {trade_id}")
        trades.append(dict(side=side, entry_time=opened, exit_time=closed, entry_price=number(entry[price_col]),
                           exit_price=number(exit_[price_col]), qty=abs(number(exit_[qty_col])),
                           pnl=number(exit_[pnl_col]), trade_id=trade_id))
    return trades, open_count


def load_reference(path, zone):
    path = Path(path)
    if path.suffix.lower() == ".xlsx":
        from openpyxl import load_workbook

        book = load_workbook(path, read_only=True, data_only=True)
        try:
            names = {norm(name): name for name in book.sheetnames}
            name = next((names[k] for k in ("listoftrades", "islemler", "trades", "islemlistesi") if k in names), None)
            if not name:
                raise ValueError(f"TradingView işlem sayfası bulunamadı: {book.sheetnames}")
            trades, open_count = read_trade_rows(book[name].values, zone)
            props_sheet = next((names[k] for k in ("properties", "ozellikler") if k in names), None)
            props = dict((str(r[0]), str(r[1])) for r in book[props_sheet].values if len(r) > 1 and r[0] and r[1] is not None) if props_sheet else {}
            return trades, open_count, props
        finally:
            book.close()
    with input_rows(path) as rows:
        trades, open_count = read_trade_rows(rows, zone)
        return trades, open_count, {}


def write_trades(path, trades):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["entry_time_ms", "exit_time_ms", "side", "entry_price", "exit_price", "qty", "pnl", "commission", "entry_id", "exit_id"])
        for t in trades:
            writer.writerow([t.get(key) for key in ("entry_time", "exit_time", "side", "entry_price", "exit_price", "qty", "pnl", "commission", "entry_id", "exit_id")])


def compare(engine, reference, price_tol=1e-8, qty_tol=1e-9, pnl_tol=0.005001):
    def key(t):
        return t["entry_time"], t["exit_time"], t["side"]

    def match(t, other):
        return all(isclose(t[k], other[k], abs_tol=tol, rel_tol=0)
                   for k, tol in (("entry_price", price_tol), ("exit_price", price_tol),
                                  ("qty", qty_tol), ("pnl", pnl_tol)))

    actual = defaultdict(list)
    for t in engine:
        actual[key(t)].append(t)
    unmatched = []
    accepted = 0
    for expected in reference:
        group = actual[key(expected)]
        for i, t in enumerate(group):
            if match(t, expected):
                group.pop(i)
                accepted += 1
                break
        else:
            unmatched.append(expected)
    extra = [t for group in actual.values() for t in group]
    return {
        "status": "matched_closed_trades" if reference and not unmatched and not extra else "mismatch" if reference else "no_reference_trades",
        "engine_closed": len(engine), "reference_closed": len(reference), "matched": accepted,
        "reference_unmatched": len(unmatched), "engine_unmatched": len(extra),
        "engine_pnl": sum(t["pnl"] for t in engine), "reference_rounded_pnl_sum": sum(t["pnl"] for t in reference),
        "tolerances": {"time_ms": 0, "price": price_tol, "quantity": qty_tol, "trade_pnl": pnl_tol},
        "first_reference_unmatched": unmatched[:20], "first_engine_unmatched": extra[:20],
    }
