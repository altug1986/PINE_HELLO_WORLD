#!/usr/bin/env python3
"""Deliberately narrow Pine v6 EMA-cross to signal CSV adapter.
Accepts ONLY data/ema_example.pine's exact ten-line template shape.
Arbitrary Pine strategies are NOT supported.
"""
import argparse
import csv
import re
from pathlib import Path

HEADER = re.compile(r'^strategy\("[^"]+",\s*overlay=true,\s*process_orders_on_close=true\)$')
EMA = re.compile(r'^(fast|slow)\s*=\s*ta\.ema\(close,\s*(\d+)\)$')

def parse_template(text: str):
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    if len(lines) != 10 or lines[0] != '//@version=6' or not HEADER.fullmatch(lines[1]):
        raise ValueError('Only the documented Pine v6 ten-line EMA template is supported')
    f, s = EMA.fullmatch(lines[2]), EMA.fullmatch(lines[3])
    if not f or not s or f.group(1) != 'fast' or s.group(1) != 'slow':
        raise ValueError('Expected fast/slow ta.ema(close, N)')
    expected = [
        'longSignal = ta.crossover(fast, slow)',
        'exitSignal = ta.crossunder(fast, slow)',
        'if longSignal', 'strategy.entry("L", strategy.long)',
        'if exitSignal', 'strategy.close("L")'
    ]
    if lines[4:] != expected:
        raise ValueError('Unsupported Pine logic: no arbitrary scripts or silent fallbacks')
    nf, ns = int(f.group(2)), int(s.group(2))
    if not (1 <= nf <= 10000 and 1 <= ns <= 10000):
        raise ValueError('EMA lengths out of range')
    return nf, ns

def convert(pine_path: str, ohlc_path: str, output: str):
    nf, ns = parse_template(Path(pine_path).read_text(encoding='utf-8-sig'))
    af, ass = 2.0/(nf+1), 2.0/(ns+1)
    fast = slow = previous = None
    events = []
    with open(ohlc_path, newline='', encoding='utf-8-sig') as inp:
        rd = csv.DictReader(inp)
        if rd.fieldnames != ['timestamp', 'open', 'high', 'low', 'close']:
            raise ValueError('Expected normalized OHLC columns')
        for row in rd:
            close = float(row['close'])
            fast = close if fast is None else af*close+(1-af)*fast
            slow = close if slow is None else ass*close+(1-ass)*slow
            diff = fast-slow
            if previous is not None:
                if previous <= 0 < diff:
                    events.append((row['timestamp'], 'LONG', ''))
                elif previous >= 0 > diff:
                    events.append((row['timestamp'], 'EXIT', ''))
            previous = diff
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w', newline='', encoding='utf-8') as out:
        wr = csv.writer(out)
        wr.writerow(['timestamp', 'action', 'fraction'])
        wr.writerows(events)
    return len(events)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pine', required=True)
    parser.add_argument('--ohlc', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    print(f'{convert(args.pine, args.ohlc, args.out)} signals generated (strict EMA demo only)')
