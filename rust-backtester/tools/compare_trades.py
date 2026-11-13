#!/usr/bin/env python3
"""Compare normalized TradingView trades to local Rust trades, first mismatch wins.

Both CSV files must contain entry_time,exit_time,side,qty,entry_price,exit_price,net_pnl.
TradingView native exported headings must be normalized independently.
"""
import argparse
import csv
import math

FIELDS=['entry_time','exit_time','side','qty','entry_price','exit_price','net_pnl']
def read(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        rd=csv.DictReader(f)
        if not all(k in (rd.fieldnames or []) for k in FIELDS):
            raise ValueError(f'{path} is not normalized: {FIELDS}')
        return list(rd)

def compare(tv, engine, absolute, relative):
    if len(tv)!=len(engine):
        return f'Trade count differs: TV={len(tv)}, Rust={len(engine)}'
    for index,(left,right) in enumerate(zip(tv,engine),1):
        for field in FIELDS:
            if field in ('entry_time','exit_time','side'):
                valid=left[field].strip()==right[field].strip()
            else:
                try:
                    l=float(left[field]);r=float(right[field])
                    valid=math.isfinite(l) and math.isfinite(r) and math.isclose(l,r,abs_tol=absolute,rel_tol=relative)
                except ValueError:
                    valid=False
            if not valid:
                return f'First mismatch trade #{index}, {field}: TV={left[field]!r} Rust={right[field]!r}'
    return None

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tv',required=True)
    ap.add_argument('--engine',required=True)
    ap.add_argument('--abs',type=float,default=1e-6)
    ap.add_argument('--rel',type=float,default=1e-6)
    x=ap.parse_args()
    tv,engine=read(x.tv),read(x.engine)
    err=compare(tv,engine,x.__dict__['abs'],x.rel)
    if err:
        print('PARITY FAIL:',err)
        raise SystemExit(1)
    print(f'PARITY PASS: {len(tv)} normalized closed trades (selected fields only)')
