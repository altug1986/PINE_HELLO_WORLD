import numpy as np, pandas as pd
from p1m import build, run, report
from eng import rma, trr
D = build(5); c = D['c']; atr = rma(trr(D['h'], D['l'], c, True), 14) / c * 100; ratio = np.nan_to_num(atr / pd.Series(atr).rolling(8640, min_periods=2000).mean().to_numpy())
rows = [report(D, run(D), 'V6.6 baz')]
for thr in (1.5, 2.0, 2.5):
    for f in (0.95, 0.9, 0.85): rows.append(report(D, run(D, hi=ratio >= thr, hfac=f), f'ATR ≥ {thr}x · eşik ×{f}'))
hi = ratio >= 2.0; rows.append(report(D, run(D, whale='tv', hi=hi, hfac=0.9), 'ATR ≥ 2x · ×0.9 · TV benzeri balina')); rows.append(report(D, run(D, whale='tv'), 'V6.6 baz · TV benzeri balina'))
pd.set_option('display.width', 250); print(pd.DataFrame(rows).to_string(index=False))
