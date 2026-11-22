import numpy as np, pandas as pd, v55
from stlib import ev, ev2, st5, D, vote_st2, SC, ADX60, AGREE, GL0, GS0, q0, TS, simR, run, t
A30 = np.nan_to_num(D['adx30']); rows = [ev(None, lab='baz V6.3')]
def c(h, f, a, x, lab, ln=10, lag=1, full=False, aT=None):
    s = st5(h, ln, f, lag); return ev2(s * (A30 >= a), s * (A30 >= (aT or a)), x, full=full, lab=lab)
rows += [c(1.5, 3.5, 25, 0.12, 'ADAY 90dk/3.5/ADX25/.12'), c(1.5, 3.5, 25, 0.12, '+1 mum gecikme', lag=2), c(1.5, 3.5, 25, 0.12, 'ATR14', ln=14),
         c(1.0, 3.5, 25, 0.12, '60dk'), c(2.0, 3.5, 25, 0.12, '120dk'), c(1.25, 3.5, 25, 0.12, '75dk'), c(1.5, 3.0, 25, 0.12, 'çarpan 3'), c(1.5, 4.0, 25, 0.12, 'çarpan 4'),
         c(1.5, 3.5, 22, 0.12, 'ADX22'), c(1.5, 3.5, 28, 0.12, 'ADX28'), c(1.5, 3.5, 25, 0.12, 'tam balina kapısı', full=True),
         c(1.5, 2.5, 25, 0.12, 'ADAY-B 90dk/2.5/ADX25-30/.12', aT=30)]
r = pd.DataFrame(rows); pd.set_option('display.width', 250); print(r.to_string(index=False))
# 6 aylık bloklar: aday − baz
def tr(tg): R = simR(D, tg); return pd.Series(R['pnl'], index=pd.to_datetime(t[R['xb']], unit='ms'))
v55.GL = GL0.astype(np.int64); v55.GS = GS0.astype(np.int64); q = dict(q0, **TS)
from stlib import TG0
v55.TG = TG0; b = tr(run(q)[0]); s = st5(1.5, 10, 3.5) * (A30 >= 25)
v55.TG = vote_st2(SC, ADX60, AGREE, s, s, 20.0, 0.04, 0.15, 0.12, -0.15); a = tr(run(q)[0])
d = (a.resample('6MS').sum() - b.resample('6MS').sum()).round(0); print('6 aylık Δ$:', d.astype(int).to_dict()); print('iyileşen blok', int((d > 0).sum()), '/', len(d))
