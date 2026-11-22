import itertools, time, pickle, numpy as np, pandas as pd
from eng import *
rows = []; t0 = time.time()
def rec(fam, prm, Ls, Ss, hold=0):
    _, f, i, o = evaluate(Ls, Ss, hold)
    rows.append(dict(fam=fam, **prm, **{f'F_{k}': v for k, v in f.items()}, **{f'I_{k}': v for k, v in i.items()}, **{f'O_{k}': v for k, v in o.items()}))

# sanity: user base V1 (30m ADX>20, 5m ADX>25, 4-bar mom 1%)
a30, _ = adx_htf(30); _, _, a5 = adx5()
# F1 ADX-MOM (user family) + rise/confirm/hold
for A30, A5, n, m, cf, rise, hold in itertools.product([0, 20, 25, 30], [0, 20, 25, 30], [4, 8, 12, 24, 48, 96],
                                                       [1, 1.5, 2, 3, 4, 6], [1, 2], [0, 1], [0, 24, 96]):
    mo = mom(n); g = (a30 > A30) & (a5 > A5)
    if rise: g &= a5 > np.concatenate([np.full(3, np.nan), a5[:-3]])
    rec('ADXMOM', dict(A30=A30, A5=A5, n=n, m=m, cf=cf, rise=rise, hold=hold), consec(g & (mo >= m), cf), consec(g & (mo <= -m), cf), hold)
print('F1', len(rows), round(time.time() - t0))
# F2 Donchian SAR (5m bars) + optional 30m ADX
for n, A30 in itertools.product([48, 96, 144, 192, 288, 432, 576, 864, 1152, 1728, 2304], [0, 20, 25]):
    hh, ll = donch(n); g = a30 > A30
    rec('DONCH', dict(n=n, A30=A30), g & (C > hh), g & (C < ll))
# F3 Supertrend flip (5m / HTF)
for tf, ap, f in itertools.product([5, 30, 60, 240], [10, 14, 22, 34, 50], [1.5, 2, 2.5, 3, 4, 5, 6, 8]):
    d = st_htf(tf, f, ap); rec('ST', dict(tf=tf, ap=ap, f=f), d == -1, d == 1)
# F4 EMA cross state
for tf, fa, sl in itertools.product([5, 30, 60, 240], [5, 9, 12, 20, 34, 50], [21, 50, 100, 200, 300]):
    if fa >= sl: continue
    ef, es = ema_htf(tf, fa), ema_htf(tf, sl); rec('EMA', dict(tf=tf, fa=fa, sl=sl), ef > es, ef < es)
# F5 DI direction on HTF with ADX gate
for tf, A in itertools.product([30, 60, 240], [0, 15, 20, 25, 30]):
    a, di = adx_htf(tf); g = a > A; rec('DI', dict(tf=tf, A=A), g & (di > 0), g & (di < 0))
print('all', len(rows), round(time.time() - t0))
R = pd.DataFrame(rows); R.to_pickle('/home/claude/work/scan1.pkl')
