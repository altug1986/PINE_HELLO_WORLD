import itertools, time, numpy as np, pandas as pd
from eng import *
rows = []; t0 = time.time()
def rec(fam, prm, Ls, Ss, hold=0):
    _, f, i, o = evaluate(Ls, Ss, hold)
    rows.append(dict(fam=fam, **prm, **{f'F_{k}': v for k, v in f.items()}, **{f'I_{k}': v for k, v in i.items()}, **{f'O_{k}': v for k, v in o.items()}))
def lag(x, k): return np.concatenate([np.full(k, np.nan), x[:-k]])
a30, di30 = adx_htf(30); a60, di60 = adx_htf(60); p5, m5, a5 = adx5()
tr = np.maximum(H - L_, np.abs(H - lag(C, 1)), np.abs(L_ - lag(C, 1))); tr[0] = H[0] - L_[0]
atr = rma(tr, 14)
# G1 dense neighbour grid around core
for A30, A5, n, m, rl, hold in itertools.product([24, 26, 28, 30, 32, 34, 36, 40], [0, 15, 20, 25], [3, 4, 5, 6, 8],
                                                 [2, 2.5, 3, 3.5, 4, 5], [0, 1, 3, 6, 12], [0, 24]):
    mo = mom(n); g = (a30 > A30) & (a5 > A5)
    if rl: g &= a5 > lag(a5, rl)
    rec('CORE', dict(A30=A30, A5=A5, n=n, m=m, rl=rl, hold=hold), g & (mo >= m), g & (mo <= -m), hold)
print('G1', len(rows), round(time.time() - t0))
# G2 ATR-normalised momentum (portable): (C-C[n])/ATR >= k
for A30, n, k, rl in itertools.product([24, 26, 28, 30, 32, 34, 36], [3, 4, 5, 6, 8], [3, 4, 5, 6, 7, 8, 10], [0, 3]):
    z = (C - lag(C, n)) / atr; g = a30 > A30
    if rl: g &= a5 > lag(a5, rl)
    rec('ATRMOM', dict(A30=A30, n=n, k=k, rl=rl), g & (z >= k), g & (z <= -k))
# G3 single changes on core (A30 30, n4, m3, rl3)
mo = mom(4); g0 = (a30 > 30) & (a5 > lag(a5, 3))
rec('X', dict(x='core'), g0 & (mo >= 3), g0 & (mo <= -3))
rec('X', dict(x='DI30 agree'), g0 & (mo >= 3) & (di30 > 0), g0 & (mo <= -3) & (di30 < 0))
rec('X', dict(x='DI5 agree'), g0 & (mo >= 3) & (p5 > m5), g0 & (mo <= -3) & (p5 < m5))
rec('X', dict(x='A30 rising'), g0 & (a30 > lag(a30, 6)) & (mo >= 3), g0 & (a30 > lag(a30, 6)) & (mo <= -3))
rec('X', dict(x='ADX60 instead'), (a60 > 30) & (a5 > lag(a5, 3)) & (mo >= 3), (a60 > 30) & (a5 > lag(a5, 3)) & (mo <= -3))
for ml, ms in itertools.product([2, 2.5, 3, 3.5, 4], [2, 2.5, 3, 3.5, 4]):
    rec('ASYM', dict(ml=ml, ms=ms), g0 & (mo >= ml), g0 & (mo <= -ms))
print('all', len(rows), round(time.time() - t0))
pd.DataFrame(rows).to_pickle('/home/claude/work/scan2.pkl')
