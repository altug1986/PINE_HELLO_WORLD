import itertools, time, numpy as np, pandas as pd
from eng import *
rows = []; t0 = time.time()
def lag(x, k): return np.concatenate([np.full(k, np.nan), x[:-k]])
def rec(fam, prm, Ls, Ss, hold=0):
    _, f, i, o = evaluate(Ls, Ss, hold)
    rows.append(dict(fam=fam, **prm, **{f'F_{k}': v for k, v in f.items()}, **{f'I_{k}': v for k, v in i.items()}, **{f'O_{k}': v for k, v in o.items()}))
a30, _ = adx_htf(30); _, _, a5 = adx5()
tr = np.maximum(np.maximum(H - L_, np.abs(H - lag(C, 1))), np.abs(L_ - lag(C, 1))); tr[0] = H[0] - L_[0]
# G1 VOLMOM: momentum threshold scaled by slow volatility baseline (ATR% over L bars)
for Lb in [288, 864, 2016]:
    vb = rma(tr, Lb) / C * 100
    for n, k, A30, A5, rl, hold in itertools.product([3, 4, 6, 8, 12], [3, 4, 5, 6, 8, 10], [0, 20, 25], [0, 20], [0, 6, 12], [0, 12, 24]):
        z = mom(n) / vb; g = (a30 > A30) & (a5 > A5)
        if rl: g &= a5 > lag(a5, rl)
        rec('VOLMOM', dict(Lb=Lb, n=n, k=k, A30=A30, A5=A5, rl=rl, hold=hold), g & (z >= k), g & (z <= -k), hold)
print('G1', len(rows), round(time.time() - t0))
# G2 OR-vote: V2 core engine OR slower second engine (same gate family)
gA = (a30 > 28) & (a5 > 20) & (a5 > lag(a5, 12)); mA = mom(4)
LA, SA = gA & (mA >= 3), gA & (mA <= -3)
for A30, n, m, rl, hold in itertools.product([0, 20, 25, 30], [8, 12, 24, 48, 96], [1, 1.5, 2, 3, 4], [0, 6, 12], [0, 24]):
    g = (a30 > A30) & ((a5 > lag(a5, rl)) if rl else True); mo = mom(n)
    rec('OR', dict(A30=A30, n=n, m=m, rl=rl, hold=hold), LA | (g & (mo >= m)), SA | (g & (mo <= -m)), hold)
print('all', len(rows), round(time.time() - t0))
pd.DataFrame(rows).to_pickle('/home/claude/work/scan3.pkl')
