import numpy as np, numba as nb, pandas as pd, itertools
from v54 import *
from stopre import mae
# AKILLI ERKEN KESİM: zarar ≥ L ve BAT/BTC 30 dk hacim oranı ≥ vk (ve isteğe bağlı BTC 1S ADX > a, DI ters) → FLAT + kilit (geri giriş %rA tepkiyle)
@nb.njit(cache=True)
def lay2(c, tg, BADX, BDI, NR, S, rA, L, vk, a, tick):
    n = len(c); out = np.zeros(n, np.int64); sh = 0; ep = 0.0; lock = 0; lext = 0.0; prevT = 0; ne = 0
    for i in range(n):
        t_ = tg[i]
        if t_ != prevT: lock = 0; prevT = t_
        want = t_
        if sh != 0 and want == sh:
            loss = (c[i] / ep - 1) * sh * 100
            early = L < 50 and loss <= -L and NR[i] >= vk and (a <= 0 or (BADX[i] > a and BDI[i] * sh < 0))
            if loss <= -S or early:
                lock = sh; lext = c[i]; want = 0
                if early and loss > -S: ne += 1
        if lock != 0 and want == lock:
            if lock > 0: lext = min(lext, c[i])
            else: lext = max(lext, c[i])
            if (c[i] / lext - 1) * lock * 100 >= rA: lock = 0
            else: want = 0
        if want != sh: sh = want; ep = c[i] + tick * want
        out[i] = sh
    return out, ne
k0 = np.searchsorted(t, SPLIT); km = np.searchsorted(t, np.int64(pd.Timestamp('2022-06-15').value // 10**6))
def cutD(a_, b_): return {kk: (v[a_:b_] if isinstance(v, np.ndarray) and len(v) == n else v) for kk, v in D.items()}
H = [cutD(0, km), cutD(km, k0)]
def halves(tg):
    out = []
    for (a_, b_), Dh in zip([(0, km), (km, k0)], H):
        m = metrics(Dh, simR(Dh, tg[a_:b_])); out.append(m['net'] - 1.5 * m['p98'])
    return min(out), out
tgB = target(); rows = []
for L, vk, a in itertools.product([2.0, 3.0, 4.0, 5.0], [1.2, 1.5, 2.0, 3.0], [0.0, 25.0, 35.0]):
    tg, ne = lay2(c, tgB, BADX, BDI, NR, S0, RA0, L, vk, a, TICK); s, hs = halves(tg); rows.append(dict(L=L, vk=vk, a=a, erken=ne, sc=s, A=hs[0], B=hs[1]))
df = pd.DataFrame(rows); A0, B0 = 2256, 1955; ok = df[(df.A > A0) & (df.B > B0)]
print('varyant', len(df), '| iki yarıda V5.3A üstü:', len(ok)); pd.set_option('display.width', 250); print(df.sort_values('sc', ascending=False).head(5).to_string(index=False))
K2 = ['net', 'n', 'pf', 'sharpe', 'calmar', 'p98', 'ulcer', 'IS', 'OOS', 'last12']
for _, r in (ok if len(ok) else df).sort_values('sc', ascending=False).head(3).iterrows():
    tg, ne = lay2(c, tgB, BADX, BDI, NR, S0, RA0, r.L, r.vk, r.a, TICK); R = simR(D, tg); m = metrics(D, R); print('TAM', dict(L=r.L, vk=r.vk, a=r.a), {k: m[k] for k in K2}, 'MAE', mae(R), 'erken kesim', ne)
