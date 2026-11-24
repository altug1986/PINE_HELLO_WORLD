import sys, json, itertools; sys.path.insert(0, '/home/claude/pine_hello_world/ZEC/engine')
import numpy as np, pandas as pd
exec(open('/home/claude/zecwork/reg4.py').read().split("PINE = [")[0])            # R1 kurucusu (build2, go, veri)
D = pd.read_csv('/home/claude/zecwork/micro1.csv')
sel = D[(D.h15IS >= 5) & (D.h60IS >= 10) & (D.h240IS >= 0)]                        # SEÇİM YALNIZ IS
print('mikro OI kuralları (IS seçimi):', len(sel), '|', ', '.join(f"{r.kural}{r.yön}@{r.rejim}" for r in sel.itertuples()))
oi = B.oi_zec.values.astype(float); lo = np.log(oi); a = F['atr']
def sh(x, k): return np.r_[np.full(k, np.nan), x[:-k]]
def micro(name):
    if name.startswith('DIV'):
        w = int(name[3:]); hh = pd.Series(c).rolling(w).max().values; ll = pd.Series(c).rolling(w).min().values; d = lo - sh(lo, w)
        return np.where((c >= hh) & (d < -0.005), -1, np.where((c <= ll) & (d < -0.005), 1, 0))
    kind = name[:4]; w, zt = name[4:].split('_'); w = int(w); zt = float(zt)
    d = lo - sh(lo, w); z = d / pd.Series(d).rolling(720 * 3, min_periods=720).std().values; dp = (c - sh(c, w)) / a
    if kind == 'YENİ': return np.where((z >= zt) & (dp >= 1), 1, np.where((z >= zt) & (dp <= -1), -1, 0))
    if kind == 'TASF': return np.where((z <= -zt) & (dp <= -1), 1, np.where((z <= -zt) & (dp >= 1), -1, 0))
    return np.where((z >= zt) & (np.abs(dp) < 0.5), 1, 0) * np.sign(np.nan_to_num(-(c - sh(c, w * 4))))
Rg = np.where(adx >= 20, di, np.where(adx < 20, 0, 9)); Lm = np.zeros(n); Sm = np.zeros(n)
for r in sel.itertuples():
    v = np.nan_to_num(micro(r.kural)); side = 1 if r.yön == 'L' else -1; val = {'UP': 1, 'DN': -1, 'YAT': 0}[r.rejim]
    m = (v == side) & (Rg == val); (Lm if side > 0 else Sm)[m] += 1
L1, S1, _ = build2(20, 25)
def run(wm, kL, kS, ts, tH, ts_s=None):
    L = L1 + wm * Lm; Sh = S1 + wm * Sm; tL = L >= kL; tS = (Sh >= kS) & ~tL
    eq, T = z02.sim4(F['o'], F['h'], F['l'], c, F['atr'], z, z, AT4 * 4, tL.astype(np.bool_), tS.astype(np.bool_), float(ts), int(tH), 99.0)
    t = pd.DataFrame(T, columns=['ei', 'xi', 'side', 'ep', 'xp', 'pnl', 'mae', 'own']); t['et'] = B.index[t.ei.astype(int)]
    return pd.Series(eq, index=B.index), t
rows = []
for wm, kL, kS, ts, tH in itertools.product([0, 1, 2], [2, 3, 4], [1, 2, 3, 99], [4, 6, 8], [120, 240, 480]):
    e, t = run(wm, kL, kS, ts, tH); ev = e.values; A_ = fast(ev, 0, mid); B_ = fast(ev, mid, n); f = fast(ev, 0, n)
    y = t.groupby(t.et.dt.year).pnl.sum()
    rows.append(dict(wm=wm, kL=kL, kS=kS, ts=ts, tH=tH, n=len(t), net=ev[-1], IS=A_[0], OOS=B_[0], MaxDD=f[2], Cal=f[1], ISc=A_[1], L=t[t.side > 0].pnl.sum(), S=t[t.side < 0].pnl.sum(),
                     posY=int((y > 0).sum()), minY=y.min(), maeMax=t.mae.max() * 100))
Rr = pd.DataFrame(rows); Rr.to_csv('/home/claude/zecwork/r2.csv', index=False)
print(Rr.sort_values('ISc', ascending=False).head(12).round(1).to_string(index=False))
print('\n6/6 yıl + IS>0 + OOS>0, Calmar sıralı:'); print(Rr[(Rr.posY == 6) & (Rr.IS > 0) & (Rr.OOS > 0)].sort_values('Cal', ascending=False).head(10).round(1).to_string(index=False))
print('\nen yüksek net (6/6 yıl):'); print(Rr[Rr.posY == 6].sort_values('net', ascending=False).head(6).round(1).to_string(index=False))
