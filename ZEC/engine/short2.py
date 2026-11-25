import sys, itertools; sys.path.insert(0, '/home/claude/pine_hello_world/ZEC/engine')
import numpy as np, pandas as pd
src = open('/home/claude/pine_hello_world/ZEC/engine/short_scan.py').read().split("T = {}")[0]; exec(src)
lo = np.log(B.oi_zec.values.astype(float)); ADX = A4[:, 0]; DI = A4[:, 1]
def b1(w=30, zt=2.5, dpk=1.0, adxT=20):
    dd = lo - sh(lo, w); z_ = dd / pd.Series(dd).rolling(2160, min_periods=720).std().values; dp = (c - sh(c, w)) / a
    cond = (z_ >= zt) & (dp <= -dpk) & (ADX >= adxT) & (DI < 0); return cond & ~(sh(cond.astype(float), 1) > 0)
def a3(k=10, adxT=20):
    m = (c - sh(c, 20)) / a; return (m >= k) & (sh(m, 1) < k) & (ADX >= adxT) & (DI < 0)
z = np.zeros(n, np.bool_)
def trade(sig, s, H):
    eq, T_ = z01.sim2(F['o'], h, l, c, a, z, np.nan_to_num(sig).astype(np.bool_), float(s), 99.0, int(H), np.ones(n))
    t = pd.DataFrame(T_, columns=['ei', 'xi', 'side', 'ep', 'xp', 'pnl', 'mae', 'own']); t['y'] = yr[t.ei.astype(int)]
    x30 = eq[::15]; dd = (np.maximum.accumulate(x30) - x30).max(); isk = t.ei < mid
    pf = t.pnl[t.pnl > 0].sum() / -t.pnl[t.pnl < 0].sum() if (t.pnl < 0).any() else np.nan
    yy = t.groupby('y').pnl.sum()
    return dict(n=len(t), net=eq[-1], IS=t.pnl[isk].sum(), OOS=t.pnl[~isk].sum(), PF=pf, MaxDD=dd, posY=int((yy > 0).sum()), yıllar=' '.join(f'{int(v)}' for v in yy.reindex(range(2021, 2027)).fillna(0)), maeMax=t.mae.max() * 100)
print('== İşlem düzeyi (yalnız short, sabit 500, SL ATR / süre bar)')
for lab, sig in (('B1 OI30 yeni short @DN', b1()), ('A3 pump fade ≥10ATR @DN', a3()), ('B1 ∪ A3', b1() | a3())):
    for s_, H in itertools.product([3, 5, 8], [60, 240, 480]):
        r = trade(sig, s_, H); print(f"{lab:24s} SL{s_} H{H:3d}: n {r['n']:4d} net {r['net']:6.0f} IS {r['IS']:5.0f} OOS {r['OOS']:5.0f} PF {r['PF']:.2f} MaxDD {r['MaxDD']:4.0f} yıl+ {r['posY']} [{r['yıllar']}] MAE maks %{r['maeMax']:.1f}")
print('\n== B1 komşuları (SL5 H240): w × z × fiyat ATR × ADX')
R = []
for w, zt, dpk, ad in itertools.product([20, 30, 45], [2.0, 2.5, 3.0], [0.5, 1.0, 1.5], [18, 20, 25]):
    r = trade(b1(w, zt, dpk, ad), 5, 240); R.append(dict(w=w, z=zt, dp=dpk, adx=ad, **{k: r[k] for k in ('n', 'net', 'IS', 'OOS', 'PF', 'posY')}))
R = pd.DataFrame(R); print('pozitif net %', round((R.net > 0).mean() * 100), '| iki yarıda pozitif %', round(((R.IS > 0) & (R.OOS > 0)).mean() * 100), '| medyan net', R.net.median().round(0), '| medyan n', R.n.median())
print(R.sort_values('net', ascending=False).head(6).round(2).to_string(index=False)); print(R.sort_values('net').head(4).round(2).to_string(index=False))
