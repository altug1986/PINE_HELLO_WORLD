import sys, json, itertools; sys.path.insert(0, '/home/claude/pine_hello_world/ZEC/engine')
import numpy as np, pandas as pd
exec(open('/home/claude/zecwork/r2.py').read().split("def run(")[0])
import importlib; importlib.reload(z02)
eL = (L1 >= 2); eS = (S1 >= 3) & ~eL; zf = np.zeros(n, np.bool_)
up = (Rg == 1)
tas = np.nan_to_num(micro('TASF8_1.5')); yen = np.nan_to_num(micro('YENİ30_2.5'))
XL = {'yok': zf, 'OIkarşı': (yen == -1) | (tas == -1)}; XS = {'yok': zf, 'OIkarşı': (yen == 1) | (tas == 1)}
def run(s=8, H=480, act=0, trl=0, tp=0, cap=0, trendUP=False, xo='yok'):
    eq, T = z02.sim6(F['o'], F['h'], F['l'], c, F['atr'], eL.astype(np.bool_), eS.astype(np.bool_), float(s), int(H), float(act), float(trl), float(tp), float(cap),
                     (up if trendUP else zf).astype(np.bool_), AT4 * 4, XL[xo].astype(np.bool_), XS[xo].astype(np.bool_))
    return eq, pd.DataFrame(T, columns=['ei', 'xi', 'side', 'ep', 'xp', 'pnl', 'mae', 'mfe', 'tr'])
eq, t = run(); print(f"BAZ R2: net {eq[-1]:.0f} n {len(t)} | MFE medyan %{t.mfe.median()*100:.2f} P90 %{t.mfe.quantile(.9)*100:.1f} | gerçekleşen medyan %{(t.pnl/5).median():.2f} | yakalanan (kazananlarda) %{(t.pnl[t.pnl>0]/5/(t.mfe[t.pnl>0]*100)).median()*100:.0f} | MAE P90 %{t.mae.quantile(.9)*100:.1f} maks %{t.mae.max()*100:.1f}")
rows = []
for s, cap, H, (act, trl), tp, tu, xo in itertools.product([6, 8], [0, 0.06, 0.10], [480, 960], [(0, 0), (3, 3), (5, 4), (8, 6)], [0, 12], [False, True], ['yok', 'OIkarşı']):
    eq, t = run(s, H, act, trl, tp, cap, tu, xo); A_ = fast(eq, 0, mid); B_ = fast(eq, mid, n); f = fast(eq, 0, n)
    y = pd.Series(t.pnl.values, index=B.index[t.ei.astype(int)]).groupby(lambda x: x.year).sum()
    rows.append(dict(s=s, cap=cap, H=H, act=act, trl=trl, tp=tp, trendUP=tu, xo=xo, n=len(t), net=eq[-1], IS=A_[0], OOS=B_[0], MaxDD=f[2], Cal=f[1], ISc=A_[1],
                     maeMax=t.mae.max() * 100, maeP90=t.mae.quantile(.9) * 100, posY=int((y > 0).sum()), minY=y.min(), trPnl=t[t.tr == 1].pnl.sum()))
R = pd.DataFrame(rows); R.to_csv('/home/claude/zecwork/exit1.csv', index=False)
print(R.sort_values('ISc', ascending=False).head(12).round(1).to_string(index=False))
print('\n6/6 yıl, IS>0, OOS>0, maks ters gidiş ≤ %15 — net sıralı:'); print(R[(R.posY == 6) & (R.IS > 0) & (R.OOS > 0) & (R.maeMax <= 15)].sort_values('net', ascending=False).head(8).round(1).to_string(index=False))
print('\n6/6 yıl, IS>0, OOS>0 — net sıralı:'); print(R[(R.posY == 6) & (R.IS > 0) & (R.OOS > 0)].sort_values('net', ascending=False).head(8).round(1).to_string(index=False))
