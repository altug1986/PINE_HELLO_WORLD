import sys, json, itertools; sys.path.insert(0, '/home/claude/pine_hello_world/ZEC/engine')
import numpy as np, pandas as pd
exec(open('/home/claude/zecwork/r2.py').read().split("def run(")[0])
import importlib; importlib.reload(z02)
tA_L = (L1 >= 2); tA_S = (S1 >= 3) & ~tA_L
rows = []
for kmL, kmS, bts, btH in itertools.product([1, 2, 3], [1, 2, 3, 99], [2, 3, 5], [15, 30, 60, 120]):
    bL = (Lm >= kmL); bS = (Sm >= kmS) & ~bL
    eq, T = z02.sim5(F['o'], F['h'], F['l'], c, F['atr'], tA_L, tA_S, 8.0, 480, bL, bS, float(bts), int(btH))
    t = pd.DataFrame(T, columns=['ei', 'xi', 'side', 'ep', 'xp', 'pnl', 'mae', 'own']); t['et'] = B.index[t.ei.astype(int)]
    A_ = fast(eq, 0, mid); B_ = fast(eq, mid, n); f = fast(eq, 0, n); y = t.groupby(t.et.dt.year).pnl.sum()
    rows.append(dict(kmL=kmL, kmS=kmS, bts=bts, btH=btH, n=len(t), nB=int((t.own == 2).sum()), net=eq[-1], IS=A_[0], OOS=B_[0], MaxDD=f[2], Cal=f[1], ISc=A_[1],
                     A=t[t.own == 1].pnl.sum(), Bp=t[t.own == 2].pnl.sum(), BL=t[(t.own == 2) & (t.side > 0)].pnl.sum(), BS=t[(t.own == 2) & (t.side < 0)].pnl.sum(), posY=int((y > 0).sum()), minY=y.min()))
Rr = pd.DataFrame(rows); Rr.to_csv('/home/claude/zecwork/r2b.csv', index=False)
print('BAZ R1 (A tek başına, kS3): 4.221 net, MaxDD 443')
print(Rr.sort_values('ISc', ascending=False).head(12).round(1).to_string(index=False))
print('\n6/6 yıl + IS>0 + OOS>0, net sıralı:'); print(Rr[(Rr.posY == 6) & (Rr.IS > 0) & (Rr.OOS > 0)].sort_values('net', ascending=False).head(8).round(1).to_string(index=False))
