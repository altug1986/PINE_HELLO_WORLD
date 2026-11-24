import sys, itertools, time; sys.path.insert(0, '/home/claude/pine_hello_world/ZEC/engine')
import numpy as np, pandas as pd
from zec_data import load; import z01
B = load(2, src='/home/claude/master/'); F = z01.prep(B)
mid = B.index[0] + (B.index[-1] - B.index[0]) / 2; rows = []; t0 = time.time()
for bars, k, s, m, H in itertools.product([10, 20, 30], [6, 8, 10, 12], [4, 6, 10], [6, 10, 99], [60, 120, 240, 480]):
    eq, t = z01.run2(B, F, bars, k, s, m, H)
    a = z01.metrics(eq, t, None, mid); b = z01.metrics(eq, t, mid, None); f = z01.metrics(eq, t)
    rows.append(dict(bars=bars, k=k, s=s, m=m, H=H, IS=a['net'], IS_cal=a['Calmar'], OOS=b['net'], OOS_cal=b['Calmar'], net=f['net'], n=f['n'],
                     PF=f['PF'], MaxDD=f['MaxDD'], P99=f['P99'], Cal=f['Calmar'], bps=f['bps'], mae90=f['maeP90']))
R = pd.DataFrame(rows); R.to_csv('/home/claude/zecwork/grid02.csv', index=False); print('süre', round(time.time() - t0), 's')
print(R.sort_values('IS_cal', ascending=False).head(12).round(2).to_string(index=False))
print('IS>0 & OOS>0 oranı', round(((R.IS > 0) & (R.OOS > 0)).mean() * 100, 1), '%')
print(R.groupby(['bars', 'k'])[['IS', 'OOS', 'n']].median().round(0).to_string())
