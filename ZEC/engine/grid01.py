import sys, itertools, time; sys.path.insert(0, '/home/claude/pine_hello_world/ZEC/engine')
import numpy as np, pandas as pd
from zec_data import load; import z01
B = load(2, src='/home/claude/master/'); F = z01.prep(B)
mid = B.index[0] + (B.index[-1] - B.index[0]) / 2
rows = []; t0 = time.time()
for mode in ['both', 'E1', 'E2']:
    for N1, k1, v1, N2, m, s in itertools.product([30, 60, 120], [1.5, 2.5, 3.5], [2, 4], [90, 180, 360], [4, 6, 9, 13], [4, 8]):
        if mode == 'E2' and (N1, k1, v1) != (60, 2.5, 2): continue
        if mode == 'E1' and N2 != 180: continue
        eq, t = z01.run(B, F, N1, k1, v1, N2, m, s, mode != 'E2', mode != 'E1')
        a = z01.metrics(eq, t, None, mid); b = z01.metrics(eq, t, mid, None); f = z01.metrics(eq, t)
        rows.append(dict(mode=mode, N1=N1, k1=k1, v1=v1, N2=N2, m=m, s=s, IS=a['net'], IS_cal=a['Calmar'], OOS=b['net'], OOS_cal=b['Calmar'],
                         net=f['net'], n=f['n'], PF=f['PF'], MaxDD=f['MaxDD'], P99=f['P99'], Cal=f['Calmar'], bps=f['bps']))
R = pd.DataFrame(rows); R.to_csv('/home/claude/zecwork/grid01.csv', index=False)
print('süre', round(time.time() - t0), 's', len(R))
for mode in ['both', 'E1', 'E2']:
    r = R[R['mode'] == mode].sort_values('IS_cal', ascending=False)
    print(mode); print(r.head(6).round(2).to_string(index=False))
