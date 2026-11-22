import numpy as np, pandas as pd, itertools, v55
from tecotest import m30layers, runR, WL, WS, q0, D, simR, run, n
v55.GL = WL.astype(np.int64); v55.GS = WS.astype(np.int64)
def go(p, lab):
    R = simR(D, run(dict(q0, **p))[0]); o = m30layers(R, lab); L = R['pnl'] <= 0; o['kaybeden süre medyan'] = int(np.median((R['xb'] - R['eb'])[L])); o['işlem'] = len(R['pnl']); return o
rows = [go({}, 'V6.1')]
for H, x, aH in itertools.product([96, 144, 192, 288, 432], [-0.5, -1.0, -2.0, -3.0], [0.0, 25.0]):
    rows.append(go(dict(H=H, xH=x, aH=aH, rH=4.0), f'{H} bar sonra hâlâ ≥%{-x:g} zarardaysa kapat' + (f' (15dk ADX<{aH:g})' if aH else '')))
r = pd.DataFrame(rows); r.to_pickle('earlycut.pkl'); b = r.iloc[0]
C = ['t', 'işlem', 'kaybeden süre medyan', 'TAM net', 'TAM MaxDD', 'TAM Calmar', 'NORMAL net', 'NORMAL MaxDD', 'NORMAL Calmar', 'BÜYÜK net', 'son200 NORMAL']
pd.set_option('display.width', 250); print(r.iloc[:1][C].to_string(index=False))
print('V6.1\'den iyi (TAM net + NORMAL net + NORMAL Calmar):', ((r['TAM net'] > b['TAM net']) & (r['NORMAL net'] > b['NORMAL net']) & (r['NORMAL Calmar'] > b['NORMAL Calmar'])).sum(), '/', len(r) - 1)
print(r.iloc[1:].sort_values('NORMAL Calmar', ascending=False).head(6)[C].to_string(index=False))
