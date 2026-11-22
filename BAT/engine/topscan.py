import numpy as np, pandas as pd, itertools, sys
import v55
from search57 import build
from v55 import evalp, n
from topfeat import feats
K = dict(x11=2.5, on=0.04, onT=0.15, aT=20.0, S=7.0, rA=4.0, H=648, xH=-1.0, aH=20.0, hm=0, nrMin=0.1, g='yok', gtf='12s', gw=0.5, gth=40, gsym=1, zone=0)
lag = int(sys.argv[1]) if len(sys.argv) > 1 else 1; F = feats(lag); q0 = build(K); rows = []
def ev(GL, GS, ZC, extra, lab):
    v55.GL = GL.astype(np.int64); v55.GS = GS.astype(np.int64); v55.ZC = ZC.astype(np.int64); e = evalp(dict(q0, **extra))
    rows.append(dict(t=lab, **{k: e[k] for k in ['sc', 'net', 'n', 'pf', 'calmar', 'p98', 'mddc', 'IS', 'OOS', 'L12', 'L200', 'P200']}))
Z = np.zeros(n, bool); ev(Z, Z, np.zeros(n), {}, 'BAZ K')
for f, N, k, md in itertools.product(['zTP', 'zTA', 'zDIV', 'zTG'], [2016, 8640], [0.5, 1.0, 1.5, 2.0], ['bizimle', 'karşı']):
    x = np.nan_to_num(F[f'{f}{N}'])
    if md == 'bizimle': GL, GS = x >= k, x <= -k          # balinalar zaten bizim yönde → girme
    else: GL, GS = x <= -k, x >= k                        # balinalar karşı yönde → girme
    ev(GL, GS, np.zeros(n), {}, f'giriş yok: {f}{N} {md} ≥{k}')
for f, N, k, md, xl in itertools.product(['zTP', 'zDIV'], [2016, 8640], [1.0, 1.5, 2.0], ['bizimle', 'karşı'], [1.0, 2.0, 3.0]):
    x = np.nan_to_num(F[f'{f}{N}']); zc = np.where(x >= k, 1, np.where(x <= -k, -1, 0))
    if md == 'bizimle': zc = -zc                          # ZC == -sh → çık: 'bizimle' modunda balinalar bizimleyken çık
    ev(Z, Z, zc, dict(zK1=0, zK2=10**7, zX=xl, zM=99.0, zC=1, zAct=1), f'çık: {f}{N} {md} ≥{k} zarar≥{xl}')
d = pd.DataFrame(rows); d.to_pickle(f'topscan_{lag}.pkl'); pd.set_option('display.width', 250)
b = d.iloc[0]; print(d.iloc[:1].to_string(index=False)); print('bazdan iyi (net, ikinci dönem, son 200 gün üçü birden):', ((d.net > b.net) & (d.OOS > b.OOS) & (d.L200 > b.L200)).sum(), '/', len(d) - 1)
print(d.iloc[1:].sort_values('sc', ascending=False).head(6).to_string(index=False)); print(d.iloc[1:].sort_values('L200', ascending=False).head(6).to_string(index=False))
