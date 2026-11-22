import numpy as np, pandas as pd, itertools, time, json
from r5 import g_lk, mae
from sysB import *
from scipy.ndimage import uniform_filter
DI = cut(load(5), SPLIT); FI = feats(DI)
G = dict(aL=[3, 3.5, 4, 4.5, 5, 5.5, 6], aS=[4, 4.5, 5, 5.5, 6, 6.5, 7], hold=[288, 360, 432, 504, 576, 648, 720], sl=[10, 12, 15, 20, 25])
A = np.zeros([len(v) for v in G.values()]); N = np.zeros_like(A)
for idx in itertools.product(*[range(len(v)) for v in G.values()]):
    s = {k: G[k][i] for k, i in zip(G, idx)}; sg = sig2(FI, DI, dict(H=288, **s))
    x = metrics(DI, simR(DI, g_lk(DI['c'], sg, s['hold'], s['sl']))); A[idx] = sc(x); N[idx] = x['net']
Sm = uniform_filter(A, size=3, mode='nearest'); i1 = np.unravel_index(A.argmax(), A.shape); i2 = np.unravel_index(Sm.argmax(), A.shape)
for lab, i in [('tepe', i1), ('plato merkezi', i2)]: print(lab, {k: G[k][j] for k, j in zip(G, i)}, 'IS net', N[i], 'skor', round(A[i]), 'komşu ort', round(Sm[i]))
print('IS pozitif oran', (N > 0).mean().round(3))
json.dump({k: G[k][j] for k, j in zip(G, i2)}, open('r6.json', 'w'))
