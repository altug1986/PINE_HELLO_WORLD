import numpy as np, pandas as pd, itertools, time
from sysB import *
from scipy.ndimage import uniform_filter
DI = cut(load(5), SPLIT); FI = feats(DI)
G = dict(aL=[3, 3.5, 4, 4.5, 5, 5.5, 6], aS=[4, 4.5, 5, 5.5, 6, 6.5, 7], hold=[288, 360, 432, 504, 576, 648, 720], sl=[3, 4, 5, 6, 8, 10])
A = np.zeros([len(v) for v in G.values()]); N = np.zeros_like(A); P98 = np.zeros_like(A); t0 = time.time()
for idx in itertools.product(*[range(len(v)) for v in G.values()]):
    s = dict(fam='PXA', H=288, **{k: G[k][i] for k, i in zip(G, idx)}); x = metrics(DI, sysR2(DI, FI, ['PXA'], dict(PXA=s)))
    A[idx] = sc(x); N[idx] = x['net']; P98[idx] = x['p98']
print(f'{time.time()-t0:.0f}s')
Sm = uniform_filter(A, size=3, mode='nearest'); i1 = np.unravel_index(A.argmax(), A.shape); i2 = np.unravel_index(Sm.argmax(), A.shape)
for lab, i in [('tepe', i1), ('plato merkezi', i2)]:
    print(lab, {k: G[k][j] for k, j in zip(G, i)}, 'IS net', N[i], 'P98', P98[i], 'skor', round(A[i]), 'komşu ort skor', round(Sm[i]))
print('IS pozitif oranı tüm ızgara', (N > 0).mean().round(3))
np.savez('r3.npz', A=A, N=N, P98=P98)
