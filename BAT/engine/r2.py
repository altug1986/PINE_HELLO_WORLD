import numpy as np, pandas as pd, json, itertools
from sysB import *
from ex import g_ex, XG
D0 = load(5); F0 = feats(D0); DI = cut(D0, SPLIT); FI = feats(DI)
V1 = dict(fam='PXA', H=288, aL=4, aS=5.5, hold=576, sl=5)
def met(D, F, S, o=['PXA']): x = metrics(D, sysR2(D, F, o, S)); return {k: x[k] for k in ['net', 'n', 'pf', 'p98', 'IS', 'OOS']}
# (a) gün katı ufuk hipotezi
for H in [144, 288, 432, 576, 864]:
    print('H', H, met(D0, F0, dict(PXA=dict(V1, H=H))))
# (b) V1 üstünde çıkış aileleri (IS)
c = DI['c']; s = sig2(FI, DI, V1); ref = np.r_[np.full(288, np.nan), c[:-288]]; ref = np.where(np.isnan(ref), c, ref)
o = np.r_[np.nan, DI['oi'][:-1]]; vav = pd.Series(DI['v']).rolling(288).mean().shift(1).fillna(0).to_numpy()
def ex(**kw):
    x = dict(tp=0, trA=0, trG=1, oiDn=0, oiDnG=0, vk=0, vg=0.5, retr=0, oiAd=0, oiAdL=1); x.update(kw)
    return summ(simR(DI, g_ex(c, s, ref, o, DI['v'], vav, 576, 5, x['tp'], x['trA'], x['trG'], x['oiDn'], x['oiDnG'], x['vk'], x['vg'], x['retr'], x['oiAd'], x['oiAdL'], 1)))
b = ex(); print('çıkış BAZ', b); rows = []
for kw in [dict(tp=t) for t in [3, 4, 6, 8, 10]] + [dict(trA=a, trG=g) for a in [3, 5, 8] for g in [1, 2, 3]] + [dict(retr=r) for r in [0.5, 0.7, 1.0]] + \
          [dict(oiDn=d, oiDnG=g) for d in [1, 2, 3] for g in [0, 1, 2]] + [dict(vk=k, vg=g) for k in [3, 5, 8] for g in [0.5, 1]] + [dict(oiAd=a, oiAdL=l) for a in [3, 5, 10] for l in [1, 3]]:
    x = ex(**kw); rows.append(dict(kural=str(kw), Δnet=x['net'] - b['net'], Δpf=round(x['pf'] - b['pf'], 3), n=x['n']))
r = pd.DataFrame(rows); print('çıkış: bazı geçen (net↑ & PF↑)', ((r.Δnet > 0) & (r.Δpf > 0)).sum(), '/', len(r)); print(r.sort_values('Δnet', ascending=False).head(6).to_string(index=False))
