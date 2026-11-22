import numpy as np, pandas as pd, v55
from tvcmp63 import tvx, Rtv, Mdf, GLb, GSb, Z
from astre_test import TS, R_
from v55 import t, n
from mtm30 import m30
T = tvx('tv/v63_tv.xlsx'); T = T[T.xt <= t[-1]]; kt = set(zip(T.et, T.dr))
_, Rg = R_(GLb, GSb, TS); _, R0 = R_(Z, Z, TS); Mg, M0 = Mdf(Rg), Mdf(R0)
blk = sorted((k0 for k0 in set(zip(M0.et, M0.dr)) - set(zip(Mg.et, Mg.dr))), key=lambda x: x[0])
b = pd.DataFrame(blk, columns=['et', 'dr']); b['tv'] = [k in kt for k in blk]; b['ts'] = pd.to_datetime(b.et, unit='ms')
r = b[b.ts >= '2026-01-01']; print(r[['ts', 'dr', 'tv']].head(30).to_string(index=False))
first_off = b[(~b.tv) & (b.ts >= '2026-01-01')].ts.min(); print('TV\'de kapının ilk kestiği giriş:', first_off, '| bundan sonra TV\'de olan kesilmiş giriş:', int(b[(b.ts > first_off) & b.tv].shape[0]))
q = pd.to_datetime(T.et, unit='ms').dt.to_period('Q'); kg = set(zip(Mg.et, Mg.dr)); k0 = set(zip(M0.et, M0.dr))
T['g'] = [k in kg for k in zip(T.et, T.dr)]; T['u'] = [k in k0 for k in zip(T.et, T.dr)]
print((T.groupby(q)[['g', 'u']].mean() * 100).round(1).T.to_string())
