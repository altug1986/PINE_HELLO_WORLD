import numpy as np, pandas as pd, ast
from lib import *
from met import SPLIT
D = load(5); k = np.searchsorted(D['t'], SPLIT)
DO = {a: (v[k:] if isinstance(v, np.ndarray) and len(v) == len(D['t']) else v) for a, v in D.items()}
rows = []
for fam, prm, g in families(DO):
    s = summ(simR(DO, g())); s.update(fam=fam, prm=str(prm)); rows.append(s)
do = pd.DataFrame(rows); di = pd.read_pickle('lib_5.pkl')
m = di.merge(do, on=['fam', 'prm'], suffixes=('_IS', '_OOS'))
m.to_pickle('lib_both.pkl')
print(m.groupby('fam', sort=False).apply(lambda g: pd.Series(dict(ayar=len(g), IS_poz=(g.net_IS > 0).mean(), OOS_poz=(g.net_OOS > 0).mean(), ikisi_poz=((g.net_IS > 0) & (g.net_OOS > 0)).mean(), corr=g.net_IS.corr(g.net_OOS))), include_groups=False).round(2).to_string())
b = m[(m.net_IS > 0) & (m.net_OOS > 0) & (m.n_IS >= 150) & (m.n_OOS >= 60)].copy(); b['mn'] = np.minimum(b.net_IS / 4.7, b.net_OOS / 2.04)
print(b.sort_values('mn', ascending=False).head(15)[['fam', 'prm', 'net_IS', 'n_IS', 'pf_IS', 'net_OOS', 'n_OOS', 'pf_OOS']].to_string(index=False))
# YENİ zincir OOS long/short
R = np.load('f3_R.npy'); xt = D['t'][R[:, 1].astype(int)]; o = R[xt >= SPLIT]
print('OOS long', o[o[:, 2] > 0, 6].sum().round(), 'short', o[o[:, 2] < 0, 6].sum().round())
