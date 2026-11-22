import numpy as np, pandas as pd, sys, ast
from lib import *
from met import regimes, monkey, metrics
tf = 5; D = cut(load(tf), SPLIT); RG = regimes(D)
fams = families(D); df = pd.read_pickle(f'lib_{tf}.pkl'); df['p'] = df.prm.apply(ast.literal_eval)
def nbr_med(g):
    keys = list(g.p.iloc[0].keys()); vals = {k: sorted(set(p[k] for p in g.p)) for k in keys}; out = []
    idx = {tuple(p[k] for k in keys): r.net for p, r in zip(g.p, g.itertuples())}
    for p in g.p:
        nb = []
        for k in keys:
            j = vals[k].index(p[k])
            for jj in (j - 1, j + 1):
                if 0 <= jj < len(vals[k]):
                    q = dict(p); q[k] = vals[k][jj]; v = idx.get(tuple(q[kk] for kk in keys))
                    if v is not None: nb.append(v)
        out.append((np.median(nb) if nb else np.nan, np.mean(np.array(nb) > 0) if nb else np.nan))
    return out
rows = []
for fam, g in df.groupby('fam', sort=False):
    nm = nbr_med(g); g = g.assign(nmed=[a for a, b in nm], npos=[b for a, b in nm])
    g = g.assign(sc=0.5 * g.net + 0.5 * g.nmed)
    cand = g[(g.n >= 200)].sort_values('sc', ascending=False).head(1)
    for i, r in cand.iterrows():
        R = simR(D, fams[i][2]()); m = metrics(D, R, f'{fam} {r.prm}')
        mk = monkey(D, R, RG, 1000)
        rows.append(dict(tetik=fam, ayar=r.prm, net=r.net, n=r.n, pf=r.pf, mdd=r.mdd, sharpe=m['sharpe'], p98=m['p98'],
                         komsu_med=round(r.nmed), komsu_pos=round(r.npos * 100), **{k: round(v, 3) for k, v in mk.items()}))
        print(rows[-1], flush=True)
out = pd.DataFrame(rows); out.to_pickle('f1b.pkl')
pd.set_option('display.width', 250); print(out.drop(columns=['ayar']).to_string(index=False)); print(out[['tetik', 'ayar']].to_string(index=False))
