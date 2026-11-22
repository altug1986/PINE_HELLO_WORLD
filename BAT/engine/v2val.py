import numpy as np, pandas as pd, json
from rs import *
from cp import mae_of
from val import dd
from met import monkey
D0 = load(5); F0 = feats(D0); RG = regimes_bat(D0, F0)
df = pd.read_pickle('rs_IS.pkl'); p2 = json.loads(df.sort_values('o', ascending=False).iloc[0].p)
p1 = dict(aL=3.5, aS=5.5, hold=648, sl=3, zU=0, zP=0, zN=96, zK=3.5, zEx=0.5, zSl=8, zMb=288, pU=0, pP=1, pH=48, pA=8, pHold=192, rU=0, rP=1, rLen=14, rLo=10, rEx=50, rSl=3, rMb=144)
def full(p, lab, mc=True):
    R = simR(D0, build(D0, F0, p)); m = metrics(D0, R, lab)
    xt = pd.to_datetime(D0['t'][R['xb']], unit='ms'); b = pd.Series(R['pnl'], index=xt).groupby([xt.year, (xt.month - 1) // 6]).sum()
    q = R['q']; st = R['pnl'] - 0.0004 * q * (R['ep'] + R['xp']) - 4 * TICK * q
    out = {k: m[k] for k in ['net', 'n', 'pf', 'win', 'sharpe', 'sortino', 'p95', 'p98', 'p99', 'mddc', 'mddi', 'upi', 'IS', 'OOS', 'posM', 'last12', 'fee', 'yr']}
    out.update(blok=f'{(b>0).sum()}/{len(b)}', stres=round(st.sum()), mae=mae_of(D0, R))
    if mc:
        mk = monkey(D0, R, RG, 1000); out['monkey_max_p'] = round(max(mk.values()), 4); pn = R['pnl']; rng = np.random.default_rng(1)
        bs = np.array([rng.choice(pn, len(pn)) for _ in range(3000)]); out['mc_zarar'] = (bs.sum(1) < 0).mean(); out['mc_net_p5'] = round(np.percentile(bs.sum(1), 5))
    return out
A = full(p1, 'V1.0'); B = full(p2, 'V2 aday'); print(pd.DataFrame([A, B], index=['V1.0', 'V2 aday']).T.to_string(), flush=True)
rows = []
for k in ['aL', 'aS', 'hold', 'sl', 'zN', 'zK', 'zEx', 'zSl', 'zMb', 'pA', 'pHold']:
    j = SP[k].index(p2[k])
    for jj in (j - 1, j + 1):
        if 0 <= jj < len(SP[k]):
            q = dict(p2); q[k] = SP[k][jj]; m = metrics(D0, simR(D0, build(D0, F0, q))); rows.append(dict(p=f'{k}={SP[k][jj]}', net=m['net'], n=m['n'], IS=m['IS'], OOS=m['OOS'], p98=m['p98'], sharpe=m['sharpe']))
pl = pd.DataFrame(rows); print(pl.to_string(index=False)); print('plato net≥%90', (pl.net >= .9 * B['net']).mean().round(2), 'OOS>0', (pl.OOS > 0).mean().round(2))
json.dump(p2, open('v2.json', 'w'))
