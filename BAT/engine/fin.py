import numpy as np, pandas as pd, json
from r5 import g_lk, mae
from sysB import *
from val import dd
from met import regimes, monkey
D0 = load(5); F0 = feats(D0); RG = regimes_bat(D0, F0)
def RV4(s, D=D0, F=F0): return simR(D, g_lk(D['c'], sig2(F, D, dict(H=288, **s)), s['hold'], s['sl']))
V4 = dict(aL=3.5, aS=5.5, hold=648, sl=15)
R = RV4(V4); m = metrics(D0, R, 'V4')
xt = pd.to_datetime(D0['t'][R['xb']], unit='ms'); p = pd.Series(R['pnl'], index=xt); b = p.groupby([xt.year, (xt.month - 1) // 6]).sum()
q = R['q']; st = R['pnl'] - 0.0004 * q * (R['ep'] + R['xp']) - 4 * TICK * q
mk = monkey(D0, R, RG, 2000); pn = R['pnl']; rng = np.random.default_rng(1); bs = np.array([rng.choice(pn, len(pn)) for _ in range(5000)])
out = dict(m, blok=f'{(b>0).sum()}/{len(b)}', blok_min=round(b.min()), stres=round(st.sum()), stres_OOS=round(st[D0['t'][R['xb']] >= SPLIT].sum()),
           monkey_max_p=round(max(mk.values()), 4), mc_net_p5=round(np.percentile(bs.sum(1), 5)), mc_zarar=(bs.sum(1) < 0).mean(),
           mc_dd_p95=round(np.percentile([dd(x) for x in bs], 95)), mc_shuf_dd_p95=round(np.percentile([dd(rng.permutation(pn)) for _ in range(5000)], 95)),
           mae=mae(D0, R), long=round(pn[R['dir'] > 0].sum()), short=round(pn[R['dir'] < 0].sum()), hold_h=round(((R['xb'] - R['eb']) * 5 / 60).mean(), 1))
print({k: v for k, v in out.items()}); print(mk)
G = dict(aL=[3, 3.5, 4], aS=[5, 5.5, 6], hold=[576, 648, 720], sl=[12, 15, 20]); rows = []
for k, vals in G.items():
    for v in vals:
        if v == V4[k]: continue
        s = dict(V4); s[k] = v; x = metrics(D0, RV4(s)); rows.append(dict(p=f'{k}={v}', net=x['net'], IS=x['IS'], OOS=x['OOS'], pf=x['pf'], p98=x['p98']))
for H in [276, 282, 294, 300]:
    s = dict(V4); x = metrics(D0, simR(D0, g_lk(D0['c'], sig2(F0, D0, dict(H=H, **s)), 648, 15))); rows.append(dict(p=f'H={H}', net=x['net'], IS=x['IS'], OOS=x['OOS'], pf=x['pf'], p98=x['p98']))
pl = pd.DataFrame(rows); print(pl.to_string(index=False)); print('plato net>=%90', (pl.net >= .9 * m['net']).mean().round(2), 'OOS>0', (pl.OOS > 0).mean())
qq = p.groupby([xt.year, xt.quarter]).sum(); print(qq.loc[2024:].round().to_dict())
np.save('fin_R.npy', np.c_[R['eb'], R['xb'], R['dir'], R['ep'], R['xp'], R['q'], R['pnl']]); pd.to_pickle(out, 'fin.pkl')
