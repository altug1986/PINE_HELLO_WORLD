import numpy as np, pandas as pd, json
from orch3 import *
from cp import mae_of
from met import monkey
from val import dd
from ent import regimes_bat
NAMES7 = [x for x in BASE8 if x != 'TAKİP 5/5']; CFG = dict(floor=0.3, on=0.2, offr=0.0, k=3, pw=2, minc=6)
if __name__ == '__main__':
    D0 = load(5); F0 = feats(D0); S = inst_set(D0, F0, NAMES7); RG = regimes_bat(D0, F0)
    def ev(cfg, S=S, names=NAMES7):
        R, SC = run_set(D0, S, names, cfg['floor'], cfg['on'], cfg['offr'], cfg['k'], cfg['pw'], cfg['minc']); return R, SC, metrics(D0, R)
    R, SC, m = ev(CFG)
    xt = pd.to_datetime(D0['t'][R['xb']], unit='ms'); b = pd.Series(R['pnl'], index=xt).groupby([xt.year, (xt.month - 1) // 6]).sum()
    q = R['q']; st = R['pnl'] - 0.0004 * q * (R['ep'] + R['xp']) - 4 * TICK * q; mk = monkey(D0, R, RG, 1000); pn = R['pnl']; rng = np.random.default_rng(1)
    bs = np.array([rng.choice(pn, len(pn)) for _ in range(3000)])
    out = {k: m[k] for k in ['net', 'pct', 'n', 'pf', 'win', 'sharpe', 'sortino', 'calmar', 'p95', 'p98', 'p99', 'mddc', 'mddi', 'ulcer', 'upi', 'IS', 'OOS', 'posM', 'last12', 'fee', 'yr']}
    out.update(blok=f'{(b>0).sum()}/{len(b)}', stres=round(st.sum()), stres_OOS=round(st[D0['t'][R['xb']] >= SPLIT].sum()), mae=mae_of(D0, R), monkey_max_p=round(max(mk.values()), 4),
               mc_zarar=(bs.sum(1) < 0).mean(), mc_net_p5=round(np.percentile(bs.sum(1), 5)), mc_dd_p95=round(np.percentile([dd(x) for x in bs], 95)))
    print(out, flush=True)
    who = [NAMES7[int(np.argmax(np.where(S[:, i] == dd_, SC[:, i], -9)))] for i, dd_ in zip(R['eb'], R['dir'])]
    print('katkı:', pd.Series(pn).groupby(who).agg(['sum', 'count']).round().to_dict())
    rows = []
    for k, vals in dict(floor=[0.2, 0.4], on=[0.15, 0.25], k=[2, 4, 6], pw=[1, 3], minc=[4, 8]).items():
        for v in vals:
            c = dict(CFG); c[k] = v; _, _, x = ev(c); rows.append(dict(p=f'{k}={v}', net=x['net'], n=x['n'], IS=x['IS'], OOS=x['OOS'], pf=x['pf'], p98=x['p98']))
    pl = pd.DataFrame(rows); print(pl.to_string(index=False)); print('plato net≥%90', (pl.net >= .9 * m['net']).mean().round(2), 'OOS>0', (pl.OOS > 0).mean().round(2))
    S8 = inst_set(D0, F0, BASE8); _, _, m8 = ev(dict(floor=0, on=0.1, offr=0, k=6, pw=1, minc=6), S8, BASE8); print('8 baz tam:', {k: m8[k] for k in ['net', 'n', 'pf', 'sharpe', 'p98', 'IS', 'OOS', 'last12']})
    np.save('orchfin_tg.npy', np.r_[0]); json.dump(dict(names=NAMES7, cfg=CFG), open('orchfin.json', 'w'))
