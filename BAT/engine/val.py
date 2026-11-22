import numpy as np, pandas as pd, json, sys
from sysB import *
D0 = load(5); F0 = feats(D0); RG0 = regimes_bat(D0, F0)
K = ['net', 'pct', 'n', 'pf', 'win', 'sharpe', 'sortino', 'calmar', 'p95', 'p98', 'p99', 'mddc', 'mddi', 'ulcer', 'upi', 'IS', 'OOS', 'posM', 'last12', 'fee', 'yr']
def dd(x):
    cp = np.cumsum(x); return (np.maximum.accumulate(np.r_[0, cp])[1:] - cp).max()
def battery(lab, order, S, plateau=True, mc=True):
    R = sysR2(D0, F0, order, S); m = metrics(D0, R, lab); out = dict(m)
    xt = pd.to_datetime(D0['t'][R['xb']], unit='ms'); p = pd.Series(R['pnl'], index=xt)
    b = p.groupby([xt.year, (xt.month - 1) // 6]).sum(); out['blok'] = f'{(b > 0).sum()}/{len(b)}'; out['blok_min'] = round(b.min())
    q = R['q']; st = R['pnl'] - 0.0004 * q * (R['ep'] + R['xp']) - 4 * TICK * q; out['stres'] = round(st.sum()); out['stres_OOS'] = round(st[D0['t'][R['xb']] >= SPLIT].sum())
    mk = monkey(D0, R, RG0, 2000); out['monkey_max_p'] = round(max(mk.values()), 4); out['monkey'] = {k: round(v, 3) for k, v in mk.items()}
    pn = R['pnl']; out['long'] = round(pn[R['dir'] > 0].sum()); out['short'] = round(pn[R['dir'] < 0].sum())
    if mc:
        rng = np.random.default_rng(1); bs = np.array([rng.choice(pn, len(pn)) for _ in range(5000)])
        out['mc_net_p5'] = round(np.percentile(bs.sum(1), 5)); out['mc_zarar'] = round((bs.sum(1) < 0).mean(), 4)
        out['mc_dd_p95'] = round(np.percentile([dd(x) for x in bs], 95)); out['mc_shuf_dd_p95'] = round(np.percentile([dd(rng.permutation(pn)) for _ in range(5000)], 95))
    if plateau:
        rows = []
        for mm in order:
            for k in GR[mm]:
                vals = GR[mm][k]; v = S[mm][k]
                if v not in vals: continue
                j = vals.index(v)
                for jj in (j - 1, j + 1):
                    if 0 <= jj < len(vals):
                        T = json.loads(json.dumps(S)); T[mm][k] = vals[jj]; x = metrics(D0, sysR2(D0, F0, order, T))
                        rows.append(dict(p=f'{mm}.{k}={vals[jj]}', net=x['net'], IS=x['IS'], OOS=x['OOS'], pf=x['pf'], p98=x['p98']))
        pl = pd.DataFrame(rows); out['plato_%90'] = round((pl.net >= 0.9 * m['net']).mean(), 2); out['plato_OOSpoz'] = round((pl.OOS > 0).mean(), 2); out['plato_tab'] = pl
    return out, R
if __name__ == '__main__':
    B = json.load(open('sysB.json'))
    C = {'V1 çekirdek (PXA)': (['PXA'], B['PXA']['S']),
         'V1s simetrik': (['PXA'], dict(PXA=dict(fam='PXA', H=288, aL=5.5, aS=5.5, hold=288, sl=5))),
         'V2 OI-KURULUM > PXA': (['OIK', 'PXA'], B['OIK+PXA']['S'])}
    res = {}
    for lab, (o, S) in C.items():
        out, R = battery(lab, o, S); res[lab] = out; print(lab, {k: out[k] for k in ['net', 'n', 'pf', 'IS', 'OOS', 'p98', 'blok', 'stres', 'stres_OOS', 'monkey_max_p', 'plato_%90', 'plato_OOSpoz', 'long', 'short']}, flush=True)
        print(out['plato_tab'].to_string(index=False))
    pd.to_pickle(res, 'val.pkl')
    # OI gecikme V2
    o, S = C['V2 OI-KURULUM > PXA']; Sp = dict(PXA=S['PXA'])
    for lag in [0, 1, 2]:
        F = feats(D0, lag); a = metrics(D0, sysR2(D0, F, o, S)); b = metrics(D0, sysR2(D0, F, ['PXA'], Sp))
        print(f'OI gecikme {lag}: V2 {a["net"]} PF {a["pf"]} P98 {a["p98"]} IS {a["IS"]} OOS {a["OOS"]} | aynı PXA tek {b["net"]} PF {b["pf"]} P98 {b["p98"]} IS {b["IS"]} OOS {b["OOS"]}')
