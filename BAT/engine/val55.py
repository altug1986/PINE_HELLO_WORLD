import numpy as np, pandas as pd, json, sys
import v55
from v55 import *
from ent import feats, regimes_bat
from met import monkey
from stopre import mae
RG = regimes_bat(D, feats(D))
BASE = {}
CA = dict(S=7.0, L=99.0, H=576, xH=-2.0, rH=4.0, aH=20.0, hm=48, nrMin=0.1)
CB = dict(S=7.0, L=99.0, H=576, xH=-2.0, rH=4.0, aH=20.0, hm=48)
CC = dict(H=576, xH=-2.0, rH=4.0, aH=20.0)                      # sadece kullanıcının fikri: 15 dk ADX'li poz içi süre sınırı
def dd(x): cp = np.cumsum(x); return (np.maximum.accumulate(np.r_[0, cp])[1:] - cp).max()
def blocks(R, months):
    x = pd.to_datetime(t[R['xb']], unit='ms'); return pd.Series(R['pnl']).groupby(((x.year * 12 + x.month - 1) // months).values).sum()
def battery(p, deep=True):
    tg, why = run(p); R = simR(D, tg); m = metrics(D, R); o = {k: m[k] for k in ['net', 'pct', 'n', 'pf', 'win', 'sharpe', 'sortino', 'calmar', 'p95', 'p98', 'p99', 'mddc', 'mddi', 'ulcer', 'upi', 'IS', 'OOS', 'posM', 'last12', 'fee', 'yr']}
    b2 = blocks(R, 2); b6 = blocks(R, 6); o.update(b2min=round(b2.min()), b2std=round(b2.std()), b2neg=int((b2 < 0).sum()), b2lt50=int((b2 < 50).sum()), b6=f'{(b6 > 0).sum()}/{len(b6)}', b6min=round(b6.min()))
    q = R['q']; st = R['pnl'] - 0.0004 * q * (R['ep'] + R['xp']) - 4 * TICK * q; o['stres'] = round(st.sum()); o['stresOOS'] = round(st[t[R['xb']] >= SPLIT].sum())
    o['mae'] = list(mae(R)); o['why'] = {int(k): (int(v['count']), int(v['sum'])) for k, v in pd.DataFrame(dict(w=why[R['xb']], p=R['pnl'])).groupby('w').p.agg(['count', 'sum']).iterrows()}
    if deep:
        mk = monkey(D, R, RG, 1000); o['monkey_p'] = round(max(mk.values()), 4)
        pn = R['pnl']; rng = np.random.default_rng(1); bs = np.array([rng.choice(pn, len(pn)) for _ in range(3000)])
        o['mc_p5'] = round(np.percentile(bs.sum(1), 5)); o['mc_loss'] = round((bs.sum(1) < 0).mean(), 4); o['mc_dd95'] = round(np.percentile([dd(x) for x in bs], 95))
    return o, R, b2, b6
if __name__ == '__main__':
    out = {}; Rs = {}
    for lab, p in [('BAZ V5.4', BASE), ('A', CA), ('B', CB), ('C', CC)]:
        o, R, b2, b6 = battery(p); out[lab] = o; Rs[lab] = (b2, b6); print(lab, o, flush=True)
    b6 = pd.DataFrame({k: v[1] for k, v in Rs.items()}); d = b6.sub(b6['BAZ V5.4'], axis=0)
    print('6 aylık blok Δ (A,B,C iyileşen blok):', [(int((d[k] > 0).sum()), round(d[k].sum())) for k in 'ABC'])
    b2 = pd.DataFrame({k: v[0] for k, v in Rs.items()}); b2.index = [f"{i*2//12}-{i*2%12+1:02d}" for i in b2.index]; b2.round(0).to_pickle('val55_b2.pkl')
    # komşu plato (A)
    rows = []
    for k, vals in dict(H=[504, 648], xH=[-1.0, -3.0], aH=[15.0, 25.0], rH=[2.0, 99.0], nrMin=[0.0, 0.2], S=[6.5, 7.5], hm=[16, 32, 0]).items():
        for v in vals:
            p = dict(CA); p[k] = v; e = evalp(p); rows.append(dict(p=f'{k}={v}', net=e['net'], OOS=e['OOS'], p98=e['p98'], sc=e['sc']))
    pl = pd.DataFrame(rows); print(pl.to_string(index=False)); print('plato: net>7017', (pl.net > 7017).mean(), '| OOS>1871', (pl.OOS > 1871).mean(), '| net≥%95 A', (pl.net >= 0.95 * out['A']['net']).mean())
    # gecikme: 15 dk ADX ve hacim oranı 1/2 bar geç
    for k in (1, 2):
        sv = (v55.A15.copy(), v55.NR.copy()); v55.A15[:] = np.r_[np.zeros(k), sv[0][:-k]]; v55.NR[:] = np.r_[np.ones(k), sv[1][:-k]]
        e = evalp(CA); eb = evalp(BASE); print(f'gecikme {k}: A net {e["net"]} OOS {e["OOS"]} p98 {e["p98"]} | baz {eb["net"]} OOS {eb["OOS"]}')
        v55.A15[:] = sv[0]; v55.NR[:] = sv[1]
    json.dump(out, open('val55.json', 'w'), default=str)
