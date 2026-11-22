import numpy as np, pandas as pd, json, time, sys
from cp2 import *
from lib import g_osc, zsc, rsi
SP = dict(aL=[3, 3.5, 4, 4.5, 5, 5.5, 6], aS=[4, 4.5, 5, 5.5, 6, 6.5, 7], hold=[144, 216, 288, 360, 432, 504, 576, 648], sl=[3, 5, 8],
          zU=[0, 1], zP=[0, 1], zN=[48, 72, 96, 144, 192], zK=[2.5, 3, 3.5, 4], zEx=[0, 0.5, 1], zSl=[3, 5, 8], zMb=[72, 144, 288],
          pU=[0, 1], pP=[0, 1], pH=[48, 96], pA=[3, 4, 5, 6, 8], pHold=[24, 48, 96, 192],
          rU=[0, 1], rP=[0, 1], rLen=[7, 14], rLo=[10, 15, 20, 25], rEx=[45, 50, 55], rSl=[3, 5], rMb=[72, 144])
_zc, _rc = {}, {}
def build(D, F, p):
    c = D['c']; core = g_ent(c, sig2(F, D, dict(H=288, aL=p['aL'], aS=p['aS'])), int(p['hold']), p['sl'])
    first, after = [], []
    if p['zU']:
        k = (id(c), p['zN'])
        if k not in _zc: _zc[k] = zsc(c, p['zN'])
        (first if p['zP'] == 0 else after).append(g_osc(c, _zc[k], -p['zK'], p['zK'], p['zEx'], -p['zEx'], p['zSl'], int(p['zMb'])))
    if p['pU']:
        H = p['pH']; dP = (c / np.r_[np.full(H, np.nan), c[:-H]] - 1) * 100
        s = np.nan_to_num(np.where(dP <= -p['pA'], 1, np.where(dP >= p['pA'], -1, 0))).astype(np.int64)
        (first if p['pP'] == 0 else after).append(g_ent(c, s, int(p['pHold']), 8.0))
    if p['rU']:
        k = (id(c), p['rLen'])
        if k not in _rc: _rc[k] = rsi(c, p['rLen'])
        (first if p['rP'] == 0 else after).append(g_osc(c, _rc[k], p['rLo'], 100 - p['rLo'], p['rEx'], 100 - p['rEx'], p['rSl'], int(p['rMb'])))
    return chain(first + [core] + after)
def samp(rng): return {k: v[rng.integers(len(v))] for k, v in SP.items()}
def obj(m, nmin): return sc(m) - (0 if m['n'] >= nmin else 5 * (nmin - m['n']))
if __name__ == '__main__':
    DI = cut(load(5), SPLIT); FI = feats(DI); rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    NMIN = 1850; rows = []; t0 = time.time(); best = None
    for it in range(3000):
        if best is not None and it > 1500 and rng.random() < 0.6:  # yerel arama
            p = dict(best[0]); k = list(SP)[rng.integers(len(SP))]; p[k] = SP[k][rng.integers(len(SP[k]))]
        else: p = samp(rng)
        m = metrics(DI, simR(DI, build(DI, FI, p))); o = obj(m, NMIN); rows.append(dict(p=json.dumps(p), o=o, **{k: m[k] for k in KS}))
        if best is None or o > best[1]: best = (p, o, m)
    df = pd.DataFrame(rows); df.to_pickle('rs_IS.pkl'); print(f'{time.time()-t0:.0f}s')
    print('n≥1850 olan', (df.n >= NMIN).sum()); pd.set_option('display.width', 300); pd.set_option('display.max_colwidth', 400)
    print(df.sort_values('o', ascending=False).head(5).to_string(index=False))
