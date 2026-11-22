import numpy as np, pandas as pd, itertools, sys, json, time
from lib import *
from met import metrics, SPLIT
GR = dict(
    LIQ=dict(px=[1, 1.5, 2, 2.5, 3], oi=[1, 1.5, 2, 2.5], ex=[55, 60, 65], sl=[3, 5, 8], mb=[72, 144]),
    Z=dict(n=[288, 432, 576, 864], k=[2.5, 3, 3.5], ex=[0, 0.5, 1], sl=[3, 5, 8], mb=[144, 288, 576]),
    RSI=dict(len=[10, 14, 21], lo=[15, 20, 25], ex=[45, 50, 55], sl=[2, 3, 5], mb=[72, 144, 288]),
    BTC=dict(n=[2, 3, 4, 6], x=[1.5, 2, 2.5, 3], hold=[12, 24, 48], lag=[1]))
BASE = dict(LIQ=dict(px=1.5, oi=1.5, ex=60, sl=5, mb=144), Z=dict(n=576, k=3, ex=0.5, sl=5, mb=288),
            RSI=dict(len=14, lo=20, ex=50, sl=3, mb=144), BTC=dict(n=3, x=2, hold=24, lag=1))
_cz, _cr = {}, {}
def motor(D, m, p, lag=0, blag=0):
    c = D['c']
    if m == 'LIQ':
        return go(D, P(liqPx=p['px'], liqOi=p['oi'], liqExit=p['ex'], liqSL=p['sl'], liqBars=p['mb']), np.array([1,0,0,0,0,0,0,0,0], np.int64), lag)['tgt']
    if m == 'Z':
        k = (id(c), p['n'])
        if k not in _cz: _cz[k] = zsc(c, p['n'])
        return g_osc(c, _cz[k], -p['k'], p['k'], p['ex'], -p['ex'], p['sl'], p['mb'])
    if m == 'RSI':
        k = (id(c), p['len'])
        if k not in _cr: _cr[k] = rsi(c, p['len'])
        return g_osc(c, _cr[k], p['lo'], 100 - p['lo'], p['ex'], 100 - p['ex'], p['sl'], p['mb'])
    if m == 'BTC':
        xc = D['xc'] if blag == 0 else np.r_[np.full(blag, np.nan), D['xc'][:-blag]]
        return g_btc(c, xc, p['n'], p['x'], p['hold'], p['lag'])
def chain(tgs):
    out = np.zeros_like(tgs[0])
    for t in reversed(tgs): out = np.where(t != 0, t, out)
    return out
def score(m): return m['net'] - 1.5 * m['p98']
def evalc(D, order, prm, lag=0, blag=0):
    R = simR(D, chain([motor(D, m, prm[m], lag, blag) for m in order])); return metrics(D, R), R

if __name__ == '__main__':
    D = cut(load(5), SPLIT)
    rows = []
    for order in [('LIQ', 'Z', 'RSI', 'BTC'), ('LIQ', 'RSI', 'Z', 'BTC'), ('Z', 'RSI', 'LIQ', 'BTC'), ('RSI', 'Z', 'LIQ', 'BTC'),
                  ('LIQ', 'Z', 'RSI'), ('LIQ', 'RSI'), ('LIQ', 'Z'), ('Z', 'RSI'), ('LIQ', 'Z', 'BTC'), ('LIQ', 'RSI', 'BTC')]:
        m, _ = evalc(D, order, BASE); rows.append(dict(sıra='>'.join(order), **{k: m[k] for k in ['net', 'n', 'pf', 'sharpe', 'p95', 'p98', 'p99', 'mddc', 'upi', 'posM']}, skor=round(score(m))))
    df = pd.DataFrame(rows).sort_values('skor', ascending=False); print(df.to_string(index=False), flush=True)
    order = tuple(df.iloc[0]['sıra'].split('>')); prm = json.loads(json.dumps(BASE)); best = score(evalc(D, order, prm)[0])
    t0 = time.time()
    for rnd in range(2):
        for m in order:
            keys = list(GR[m]); bestp = dict(prm[m])
            for vals in itertools.product(*[GR[m][k] for k in keys]):
                q = dict(prm); q[m] = dict(zip(keys, vals)); s = score(evalc(D, order, q)[0])
                if s > best: best = s; bestp = q[m]
            prm[m] = bestp; print(rnd, m, prm[m], round(best), f'{time.time()-t0:.0f}s', flush=True)
    json.dump(dict(order=order, prm=prm), open('f2_best.json', 'w'))
    m, _ = evalc(D, order, prm); print({k: v for k, v in m.items()})
