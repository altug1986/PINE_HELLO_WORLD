import numpy as np, pandas as pd, json, time, itertools
from eng import load, go, P, TICK
from lib import simR, g_osc, zsc, rsi, g_btc
from ent import g_ent, feats, sig_of
from zz import g_rev
from sysB import mot2
from comps import fade, daily_mom
from vote import vote_pos
from met import metrics, SPLIT
# GENEL TARAMA — baz V5.0 (14 bileşen, ağırlıklı oy). Her değişiklik tek başına baza karşı.
D = load(5); F = feats(D); c = D['c']
BASEP = {1: dict(aL=3.5, aS=5.5, hold=648, sl=3.0), 2: dict(k=3.5, ex=0.5), 3: dict(k=3.5, ex=0.5), 4: dict(a=6.0), 5: dict(a=12.0),
         6: dict(lo=20, ex=50), 7: dict(px=1.5, oi=1.5, ex=60), 8: dict(a=5.0, b=5.0, hold=288), 9: dict(b=1.3), 10: dict(a=6.0),
         11: dict(x=2.0), 12: dict(u=3.0), 13: dict(u=8.0), 14: dict(n=30)}
_cache = {}; _z = {48: zsc(c, 48), 96: zsc(c, 96)}; _r14 = rsi(c, 14)
def comp(j, p, lag=0):
    key = (j, json.dumps(p, sort_keys=True), lag)
    if key in _cache: return _cache[key]
    if j == 1: v = mot2(D, F, dict(fam='PXA', H=288, aL=p['aL'], aS=p['aS'], hold=p['hold'], sl=p['sl']))
    elif j == 2: v = g_osc(c, _z[48], -p['k'], p['k'], p['ex'], -p['ex'], 8, 288)
    elif j == 3: v = g_osc(c, _z[96], -p['k'], p['k'], p['ex'], -p['ex'], 8, 288)
    elif j == 4: v = fade(c, 48, p['a'], 48, 10.0)
    elif j == 5: v = fade(c, 864, p['a'], 1728, 10.0)
    elif j == 6: v = g_osc(c, _r14, p['lo'], 100 - p['lo'], p['ex'], 100 - p['ex'], 3, 144)
    elif j == 7: v = go(D, P(liqPx=p['px'], liqOi=p['oi'], liqExit=p['ex'], liqSL=5, liqBars=144), np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], np.int64), lag)['tgt']
    elif j == 8:
        FF = F if lag == 0 else feats(D, lag); v = g_ent(c, sig_of(FF, 'OI-KURULUM', 288, p['a'], p['b'], -1), int(p['hold']), 10.0)
    elif j == 9: v = g_ent(c, sig_of(F, 'HACİM-KLİMAKS', 288, 5.0, p['b'], -1), 288, 10.0)
    elif j == 10: v = g_ent(c, sig_of(F, 'BTC-GÖRELİ', 288, p['a'], 0, -1), 288, 10.0)
    elif j == 11: v = g_btc(c, D['xc'], 3, p['x'], 24, 1)
    elif j == 12 or j == 13: v = g_rev(c, p['u'], p['u'])
    else: v = daily_mom(D, p['n'])
    _cache[key] = v.astype(np.int64); return _cache[key]
def system(cfg, lag=0):
    X = np.stack([comp(j, cfg['p'][j], lag if j in (7, 8) else 0) for j in range(1, 15)])
    return vote_pos(X, np.array(cfg['w'], float), cfg['on'], cfg['off'], 1)[0]
def ev(cfg, lag=0):
    m = metrics(D, simR(D, system(cfg, lag))); return m
BASE = dict(p=BASEP, w=[3, 3, 0.5, 0, 0, 1, 3, 3, 0.5, 1, 0.5, 0.5, 2, 0], on=0.05, off=0.0)
K = ['net', 'n', 'pf', 'sharpe', 'sortino', 'p95', 'p98', 'p99', 'mddc', 'mddi', 'ulcer', 'upi', 'IS', 'OOS', 'last12', 'posM']
def variants():
    WG = [0, 0.5, 1, 2, 3, 4]
    for j in range(14):
        for v in WG:
            if v != BASE['w'][j]: yield f'ağırlık {j+1} = {v}', ('w', j, v)
    for v in [0.03, 0.04, 0.06, 0.07, 0.08, 0.1]: yield f'giriş eşiği = {v}', ('on', None, v)
    for v in [-0.03, -0.01, 0.01, 0.02, 0.03]: yield f'çıkış eşiği = {v}', ('off', None, v)
    G = {1: dict(aL=[3, 4], aS=[5, 6], hold=[432, 864], sl=[2, 5, 8]), 2: dict(k=[3, 4], ex=[0, 1]), 3: dict(k=[3, 4], ex=[0, 1]), 4: dict(a=[5, 8]), 5: dict(a=[10, 15]),
         6: dict(lo=[15, 25], ex=[45, 55]), 7: dict(px=[1, 2], oi=[1, 2], ex=[55, 65]), 8: dict(a=[4, 6], b=[3, 4, 6, 8], hold=[144, 432]), 9: dict(b=[1.2, 1.5]),
         10: dict(a=[5, 7]), 11: dict(x=[1.5, 2.5]), 12: dict(u=[2.5, 3.5]), 13: dict(u=[6, 7, 10]), 14: dict(n=[20, 60])}
    for j, g in G.items():
        for k, vals in g.items():
            for v in vals: yield f'bileşen {j} {k} = {v}', ('p', (j, k), v)
def apply(cfg, ch):
    c2 = json.loads(json.dumps(cfg)); c2['p'] = {int(k): v for k, v in c2['p'].items()}; kind, j, v = ch
    if kind == 'w': c2['w'][j] = v
    elif kind == 'on': c2['on'] = v
    elif kind == 'off': c2['off'] = v
    else: c2['p'][j[0]][j[1]] = v
    return c2
if __name__ == '__main__':
    t0 = time.time(); b = ev(BASE); print('BAZ V5.0', {k: b[k] for k in K}, flush=True); rows = []
    for lab, ch in variants():
        m = ev(apply(BASE, ch)); rows.append(dict(değişiklik=lab, ch=json.dumps(ch), **{k: m[k] for k in K}))
    df = pd.DataFrame(rows); df.to_pickle('scan.pkl'); print(f'{time.time()-t0:.0f}s', len(df), 'varyant')
    ok = df[(df.net > b['net']) & (df.IS >= b['IS']) & (df.OOS >= b['OOS']) & (df.pf >= b['pf'] - 0.005) & (df.p98 <= b['p98'] * 1.1)]
    print('tek başına kabul (net↑, iki dönem ↑, PF≈, P98≤%110):', len(ok)); pd.set_option('display.width', 250)
    print(ok.sort_values('net', ascending=False)[['değişiklik', 'net', 'n', 'pf', 'sharpe', 'p98', 'ulcer', 'IS', 'OOS', 'last12']].to_string(index=False))
    # açgözlü birleştirme
    cur = json.loads(json.dumps(BASE)); cur['p'] = {int(k): v for k, v in cur['p'].items()}; cm = b; used = []
    for _, r in ok.sort_values('net', ascending=False).iterrows():
        ch = json.loads(r.ch); ch = (ch[0], tuple(ch[1]) if isinstance(ch[1], list) else ch[1], ch[2]); c2 = apply(cur, ch); m = ev(c2)
        if m['net'] > cm['net'] and m['IS'] >= cm['IS'] and m['OOS'] >= cm['OOS'] and m['pf'] >= cm['pf'] - 0.005 and m['p98'] <= b['p98'] * 1.1:
            cur, cm = c2, m; used.append(r.değişiklik); print('  + ', r.değişiklik, '→ net', m['net'], 'IS', m['IS'], 'OOS', m['OOS'], 'PF', m['pf'], 'P98', m['p98'], flush=True)
    print('BİRLEŞİK', {k: cm[k] for k in K}); json.dump(dict(cfg=cur, used=used), open('scan_best.json', 'w'), default=str)
