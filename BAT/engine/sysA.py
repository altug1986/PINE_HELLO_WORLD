import numpy as np, pandas as pd, itertools, json, sys, time
from ent import *
from lib import g_osc, zsc
from f2 import chain
CAND = dict(OIK=dict(fam='OI-KURULUM', H=288, a=5.0, b=5.0, hold=288, sl=10),
            HVX=dict(fam='HACİM-KLİMAKS', H=288, a=5.0, b=1.3, hold=288, sl=10),
            REL=dict(fam='BTC-GÖRELİ', H=288, a=6.0, b=0, hold=288, sl=10),
            PX1D=dict(fam='FİYAT', H=288, a=5.0, b=0, hold=288, sl=10),
            PX4H=dict(fam='FİYAT', H=48, a=6.0, b=0, hold=48, sl=10),
            PX8H=dict(fam='FİYAT', H=96, a=6.0, b=0, hold=192, sl=10),
            PX3D=dict(fam='FİYAT', H=864, a=12.0, b=0, hold=1728, sl=10),
            Z96=dict(fam='Z', H=96, a=3.5, b=0.5, hold=288, sl=8))
_z = {}
def mot(D, F, s):
    if s['fam'] == 'Z':
        k = (id(D['c']), s['H'])
        if k not in _z: _z[k] = zsc(D['c'], s['H'])
        return g_osc(D['c'], _z[k], -s['a'], s['a'], s['b'], -s['b'], s['sl'], int(s['hold']))
    return g_ent(D['c'], sig_of(F, s['fam'], s['H'], s['a'], s['b'], -1), int(s['hold']), s['sl'])
def sysR(D, F, order, S): return simR(D, chain([mot(D, F, S[m]) for m in order]))
def sc(m): return m['net'] - 1.5 * m['p98']
KS = ['net', 'n', 'pf', 'sharpe', 'p95', 'p98', 'p99', 'mddc', 'upi', 'posM']
if __name__ == '__main__':
    D = cut(load(5), SPLIT); F = feats(D); S = json.loads(json.dumps(CAND))
    single = {m: metrics(D, sysR(D, F, [m], S)) for m in S}
    for m, x in single.items(): print('tek', m, {k: x[k] for k in KS}, 'skor', round(sc(x)))
    order = [max(single, key=lambda m: sc(single[m]))]; cur = single[order[0]]; print('başlangıç', order, flush=True)
    while True:
        best = None
        for m in [x for x in S if x not in order]:
            for pos in range(len(order) + 1):
                o = order[:pos] + [m] + order[pos:]; x = metrics(D, sysR(D, F, o, S))
                ok = x['net'] >= cur['net'] * 1.06 and x['p98'] <= cur['p98'] * 1.10 and x['pf'] >= cur['pf'] - 0.05
                if ok and (best is None or sc(x) > sc(best[1])): best = (o, x)
        if best is None: break
        order, cur = best; print('ekle →', order, {k: cur[k] for k in KS}, flush=True)
    json.dump(dict(order=order, S=S), open('sysA.json', 'w')); print('SEÇİM', order)
