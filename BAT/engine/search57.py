import numpy as np, pandas as pd, sys, time, json
import base56b, v55
from base56b import A, CFG
from v55 import evalp, c, t, n, D
from scan import comp
from vote import vote_pos
from rg55 import vote2, AGREE, ADX60
from zone import ZCS
from feat import hk15, k15, c15
from eng import rsi, htf_map
R15 = np.nan_to_num(htf_map(rsi(c15, 14), hk15, k15), nan=50.0); hr = (t // 3600000) % 24
X15 = base56b.X15; W = np.array(CFG['w'] + [1.0], float); SCs = {}
for x11 in (2.0, 2.25, 2.5, 3.0):
    p11 = dict(CFG['p'][11]); p11['x'] = x11
    SCs[x11] = vote_pos(np.vstack([np.stack([comp(j, p11 if j == 11 else CFG['p'][j]) for j in range(1, 15)]), X15[None, :]]), W, CFG['on'], CFG['off'], 1)[1]
def prevcol(ms):
    k = t // ms; u, st = np.unique(k, return_index=True); en = np.r_[st[1:], n]; r = c[en - 1] < D['o'][st]; g = c[en - 1] > D['o'][st]; ii = np.searchsorted(u, k)
    return np.where(ii > 0, r[np.maximum(ii - 1, 0)], False), np.where(ii > 0, g[np.maximum(ii - 1, 0)], False), ((t - k * ms) // 3600000)
COL = {tf: prevcol(ms) for tf, ms in [('1g', 86400000), ('12s', 43200000), ('4s', 14400000)]}
SP = dict(x11=[2.0, 2.25, 2.5, 3.0], on=[0.04, 0.05, 0.06, 0.07, 0.08, 0.09], onT=[0.10, 0.12, 0.15], aT=[20.0, 25.0, 30.0], S=[6.5, 7.0, 7.5, 8.0], rA=[3.0, 4.0, 5.0],
          H=[504, 576, 648], xH=[-1.0, -2.0, -3.0], aH=[15.0, 20.0, 25.0], hm=[0, 48], nrMin=[0.0, 0.1],
          g=['yok', 'yok', 'dip', 'teyit'], gtf=['1g', '12s', '4s'], gw=[0.5, 1.0], gth=[40, 45, 50, 55, 60], gsym=[0, 1], zone=[0, 0, 1])
def build(p):
    v55.TG = vote2(SCs[p['x11']], ADX60, AGREE, 0.0, p['aT'], p['on'], p['on'], p['onT'], p['on'], 0.0)
    GL = np.zeros(n, bool); GS = np.zeros(n, bool)
    if p['g'] != 'yok':
        pr, pg, hin = COL[p['gtf']]; span = {'1g': 24, '12s': 12, '4s': 4}[p['gtf']]; win = hin < span * p['gw']
        if p['g'] == 'teyit': GL = pr & win & (R15 <= p['gth']); GS = pg & win & (R15 >= 100 - p['gth'])
        else: GL = pr & win & (R15 > p['gth']); GS = pg & win & (R15 < 100 - p['gth'])
        if not p['gsym']: GS[:] = False
    v55.GL = GL.astype(np.int64); v55.GS = GS.astype(np.int64); v55.ZC = ZCS['ortak1s_adx25']
    q = {k: p[k] for k in ['S', 'rA', 'H', 'xH', 'aH', 'hm', 'nrMin']}; q.update(L=99.0, rH=4.0)
    if p['zone']: q.update(zK1=10, zK2=30, zX=4.0, zM=0.5, zC=1, zAct=1)
    return q
if __name__ == '__main__':
    rng = np.random.default_rng(int(sys.argv[1])); rows = []; t0 = time.time()
    while time.time() - t0 < float(sys.argv[2]):
        p = {k: v[rng.integers(len(v))] for k, v in SP.items()}; q = build(p); e = evalp(q); rows.append(dict(**p, **e))
    pd.DataFrame(rows).to_pickle(f's57_{sys.argv[1]}.pkl'); print(len(rows), f'{time.time()-t0:.0f}s')
