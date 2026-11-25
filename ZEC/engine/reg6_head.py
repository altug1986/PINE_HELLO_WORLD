import sys, itertools, time, gc; sys.path.insert(0, '/home/claude/pine_hello_world/ZEC/engine'); sys.path.insert(0, '/home/claude/zecwork')
import numpy as np, pandas as pd, ztf, z03, peerfeat
D = ztf.build(1, thr=25.0); B = D['B']; c, h, l, o, a = D['c'], D['h'], D['l'], D['o'], D['atr']; n = D['n']; mid = D['mid']; idx = B.index
sh = lambda x, k: np.r_[np.full(k, np.nan), x[:-k]]; r10 = (c / sh(c, 10) - 1) * 100; spk = np.nan_to_num(r10 >= 3).astype(np.bool_)
order = ['ttposK4h', 'ttposK1g', 'whaleT', 'X17_basis', 'P09_spike', 'R21_glb', 'P07_rsi', 'P08_zema', 'P10_bb', 'P12_oiLiq', 'P11_oiCont', 'P15_resid', 'R18_whale', 'R19_book', 'P04_htf4h']
V = {k: np.asarray(D['V'][k], np.float32) for k in order}; RG = np.nan_to_num(D['RG']).astype(np.int8)
FH = peerfeat.build(B); PF = peerfeat.to1m(FH, B); del D; gc.collect()
print('özellik kapsama %:', (PF.notna().mean() * 100).round(1).to_dict(), flush=True)
F = {hz: np.r_[c[hz:] / c[:-hz] - 1, np.full(hz, np.nan)] * 1e4 for hz in (120, 240, 480)}
bit = lambda x: (np.nan_to_num(x, nan=0.0) >= 0).astype(np.int8)
K = bit(PF.btcC.values); Vh = bit(PF.relz.values); A_ = bit(PF.atrC.values); Pk = bit(PF.peerAtrZ.values)
REG = {'baz': RG + 1, 'BTCkor': (RG + 1) * 2 + K, 'ZECsıcak': (RG + 1) * 2 + Vh, 'ATRkor': (RG + 1) * 2 + A_, 'BTCkor×sıcak': (RG + 1) * 4 + K * 2 + Vh,
       'BTCkor×ATRkor': (RG + 1) * 4 + K * 2 + A_, 'sıcak×piyasaVol': (RG + 1) * 4 + Vh * 2 + Pk, 'BTCkor×sıcak×ATRkor': (RG + 1) * 8 + K * 4 + Vh * 2 + A_}
def select(R, thr, hz):
    L = np.zeros(n, np.float32); S = np.zeros(n, np.float32); f = F[hz]; fi = f[:mid]; Ri = R[:mid]
    for k in order:
        vv = V[k]; vi = vv[:mid]
        for side in (1, -1):
            for val in np.unique(R):
                m = (vi == side) & (Ri == val)
                if m.sum() < 400: continue
                if np.nanmean(fi[m] * side) >= thr: (L if side > 0 else S)[(vv == side) & (R == val)] += 1
    return L, S
pos_ = lambda t_: np.searchsorted(idx.values, np.datetime64(t_))
WIN = {'son2ay': (pos_('2026-09-24'), n - 1), 'son3hf': (pos_('2026-11-03'), n - 1), 'zayıf': (pos_('2022-06-01'), pos_('2022-06-01') * 0 + pos_('2023-04-01'))}
blk = np.searchsorted(idx.values, pd.date_range(idx[0].normalize(), idx[-1], freq='14D').values)
