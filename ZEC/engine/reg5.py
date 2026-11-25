import sys, itertools, time, gc; sys.path.insert(0, '/home/claude/pine_hello_world/ZEC/engine'); sys.path.insert(0, '/home/claude/zecwork')
import numpy as np, pandas as pd, ztf, z03, peerfeat
D = ztf.build(1, thr=25.0); B = D['B']; c, h, l, o, a = D['c'], D['h'], D['l'], D['o'], D['atr']; n = D['n']; mid = D['mid']; idx = B.index
sh = lambda x, k: np.r_[np.full(k, np.nan), x[:-k]]; r10 = (c / sh(c, 10) - 1) * 100; spk = np.nan_to_num(r10 >= 3).astype(np.bool_)
order = ['ttposK4h', 'ttposK1g', 'whaleT', 'X17_basis', 'P09_spike', 'R21_glb', 'P07_rsi', 'P08_zema', 'P10_bb', 'P12_oiLiq', 'P11_oiCont', 'P15_resid', 'R18_whale', 'R19_book', 'P04_htf4h']
V = {k: np.asarray(D['V'][k], np.float32) for k in order}; RG = np.nan_to_num(D['RG']).astype(np.int8)
PF = peerfeat.build(B); del D; gc.collect()
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
z = np.zeros(n, np.int64); A4 = np.zeros(n); rows = []; t0 = time.time()
for rn, R in REG.items():
    for thr, hz in itertools.product((10, 15, 25), (120, 240, 480)):
        L, S = select(R, thr, hz)
        for kL, kS, ts, tH in itertools.product((2, 3), (2, 3, 4), (8.0, 16.0, 24.0), (480, 960, 1920)):
            aL = L >= kL; aS = (S >= kS) & ~aL
            eq, T = z03.sim10(o, h, l, c, a, aL, aS, ts, tH, spk, 60, z, A4, 0)
            x30 = eq[::30]; dd = np.maximum.accumulate(x30) - x30; bp = np.diff(eq[blk])
            r = dict(rej=rn, thr=thr, hz=hz, kL=kL, kS=kS, ts=ts, tH=tH, n=len(T), net=eq[-1], IS=eq[mid] - eq[0], OOS=eq[-1] - eq[mid], MaxDD=dd.max(), DDp98=np.percentile(dd, 98),
                     R2=np.corrcoef(np.arange(len(x30)), x30)[0, 1] ** 2, blokPoz=(bp > 0).mean() * 100, top3=np.sort(T[:, 5])[-3:].sum() / max(eq[-1], 1) * 100, short=T[T[:, 2] < 0, 5].sum())
            for k_, (a_, b_) in WIN.items(): r[k_] = eq[b_] - eq[a_]
            rows.append(r)
    print(rn, 'bitti', round(time.time() - t0), 'sn', flush=True)
Rr = pd.DataFrame(rows); Rr.to_csv('/home/claude/pine_hello_world/ZEC/research/REJIM_AKRAN_1DK.csv', index=False)
pd.set_option('display.width', 250)
f = Rr[(Rr.IS > 0) & (Rr.OOS > 0) & Rr.n.between(1400, 2300)]
print('\nRejim türüne göre (işlem 1.400–2.300, iki yarı pozitif): en yüksek net / medyan net / aday sayısı')
print(f.groupby('rej').net.agg(['max', 'median', 'count']).round(0).sort_values('max', ascending=False).to_string())
print(f.sort_values('net', ascending=False).head(15).round(1).to_string(index=False))
