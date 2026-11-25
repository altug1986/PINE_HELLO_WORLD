"""BEST OF BEST — 1 dk, 2 haftalık eşit bloklar. Hedef: net ≥ 10.000 $ (%2000), işlem ~1.200–2.500, blokların medyanı ve pozitif oranı yüksek,
2022-06 → 2023-03 zayıf dönemi pozitif, bar içi ters gidiş kuyruğu düşük."""
import sys, itertools, time, gc; sys.path.insert(0, '/home/claude/pine_hello_world/ZEC/engine')
import numpy as np, pandas as pd, ztf, z03
D = ztf.build(1, thr=25.0); B = D['B']; c, h, l, o, a = D['c'], D['h'], D['l'], D['o'], D['atr']; n = D['n']; mid = D['mid']; idx = B.index
sh = lambda x, k: np.r_[np.full(k, np.nan), x[:-k]]; r10 = (c / sh(c, 10) - 1) * 100; spk = np.nan_to_num(r10 >= 3).astype(np.bool_); z = np.zeros(n, np.bool_)
order = ['ttposK4h', 'ttposK1g', 'whaleT', 'X17_basis', 'P09_spike', 'R21_glb', 'P07_rsi', 'P08_zema', 'P10_bb', 'P12_oiLiq', 'P11_oiCont', 'P15_resid', 'R18_whale', 'R19_book', 'P04_htf4h']
V = {k: np.asarray(D['V'][k], np.float32) for k in order}; RG = D['RG']
del D; gc.collect()
F = {hz: np.r_[c[hz:] / c[:-hz] - 1, np.full(hz, np.nan)] * 1e4 for hz in (120, 240, 480)}
def select(thr, hz):
    L = np.zeros(n, np.float32); S = np.zeros(n, np.float32); f = F[hz]
    for k in order:
        vv = V[k]
        for side in (1, -1):
            for val in (1, -1, 0):
                m = (vv == side) & (RG == val)
                if m[:mid].sum() < 400: continue
                if np.nanmean((f * side)[:mid][m[:mid]]) >= thr: (L if side > 0 else S)[m] += 1
    return L, S
blk = np.searchsorted(idx.values, pd.date_range(idx[0].normalize(), idx[-1], freq='14D').values)
w0, w1 = np.searchsorted(idx.values, np.datetime64('2022-06-01')), np.searchsorted(idx.values, np.datetime64('2023-04-01'))
yrs = n / 525600
rows = []; t0 = time.time()
for thr, hz in itertools.product([10, 15, 20, 25, 35], [120, 240, 480]):
    L, S = select(thr, hz)
    for kL, kS, ts, tH in itertools.product([1, 2, 3, 4], [2, 3, 4, 99], [6, 8, 11, 16, 24], [240, 480, 960, 1920]):
        aL = (L >= kL); aS = (S >= kS) & ~aL
        if aL.sum() == 0: continue
        eq, T = z03.sim7(o, h, l, c, a, aL, aS, float(ts), int(tH), z, 8.0, 12.0, 240, spk, 60)
        nt = len(T)
        if nt < 300: continue
        bp = np.diff(eq[blk]); x30 = eq[::30]; dd = np.maximum.accumulate(x30) - x30
        r2 = np.corrcoef(np.arange(len(x30)), x30)[0, 1] ** 2 if eq[-1] > 0 else 0
        pos = (bp > 0).mean(); cv = bp.std() / bp.mean() if bp.mean() > 0 else 99
        p98 = np.percentile(dd, 98)
        bob = (eq[-1] / yrs / p98) * r2 * pos / (1 + 0.25 * min(cv, 8)) if p98 > 0 and eq[-1] > 0 else -1
        mae = T[:, 6]
        rows.append(dict(thr=thr, hz=hz, kL=kL, kS=kS, ts=ts, tH=tH, n=nt, net=eq[-1], IS=eq[mid] - eq[0], OOS=eq[-1] - eq[mid], MaxDD=dd.max(), DDp98=p98,
                         R2=r2, blokMed=np.median(bp), blokPoz=pos * 100, blokCV=cv, zayıf=eq[w1] - eq[w0], maeP95=np.percentile(mae, 95) * 100, maeMax=mae.max() * 100,
                         short=T[T[:, 2] < 0, 5].sum(), BoB=bob))
    print(f'thr {thr} hz {hz} bitti, {len(rows)} koşu, {round(time.time() - t0)} s', flush=True)
R = pd.DataFrame(rows); R.to_csv('/home/claude/pine_hello_world/ZEC/research/BOB_1DK_2HAFTA.csv', index=False)
pd.set_option('display.width', 260)
print('\nEN YÜKSEK NET (ilk 8):'); print(R.sort_values('net', ascending=False).head(8).round(2).to_string(index=False))
f = R[(R.n >= 1200) & (R.n <= 2500)]
print(f'\nİşlem 1.200–2.500 arası ({len(f)} ayar) — en yüksek net:'); print(f.sort_values('net', ascending=False).head(6).round(2).to_string(index=False))
g = f[(f.zayıf > 0) & (f.IS > 0) & (f.OOS > 0)]
print(f'\n+ zayıf dönem pozitif + iki yarı pozitif ({len(g)}) — BoB sıralı:'); print(g.sort_values('BoB', ascending=False).head(8).round(2).to_string(index=False))
print('\nNET ≥ 10.000 olan ayar sayısı:', (R.net >= 10000).sum(), '| ≥ 6.000:', (R.net >= 6000).sum(), '| ≥ 5.000:', (R.net >= 5000).sum())
