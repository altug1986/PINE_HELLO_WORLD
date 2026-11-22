import numpy as np, pandas as pd, itertools, time
from eng import load, TICK
from lib import simR, g_osc
from ent import g_ent
D = load(5); c = D['c']; n = len(c); t = pd.to_datetime(D['t'], unit='ms')
o = np.r_[np.nan, D['oi'][:-1]]; xc = D['xc']; v = D['v']
vav = pd.Series(v).rolling(288).mean().shift(1).to_numpy()
def ret(a, k): return np.r_[np.full(k, np.nan), a[k:] / a[:-k] - 1] * 100
def zs(k): s = pd.Series(c); return ((s - s.rolling(k).mean()) / s.rolling(k).std(ddof=0)).to_numpy()
def S(x): return np.nan_to_num(x).astype(np.int64)
fams = []
for k, x, g, h in itertools.product([1, 2, 3, 6], [0.3, 0.5, 0.8, 1.2], [0, 0.3, 0.6], [3, 6, 12, 24, 48]):
    fams.append(('BTC-ÖNCÜ', f'k{k} x{x} g{g} h{h}', lambda k=k, x=x, g=g, h=h: g_ent(c, S(np.where((ret(xc, k) >= x) & (ret(c, k) <= ret(xc, k) - g), 1, np.where((ret(xc, k) <= -x) & (ret(c, k) >= ret(xc, k) + g), -1, 0))), h, 5.0)))
for k, ox, y, md, h in itertools.product([1, 2, 3, 6], [0.5, 1, 2, 3], [0.3, 0.6, 1, 2], ['kur-takip', 'kur-ters', 'boş-ters', 'boş-takip'], [6, 24, 48]):
    def f(k=k, ox=ox, y=y, md=md, h=h):
        r = ret(c, k); d = ret(o, k); cond = (np.abs(r) >= y) & ((d >= ox) if md.startswith('kur') else (d <= -ox))
        return g_ent(c, S(np.where(cond, np.sign(r) * (1 if md.endswith('takip') else -1), 0)), h, 5.0)
    fams.append(('OI-İVME', f'k{k} oi{ox} p{y} {md} h{h}', f))
for m, y, d, h in itertools.product([3, 5, 8, 12], [0.5, 1, 2, 3], [1, -1], [3, 6, 12, 24, 48]):
    fams.append(('HACİM-ŞOK', f'm{m} p{y} {"takip" if d > 0 else "ters"} h{h}', lambda m=m, y=y, d=d, h=h: g_ent(c, S(np.where((v >= m * vav) & (np.abs(ret(c, 1)) >= y), np.sign(ret(c, 1)) * d, 0)), h, 5.0)))
_z = {12: zs(12), 24: zs(24)}
for zn, kk, ex, sl, mb in itertools.product([12, 24], [2.5, 3, 3.5], [0, 0.5], [3, 5], [12, 48]):
    fams.append(('1S-Z', f'n{zn} k{kk} ex{ex} sl{sl} mb{mb}', lambda zn=zn, kk=kk, ex=ex, sl=sl, mb=mb: g_osc(c, _z[zn], -kk, kk, ex, -ex, sl, mb)))
# 6 aylık bloklar
blk = ((t.year - 2020) * 2 + (t.month - 1) // 6).to_numpy(); B = np.unique(blk)
rows = []; t0 = time.time()
for fam, prm, g in fams:
    R = simR(D, g()); pb = np.bincount(blk[R['xb']], weights=R['pnl'], minlength=B.max() + 1); nb = np.bincount(blk[R['xb']], minlength=B.max() + 1)
    rows.append(dict(fam=fam, prm=prm, net=R['pnl'].sum(), n=len(R['pnl']), pb=pb, nb=nb))
print(len(rows), f'{time.time()-t0:.0f}s')
df = pd.DataFrame(rows); df.to_pickle('intra.pkl')
# yürüyen pencere: her blok için geçmiş TÜM bloklarda en iyi net (≥50 işlem) ayarı seç, o blokta uygula
out = []
for fam, g in df.groupby('fam', sort=False):
    PB = np.stack(g.pb.values); NB = np.stack(g.nb.values); wf = []; picks = []
    for b in range(2, B.max() + 1):
        past = PB[:, :b].sum(1); ok = NB[:, :b].sum(1) >= 50
        if not ok.any(): continue
        j = np.where(ok, past, -1e18).argmax(); wf.append(PB[j, b]); picks.append(g.prm.values[j])
    wf = np.array(wf); top = g.sort_values('net', ascending=False).iloc[0]
    out.append(dict(aile=fam, ayar=len(g), tam_poz=round((g.net > 0).mean(), 2), YP_net=round(wf.sum()), YP_poz_blok=f'{(wf > 0).sum()}/{len(wf)}',
                    YP_son4blok=round(wf[-4:].sum()), en_cok_secilen=pd.Series(picks).mode()[0], tam_en_iyi=f'{top.prm} net {top.net:.0f} n {top.n}'))
pd.set_option('display.width', 300); pd.set_option('display.max_colwidth', 60); print(pd.DataFrame(out).to_string(index=False))
