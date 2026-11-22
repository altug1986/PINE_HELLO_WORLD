import numpy as np, pandas as pd, itertools
import v55
from search57 import build, SCs
from v55 import evalp, run, simR, D, t, n, c
from topfeat import d
from mtm30 import m30
from rg55 import vote2, AGREE, ADX60
from v54 import CFG
from scan import comp
from vote import vote_pos
from ent import g_ent
import base56b
K = dict(x11=2.5, on=0.04, onT=0.15, aT=20.0, S=7.0, rA=4.0, H=648, xH=-1.0, aH=20.0, hm=0, nrMin=0.1, g='yok', gtf='12s', gw=0.5, gth=40, gsym=1, zone=0); q0 = build(K)
x = d[(d.count_toptrader_long_short_ratio > 0) & (d.sum_toptrader_long_short_ratio > 0)]
ix = pd.to_datetime(x.ms, unit='ms'); TPs = pd.Series(np.log(x.sum_toptrader_long_short_ratio.values), index=ix); TAs = pd.Series(np.log(x.count_toptrader_long_short_ratio.values), index=ix)
hk = None
def hourly(s):
    global hk
    h = s.resample('1h', label='left').last(); hk = ((h.index - pd.Timestamp('1970-01-01')) // pd.Timedelta('1ms')).to_numpy(np.int64); return h
def to5(hv):
    j = np.searchsorted(hk, (t // 3600000) * 3600000) - 1; return np.where(j >= 0, hv[np.clip(j, 0, None)], np.nan)
def zh(h, N): return ((h - h.rolling(N, min_periods=N // 2).mean()) / h.rolling(N, min_periods=N // 2).std()).to_numpy()
DIVh = hourly(TPs - TAs); TPh = hourly(TPs)
G30 = np.nan_to_num(to5(zh(DIVh, 720))); BGL = G30 >= 1.0; BGS = G30 <= -1.0          # V5.8 30 günlük balina kapısı
def ev(GL, GS, lab, tg=None):
    v55.TG = base56b.TGQ if tg is None else tg
    if tg is None: v55.TG = q0_tg
    v55.GL = (BGL | GL).astype(np.int64); v55.GS = (BGS | GS).astype(np.int64); e = evalp(q0); tg_, _ = run(q0); m = m30(simR(D, tg_))
    return dict(t=lab, net=e['net'], n=e['n'], IS=e['IS'], OOS=e['OOS'], L12=e['L12'], L200=e['L200'], MaxDD30=m['MaxDD 30dk mum içi'], Calmar30=m['Calmar 30dk'], DD30=m['DD P95/P98/P99 30dk'])
q0_tg = v55.TG.copy(); Z = np.zeros(n, bool)
if __name__ != "__main__": raise ImportError("skip")
for N, k, md in itertools.product([24, 72, 168], [1.0, 1.5, 2.0], ['bizimle', 'karşı']):
    z = np.nan_to_num(to5(zh(DIVh, N))); z = z if md == 'bizimle' else -z
    rows.append(ev(z >= k, z <= -k, f'kapı +: fark z{N}s {md} ≥{k}'))
# yön tahmini bileşeni (16.): kısa pencerede balina parası yığılması → karşı yön (kontra) ya da aynı yön (takip)
X = np.vstack([np.stack([comp(j, dict(CFG['p'][11], x=2.5) if j == 11 else CFG['p'][j]) for j in range(1, 15)]), base56b.X15[None, :]])
for N, k, md, hold, w in itertools.product([24, 72, 168], [1.5, 2.0, 2.5], ['kontra', 'takip'], [48, 144], [1.0, 2.0]):
    z = np.nan_to_num(to5(zh(DIVh, N))); s = np.where(np.abs(z) >= k, np.sign(z), 0) * (-1 if md == 'kontra' else 1)
    X16 = np.vstack([X, g_ent(c, s.astype(np.int64), hold, 10.0)[None, :]]); sc = vote_pos(X16, np.array(CFG['w'] + [1.0, w], float), CFG['on'], CFG['off'], 1)[1]
    tg = vote2(sc, ADX60, AGREE, 0.0, 20.0, 0.04, 0.04, 0.15, 0.04, 0.0); q0_tg_save = q0_tg; q0_tg = tg
    rows.append(ev(Z, Z, f'bileşen: fark z{N}s {md} ≥{k} tut {hold} ağırlık {w}')); q0_tg = q0_tg_save
D_ = pd.DataFrame(rows); D_.to_pickle('short_top.pkl'); pd.set_option('display.width', 250)
b = D_.iloc[0]; print(D_.iloc[:1].to_string(index=False)); print('V5.8\'den iyi (net, ikinci dönem, son 200 gün üçü):', ((D_.net > b.net) & (D_.OOS > b.OOS) & (D_.L200 > b.L200)).sum(), '/', len(D_) - 1)
print(D_.iloc[1:].sort_values('Calmar30', ascending=False).head(8).to_string(index=False))
