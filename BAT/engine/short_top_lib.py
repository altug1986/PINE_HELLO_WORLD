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
q0_tg = v55.TG.copy()
