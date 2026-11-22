# Astre V6.7 REV-25 trend katmanı — Pine'dan birebir port. Çekirdek yönünü (realDir) alır, icra yönünü (fExec) verir.
import numpy as np, pandas as pd
from numba import njit
from p1m import raw, rs, hmap, st_dir, TICK, TV0, MS
def feats(D, whale_tv=True):
    R = raw(); b, x = R['b'], R['x']; t = D['t']; c = D['c']; h = D['h']; l = D['l']; n = len(c)
    dk, _, dh, dl, dc, _ = rs(b, 1440); S = pd.Series
    hi30 = hmap(S(dh).rolling(30).max().to_numpy(), dk, t, 1440); lo30 = hmap(S(dl).rolling(30).min().to_numpy(), dk, t, 1440)
    b3 = hmap((dc / np.r_[np.full(3, np.nan), dc[:-3]] - 1), dk, t, 1440)
    xk, _, _, _, xc, _ = rs(x, 1440); x3 = hmap((xc / np.r_[np.full(3, np.nan), xc[:-3]] - 1), xk, t, 1440)
    ready = t >= (dk[0] + 30) * 1440 * MS
    H90 = D['H90']; std = np.nan_to_num(hmap(st_dir(H90[2], H90[3], H90[4], 10, 2.5).astype(float), H90[0], t, 90)).astype(np.int64)
    rng_ = hi30 - lo30; loc = np.where(ready & (rng_ > 0), (c - lo30) / rng_, np.nan); draw = 1 - c / hi30; rise = 1 - lo30 / c
    deep = np.zeros(n, np.int64)
    deep[ready & (std < 0) & (b3 < 0) & (loc <= 1 / 3) & (draw >= 0.25)] = -1
    deep[ready & (std > 0) & (b3 > 0) & (loc >= 2 / 3) & (rise >= 0.25) & (deep == 0)] = 1
    up = S(h).rolling(24).max().shift(1).to_numpy(); lo = S(l).rolling(24).min().shift(1).to_numpy()
    brk = np.where(c > up, 1, np.where(c < lo, -1, 0)); bnc = np.sign(c - np.r_[np.full(6, np.nan), c[:-6]]); bnc = np.nan_to_num(bnc).astype(np.int64)
    oT, oV = R['oT'], R['oV']; f5 = t // 300000 * 300000; od = dict(zip(oT, oV)); oiv = np.array([od.get(k, np.nan) for k in f5 - 300000])
    oiBar = ~np.isnan(oiv) & (oiv > 0); clock = np.r_[np.zeros(6, bool), (t[6:] - t[:-6]) == 6 * 300000]
    roll7 = lambda m: S(m.astype(float)).rolling(7).sum().to_numpy() == 7
    oiGood = roll7(oiBar) & clock; oiD = np.where(oiGood, oiv / np.r_[np.full(6, np.nan), oiv[:-6]] - 1, np.nan)
    m = pd.read_pickle('top5.pkl'); P = dict(zip(m.ms, m.sum_toptrader_long_short_ratio)); A = dict(zip(m.ms, m.count_toptrader_long_short_ratio))
    pv = np.array([P.get(k, np.nan) for k in f5 - 300000]); av = np.array([A.get(k, np.nan) for k in f5 - 300000])
    pair = ~np.isnan(pv) & ~np.isnan(av) & (pv > 0) & (av > 0)
    if whale_tv: pair &= t >= TV0
    allGood = roll7(pair & oiBar) & clock; wh = np.where(pair, np.log(np.where(pair, pv / av, 1.0)), np.nan)
    whD = np.where(allGood, wh - np.r_[np.full(6, np.nan), wh[:-6]], np.nan); p7 = np.r_[np.full(6, np.nan), pv[:-6]]
    flow = np.where(allGood & (whD > 0) & (pv > p7), 1, np.where(allGood & (whD < 0) & (pv < p7), -1, 0)).astype(np.int64)
    expand = oiGood & (np.nan_to_num(oiD) > 0); shrink = oiGood & (np.nan_to_num(oiD) < 0)
    rel = ~np.isnan(x3) & ((b3 - x3) * deep > 0)
    rideOk = (deep != 0) & (brk == deep) & (rel | (allGood & (flow == deep) & expand)) & ~(allGood & (flow == -deep) & expand)
    return dict(deep=deep, brk=brk.astype(np.int64), bnc=bnc, draw=np.nan_to_num(draw), rise=np.nan_to_num(rise), allGood=allGood, flow=flow, expand=expand, shrink=shrink, rideOk=rideOk, std=std)
@njit(cache=True)
def layer25(core, c, deep, brk, bnc, draw, rise, allGood, flow, expand, shrink, rideOk, std, stopPct, tick, uR, uS, uE):
    n = len(c); ex = np.zeros(n, np.int64); prev = 0; fx = 0; ride = 0; ent = 0.0; used = True; soft = False; extr = False; cb = 0
    for i in range(n):
        d = core[i]; chg = d != prev
        if chg:
            soft = False; extr = False; cb = i
            if d == 0: used = False; ride = 0
            else:
                counter = deep[i] == -d and brk[i] != d
                soft = uS == 1 and counter and allGood[i] and flow[i] == -d and shrink[i]
                extr = uE == 1 and counter and (draw[i] if d == 1 else rise[i]) >= 0.45 and bnc[i] == d and shrink[i]
                if not (extr and prev == 0 and fx == -d and ride == -d): ride = 0
        if d == 0:
            if ride != 0:
                if std[i] != ride or (c[i] / ent - 1.0) * ride <= -stopPct / 100.0: ride = 0
            elif uR == 1 and (not chg) and (not used) and rideOk[i]:
                ride = deep[i]; ent = c[i] + tick * ride; used = True
            fx = ride
        else:
            if soft and (i - cb >= 12 or deep[i] != -d or brk[i] == d or (allGood[i] and flow[i] == d and expand[i])): soft = False
            if extr and (deep[i] != -d or (brk[i] == d and expand[i] and ((not allGood[i]) or flow[i] == d))): extr = False
            if extr and ride != 0 and (c[i] / ent - 1.0) * ride <= -stopPct / 100.0: ride = 0
            if not extr: ride = 0
            fx = ride if extr else (0 if soft else d)
        prev = d; ex[i] = fx
    return ex
def apply25(D, core, F, uR=1, uS=1, uE=1):
    return layer25(core, D['c'], F['deep'], F['brk'], F['bnc'], F['draw'], F['rise'], F['allGood'], F['flow'], F['expand'], F['shrink'], F['rideOk'], F['std'], 7.0, TICK, uR, uS, uE)
