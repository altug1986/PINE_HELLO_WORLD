# V6.6 Pine kodunun zaman dilimi parametreli birebir portu (1 dk / 5 dk). Bar sayılı pencereler 5 dk karşılığının tf katı.
import numpy as np, pandas as pd
from numba import njit
from collections import deque
from eng import dmi, rsi, htf_map
TICK = 1e-5; MS = 60000; HMS = 3600000
SPL = (pd.Timestamp('2024-11-01') - pd.Timestamp('1970-01-01')) // pd.Timedelta('1ms')
TV0 = (pd.Timestamp('2026-04-01') - pd.Timestamp('1970-01-01')) // pd.Timedelta('1ms')
_R = {}
def raw():
    if not _R:
        b = pd.read_csv('BATUSDT_1DK_OHLCV_UTC.csv', usecols=['open_time_ms', 'open', 'high', 'low', 'close', 'volume']).to_numpy(float)
        x = pd.read_csv('btc1m.csv.gz', usecols=['time', 'open', 'high', 'low', 'close', 'volume']).to_numpy(float)
        b = b[b[:, 0] <= x[-1, 0]]
        o = pd.read_csv('oi/BATUSDT_OI_KAYNAK_5DK_UTC.csv', usecols=['create_time_utc', 'sum_open_interest'])
        oT = pd.to_datetime(o.create_time_utc).to_numpy().astype('datetime64[ms]').astype(np.int64); oV = o.sum_open_interest.to_numpy(float); m = oV > 0
        _R.update(b=b, x=x, oT=oT[m], oV=oV[m])
    return _R
def rs(a, tf):
    t = a[:, 0].astype(np.int64); k = t // (tf * MS); st = np.r_[0, np.flatnonzero(np.diff(k)) + 1]; en = np.r_[st[1:], len(t)]
    return k[st], a[st, 1], np.maximum.reduceat(a[:, 2], st), np.minimum.reduceat(a[:, 3], st), a[en - 1, 4], np.add.reduceat(a[:, 5], st)
def hmap(vals, hk, t, htf): return htf_map(vals, hk, t // (htf * MS))      # lookahead_on + [1]
def ffill(k_src, v_src, k_dst):
    j = np.searchsorted(k_src, k_dst, side='right') - 1; return np.where(j >= 0, v_src[np.clip(j, 0, None)], np.nan)
@njit(cache=True)
def st_dir(h, l, c, ln, fac):
    m = len(c); d = np.zeros(m, np.int64); atr = np.full(m, np.nan); tr = np.zeros(m)
    for i in range(m): tr[i] = h[i] - l[i] if i == 0 else max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    s = 0.0
    for i in range(m):
        if i < ln - 1: s += tr[i]
        elif i == ln - 1: s += tr[i]; atr[i] = s / ln
        else: atr[i] = (atr[i - 1] * (ln - 1) + tr[i]) / ln
    up = 0.0; lo = 0.0; stv = 0.0; dr = 1
    for i in range(m):
        if np.isnan(atr[i]): continue
        src = (h[i] + l[i]) / 2; U = src + fac * atr[i]; L = src - fac * atr[i]; pU = up; pL = lo
        if i > 0 and not np.isnan(atr[i - 1]):
            L = L if (L > pL or c[i - 1] < pL) else pL; U = U if (U < pU or c[i - 1] > pU) else pU
            if stv == pU: dr = -1 if c[i] > U else 1
            else: dr = 1 if c[i] < L else -1
        else: dr = 1
        stv = L if dr == -1 else U; up = U; lo = L; d[i] = -dr
    return d
def whale_hours():                                         # Astre/TV mantığı: H saatinde H−1:55 kaydı, tekrar atla, 720 s / 24 s takvim pencereleri
    d = pd.read_pickle('top5.pkl'); P = dict(zip(d.ms[d.sum_toptrader_long_short_ratio > 0], d.sum_toptrader_long_short_ratio[d.sum_toptrader_long_short_ratio > 0]))
    A = dict(zip(d.ms[d.count_toptrader_long_short_ratio > 0], d.count_toptrader_long_short_ratio[d.count_toptrader_long_short_ratio > 0]))
    H0 = (d.ms.min() // HMS) * HMS; hours = np.arange(H0, (d.ms.max() // HMS) * HMS + 2 * HMS, HMS); V, T = deque(), deque(); last = None; st = np.zeros(len(hours), np.int8)
    for k, H in enumerate(hours):
        p, a = P.get(H - 300000), A.get(H - 300000); ok = p is not None and a is not None; z = z24 = None; rep = False
        while T and T[0] <= H - 720 * HMS: T.popleft(); V.popleft()
        if ok:
            dv = np.log(p) - np.log(a); rep = last is not None and dv == last
            if not rep:
                V.append(dv); T.append(H)
                if len(V) >= 360: x = np.fromiter(V, float); sd = x.std(ddof=1); z = (dv - x.mean()) / sd if sd > 0 else None
            last = dv
        cnt = 0
        for tt in reversed(T):
            if tt <= H - 24 * HMS: break
            cnt += 1
        if ok and not rep and cnt >= 12: x = np.array(list(V)[-cnt:]); sd = x.std(ddof=1); z24 = (V[-1] - x.mean()) / sd if sd > 0 else None
        st[k] = ((z is not None and z >= 1) or (z24 is not None and z24 >= 1.5)) + 2 * ((z is not None and z <= -1) or (z24 is not None and z24 <= -1.5))
    return hours, st
_W = {}
def build(tf):
    R = raw(); b, x = R['b'], R['x']; m5 = 5 / tf
    kc, o, h, l, c, v = rs(b, tf); t = kc * tf * MS
    kx, _, _, _, xc, xv = rs(x, tf); btc = ffill(kx, xc, kc); btcV = ffill(kx, xv, kc)
    f5 = t // 300000 * 300000; oiP = ffill(R['oT'], R['oV'], f5 - 300000); oiP[f5 - 300000 < R['oT'][0]] = np.nan
    S = lambda a: pd.Series(a)
    def lag(a, k): r = np.full(len(a), np.nan); r[k:] = a[:-k]; return r
    N = lambda bars5: int(round(bars5 * m5))
    d288 = (c / lag(c, N(288)) - 1) * 100
    z = {}
    for w in (48, 96): r = S(c).rolling(N(w)); z[w] = ((c - r.mean()) / r.std(ddof=0)).to_numpy()
    r14 = rsi(c, N(14)); r7 = rsi(c, N(7))
    oP6 = lag(oiP, N(6)); okO = (oiP > 0) & (oP6 > 0); dOi6 = np.where(okO, (oiP / oP6 - 1) * 100, 0.0); dPx6 = (c / lag(c, N(6)) - 1) * 100
    o288 = lag(oiP, N(288)); dO288 = np.where(o288 > 0, (oiP / o288 - 1) * 100, np.nan)
    vr = (S(v).rolling(N(288)).sum() / (S(v).rolling(N(1728)).sum() / N(1728) * N(288))).to_numpy()
    rr = d288 - (btc / lag(btc, N(288)) - 1) * 100
    br = (btc / lag(btc, N(3)) - 1) * 100; ar = (c / lag(c, N(3)) - 1) * 100
    r30 = (S(v * c).rolling(N(6)).sum() / S(btcV * btc).rolling(N(6)).sum()).to_numpy(); r30m = S(r30).rolling(N(1728)).mean().to_numpy()
    nrv = np.where(~np.isnan(r30) & ~np.isnan(r30m) & (r30m > 0), r30 / r30m, 1.0)
    H = {}
    for htf in (15, 30, 60, 90, 1440): H[htf] = rs(b, htf)
    hx = rs(x, 60)
    adx = lambda htf: np.nan_to_num(hmap(dmi(H[htf][2], H[htf][3], H[htf][4], 14, 14)[2], H[htf][0], t, htf), nan=0.0)
    ema = lambda cc, n: S(cc).ewm(span=n, adjust=False).mean().to_numpy()
    bEf, bEs = hmap(ema(H[60][4], 5), H[60][0], t, 60), hmap(ema(H[60][4], 17), H[60][0], t, 60)
    xEf, xEs = hmap(ema(hx[4], 5), hx[0], t, 60), hmap(ema(hx[4], 20), hx[0], t, 60)
    bTr = np.where(np.isnan(bEf) | np.isnan(bEs), 0, np.sign(bEf - bEs)); xTr = np.where(np.isnan(xEf) | np.isnan(xEs), 0, np.sign(xEf - xEs)); trDir = np.where(bTr == xTr, bTr, 0).astype(np.int64)
    dc = H[1440][4]; mm = (dc / lag(dc, 30) - 1) * 100; dsig = np.where(np.isnan(mm), 0, np.where(mm >= 5, 1, np.where(mm <= -5, -1, 0))).astype(float)
    dmSig = np.nan_to_num(hmap(dsig, H[1440][0], t, 1440)).astype(np.int64)
    if 'h' not in _W: _W['h'] = whale_hours()
    hrs, wst = _W['h']; j = np.clip((t // HMS * HMS - hrs[0]) // HMS, 0, len(wst) - 1); ws = np.where(t >= hrs[0], wst[j], 0)
    return dict(tf=tf, m5=m5, t=t, h=h, l=l, c=c, d288=d288, z48=z[48], z96=z[96], r14=r14, r7=r7, okO=okO, dOi6=dOi6, dPx6=dPx6, dO288=dO288, vr=vr, rr=rr, br=br, ar=ar, nrv=nrv,
                dmSig=dmSig, adx15=adx(15), adx30=adx(30), adx1h=adx(60), trDir=trDir, H90=H[90], ws=ws)
@njit(cache=True)
def _ent(V, E, B, k, s, c, bi, hold, sl):
    if V[k] != 0 and ((c / E[k] - 1.0) * V[k] <= -sl / 100.0 or bi - B[k] >= hold): V[k] = 0
    if s != 0:
        if V[k] != s: V[k] = s; E[k] = c; B[k] = bi
        else: B[k] = bi
    return V[k]
@njit(cache=True)
def _osc(V, E, B, k, x, c, bi, lo, hi, exL, exS, sl, mb):
    if not np.isnan(x):
        if V[k] != 0:
            if (c / E[k] - 1.0) * V[k] <= -sl / 100.0 or bi - B[k] >= mb or (V[k] > 0 and x > exL) or (V[k] < 0 and x < exS): V[k] = 0
        elif x < lo: V[k] = 1; E[k] = c; B[k] = bi
        elif x > hi: V[k] = -1; E[k] = c; B[k] = bi
    return V[k]
@njit(cache=True)
def _rev(V, Hh, Ll, k, c, pct):
    if np.isnan(Hh[k]): Hh[k] = c; Ll[k] = c
    if V[k] >= 0:
        Hh[k] = max(Hh[k], c)
        if c <= Hh[k] * (1.0 - pct / 100.0): V[k] = -1; Ll[k] = c
    if V[k] <= 0:
        Ll[k] = min(Ll[k], c)
        if c >= Ll[k] * (1.0 + pct / 100.0): V[k] = 1; Hh[k] = c
    return V[k]
@njit(cache=True)
def core(c, d288, z48, z96, r14, r7, okO, dOi6, dPx6, dO288, vr, rr, br, ar, nrv, dmSig, adx15, adx30, adx1h, trDir, stDir, GL, GS, m5, useST, tsB, conf, dec, crev, hiA, hfac, xadd, xP, xw, ovr, tick):
    n = len(c); out = np.zeros(n, np.int64); sc = np.zeros(n)
    W = np.array([3., 3., 4., 0., 0., 1., 1., 3., .5, 1., 4., .5, 2., .5, 1.]); den = W.sum()
    eV = np.zeros(5, np.int64); eE = np.zeros(5); eB = np.zeros(5, np.int64); oV = np.zeros(3, np.int64); oE = np.zeros(3); oB = np.zeros(3, np.int64)
    rV = np.zeros(2, np.int64); rH = np.full(2, np.nan); rL = np.full(2, np.nan)
    vLiq = 0; vLiqEp = 0.0; vLiqB = 0; vBL = 0; vBLb = 0; vDM = 0; pos = 0; realDir = 0; rEp = 0.0; lockDir = 0; lockExt = 0.0; prevT = 0; rEb = 0; pend = 0; pendN = 0
    P = np.zeros(15)
    for i in range(n):
        x = c[i]; d = d288[i]; okd = not np.isnan(d)
        liqL = okO[i] and dPx6[i] <= -1.5 and dOi6[i] <= -2.0; liqS = okO[i] and dPx6[i] >= 1.5 and dOi6[i] <= -2.0
        if vLiq != 0:
            if (x / vLiqEp - 1.0) * vLiq <= -0.05 or i - vLiqB >= 144 * m5 or (vLiq > 0 and r7[i] > 65.0) or (vLiq < 0 and r7[i] < 35.0): vLiq = 0
            elif vLiq > 0 and liqS: vLiq = -1; vLiqEp = x; vLiqB = i
            elif vLiq < 0 and liqL: vLiq = 1; vLiqEp = x; vLiqB = i
        elif liqL: vLiq = 1; vLiqEp = x; vLiqB = i
        elif liqS: vLiq = -1; vLiqEp = x; vLiqB = i
        if vBL != 0 and i - vBLb >= 24 * m5: vBL = 0
        if not np.isnan(br[i]) and not np.isnan(ar[i]):
            if br[i] >= 2.5 and ar[i] < br[i]: vBL = 1; vBLb = i
            elif br[i] <= -2.5 and ar[i] > br[i]: vBL = -1; vBLb = i
        if dmSig[i] != 0: vDM = dmSig[i]
        sV1 = 0 if not okd else (1 if d <= -3.5 else (-1 if d >= 6.0 else 0))
        fd = -1 if d > 0 else (1 if d < 0 else 0)
        sOIK = fd if (okd and not np.isnan(dO288[i]) and abs(d) >= 4.0 and dO288[i] >= 6.0) else 0
        sHVX = fd if (okd and not np.isnan(vr[i]) and abs(d) >= 5.0 and vr[i] >= 1.5) else 0
        sREL = 0 if np.isnan(rr[i]) else ((-1 if rr[i] > 0 else 1) if abs(rr[i]) >= 6.0 else 0)
        s15 = fd if (okd and nrv[i] >= 1.5 and abs(d) >= 4.0) else 0
        P[0] = _ent(eV, eE, eB, 0, sV1, x, i, 648 * m5, 3.0)
        P[1] = _osc(oV, oE, oB, 0, z48[i], x, i, -4.0, 4.0, 0.5, -0.5, 8.0, 288 * m5)
        P[2] = _osc(oV, oE, oB, 1, z96[i], x, i, -3.5, 3.5, 0.0, 0.0, 8.0, 288 * m5)
        P[5] = _osc(oV, oE, oB, 2, r14[i], x, i, 20.0, 80.0, 45.0, 55.0, 3.0, 144 * m5)
        P[6] = vLiq
        P[7] = _ent(eV, eE, eB, 1, sOIK, x, i, 144 * m5, 10.0)
        P[8] = _ent(eV, eE, eB, 2, sHVX, x, i, 288 * m5, 10.0)
        P[9] = _ent(eV, eE, eB, 3, sREL, x, i, 288 * m5, 10.0)
        P[10] = vBL
        P[11] = _rev(rV, rH, rL, 0, crev[i], 3.0)
        P[12] = _rev(rV, rH, rL, 1, crev[i], 8.0)
        P[13] = vDM
        P[14] = _ent(eV, eE, eB, 4, s15, x, i, 288 * m5, 10.0)
        s = ((W * P).sum() + xw * xP[i]) / den; sc[i] = s
        onL = 0.15 if (trDir[i] == -1 and adx1h[i] >= 20.0) else 0.04; onS = 0.15 if (trDir[i] == 1 and adx1h[i] >= 20.0) else 0.04
        sH = stDir[i] if (useST == 1 and adx30[i] >= 25.0) else 0; sT = stDir[i] if (useST == 1 and adx30[i] >= 30.0) else 0
        offNow = -0.15 if (pos != 0 and sH == pos) else 0.0
        if sT == -1: onL = max(onL, 0.12)
        if sT == 1: onS = max(onS, 0.12)
        if hiA[i]: onL = onL * hfac; onS = onS * hfac; offNow = offNow + xadd
        want = pos
        if pos != 0 and s * pos < offNow: want = 0
        if s >= onL: want = 1
        elif s <= -onS: want = -1
        if conf > 1:                                            # 1 dk organizasyonu: yön değişimi conf bar üst üste aynı kalmalı
            if want != pos:
                if want == pend: pendN += 1
                else: pend = want; pendN = 1
                if pendN < conf: want = pos
                else: pendN = 0
            else: pend = pos; pendN = 0
        if not dec[i]: want = pos
        pos = want; tgt = pos
        if ovr[i] != 0: tgt = ovr[i]
        if tgt != prevT: lockDir = 0; prevT = tgt
        wD = tgt
        if realDir != 0 and wD == realDir:
            pl = (x / rEp - 1.0) * realDir * 100.0
            if pl <= -7.0: lockDir = realDir; lockExt = x; wD = 0
            elif i - rEb >= tsB and pl <= -1.0 and adx15[i] < 25.0: lockDir = realDir; lockExt = x; wD = 0
        if lockDir != 0 and wD == lockDir:
            lockExt = min(lockExt, x) if lockDir > 0 else max(lockExt, x)
            if (x / lockExt - 1.0) * lockDir * 100.0 >= 4.0: lockDir = 0
            else: wD = 0
        if wD != 0 and wD != realDir and (nrv[i] < 0.1 or (wD == 1 and GL[i]) or (wD == -1 and GS[i])): wD = 0
        if wD != realDir: realDir = wD; rEp = x + tick * wD; rEb = i
        out[i] = realDir
    return out, sc
def run(Dd, useST=1, whale='tam', stf=2.5, conf=1, tsB=None, clk=1, X=None, hi=None, hfac=1.0, xadd=0.0, xGL=None, xGS=None, xP=None, xw=0.0, ovr=None):
    H90 = Dd['H90']; std = np.nan_to_num(hmap(st_dir(H90[2], H90[3], H90[4], 10, stf).astype(float), H90[0], Dd['t'], 90)).astype(np.int64)
    ws = Dd['ws'].copy()
    if whale == 'tv': ws[Dd['t'] < TV0] = 0
    elif whale == 'yok': ws[:] = 0
    GL = (ws == 1) | (ws == 3); GS = (ws == 2) | (ws == 3); m5 = Dd['m5']
    if xGL is not None: GL = GL | xGL; GS = GS | xGS
    dec = ((Dd['t'] // MS) % clk) == clk - 1
    if X: Dd = dict(Dd, **X)
    out, sc = core(Dd['c'], Dd['d288'], Dd['z48'], Dd['z96'], Dd['r14'], Dd['r7'], Dd['okO'], Dd['dOi6'], Dd['dPx6'], Dd['dO288'], Dd['vr'], Dd['rr'], Dd['br'], Dd['ar'], Dd['nrv'],
                   Dd['dmSig'], Dd['adx15'], Dd['adx30'], Dd['adx1h'], Dd['trDir'], std, GL, GS, float(m5), useST, float(round(288 * m5)) if tsB is None else tsB, conf, dec, Dd.get('crev', Dd['c']), Dd['c'] < 0 if hi is None else hi, hfac, xadd, np.zeros(len(Dd['c']), np.int64) if xP is None else xP, xw, np.zeros(len(Dd['c']), np.int64) if ovr is None else ovr, TICK)
    return out
def trades(Dd, out):
    c = Dd['c']; ch = np.flatnonzero(np.diff(np.r_[0, out]) != 0); rows = []; cur = None
    for i in ch:
        if cur is not None:
            d, eb, ep, q = cur; xp = c[i] - TICK * d; rows.append((eb, i, d, ep, q, d * (xp - ep) * q - 0.0004 * q * (ep + xp))); cur = None
        if out[i] != 0: d = out[i]; ep = c[i] + TICK * d; cur = (i, ep, 500.0 / ep) if False else (d, i, ep, 500.0 / ep)
    R = np.array(rows, dtype=[('eb', 'i8'), ('xb', 'i8'), ('dir', 'i8'), ('ep', 'f8'), ('q', 'f8'), ('pnl', 'f8')])
    return R, cur
def mtm30(Dd, R, op):
    c, h, l, t = Dd['c'], Dd['h'], Dd['l'], Dd['t']; n = len(c); cum = np.zeros(n); np.add.at(cum, R['xb'], R['pnl']); cum = np.cumsum(cum)
    opn = np.zeros(n); wo = np.zeros(n)
    for eb, xb, d, ep, q in zip(R['eb'], R['xb'], R['dir'], R['ep'], R['q']):
        s = slice(eb + 1, xb); opn[s] = (c[s] - ep) * q * d; wo[s] = ((l[s] if d > 0 else h[s]) - ep) * q * d
    if op is not None:
        d, eb, ep, q = op; s = slice(eb + 1, n); opn[s] = (c[s] - ep) * q * d; wo[s] = ((l[s] if d > 0 else h[s]) - ep) * q * d
    ix = pd.to_datetime(t + Dd['tf'] * MS, unit='ms'); E = pd.Series(cum + opn, index=ix).resample('30min', label='right', closed='right').last().ffill()
    Wn = pd.Series(cum + wo, index=ix).resample('30min', label='right', closed='right').min().reindex(E.index).fillna(E)
    pk = E.cummax(); uw = pk - E; uwW = pk - Wn; yrs = (t[-1] - t[0]) / 3.15576e10; net = E.iloc[-1]; dd = E.diff().fillna(0); k = np.sqrt(48 * 365.25)
    return dict(MaxDD=round(uwW.max()), Cal=round(net / yrs / uw.max(), 2), P=f"{uw.quantile(.95):.0f}/{uw.quantile(.98):.0f}/{uw.quantile(.99):.0f}", SS=f"{dd.mean() / dd.std() * k:.2f}/{dd.mean() / dd[dd < 0].std() * k:.2f}")
def report(Dd, out, lab):
    R, op = trades(Dd, out); t = Dd['t']; p = R['pnl']; et = t[R['eb']]; xt = t[R['xb']]; e = t[-1]
    o = dict(t=lab, net=round(p.sum()), n=len(p), PF=round(p[p > 0].sum() / -p[p < 0].sum(), 3), IS=round(p[et < SPL].sum()), OOS=round(p[et >= SPL].sum()),
             L12=round(p[xt >= e - 365 * 86400000].sum()), L200=round(p[xt >= e - 200 * 86400000].sum()), kom=round((0.0004 * R['q'] * (R['ep'] * 2)).sum()))
    o.update(mtm30(Dd, R, op)); return o
