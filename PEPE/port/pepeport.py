# CYBORG PEPE 5M SWING V11.7 mantığının birebir Python/numba portu — herhangi bir coin için (ZEC taşınabilirlik / ERR testi)
# Pine sırası korunur: sanal motorlar → hedef → durum makinesi → gölge → zarar serisi kesici → emir (kapanışta, 1 tick kayma, %0,04 x2)
import numpy as np, pandas as pd, glob, numba
TICK = 0.01; FEE = 0.0004; CAP = 500.0

@numba.njit(cache=True)
def rma(x, n):
    out = np.full(len(x), np.nan); s = 0.0; k = 0; prev = np.nan
    for i in range(len(x)):
        v = x[i]
        if np.isnan(prev):
            if not np.isnan(v):
                s += v; k += 1
                if k == n: prev = s / n; out[i] = prev
        else:
            if not np.isnan(v): prev = (prev * (n - 1) + v) / n
            out[i] = prev
    return out

def dmi(h, l, c, n=14, m=14):
    up = np.r_[np.nan, h[1:] - h[:-1]]; dn = np.r_[np.nan, l[:-1] - l[1:]]
    pdm = np.where((up > dn) & (up > 0), up, 0.0); mdm = np.where((dn > up) & (dn > 0), dn, 0.0); pdm[0] = np.nan; mdm[0] = np.nan
    tr = np.r_[np.nan, np.maximum(h[1:] - l[1:], np.maximum(abs(h[1:] - c[:-1]), abs(l[1:] - c[:-1])))]
    trur = rma(tr, n); p = pd.Series(100 * rma(pdm, n) / trur).ffill().values; mi = pd.Series(100 * rma(mdm, n) / trur).ffill().values
    s = p + mi; adx = 100 * rma(abs(p - mi) / np.where(s == 0, 1, s), m); return p, mi, adx

def rsi(c, n):
    ch = np.r_[np.nan, np.diff(c)]; u = rma(np.maximum(ch, 0), n); d = rma(np.maximum(-ch, 0), n)
    return np.where(d == 0, 100.0, np.where(u == 0, 0.0, 100 - 100 / (1 + u / np.where(d == 0, 1, d))))

def atr_tr(h, l, c, n):
    tr = np.r_[h[0] - l[0], np.maximum(h[1:] - l[1:], np.maximum(abs(h[1:] - c[:-1]), abs(l[1:] - c[:-1])))]; return rma(tr, n)

def htf(df5, rule, fn):
    """fn(DataFrame HTF) -> np.array; değer önceki TAMAMLANMIŞ HTF barı (lookahead_on + [1])."""
    H = df5.resample(rule, label='left', closed='left').agg({'o': 'first', 'h': 'max', 'l': 'min', 'c': 'last'}).dropna()
    v = pd.Series(fn(H), H.index).shift(1); key = df5.index.floor(rule)
    return v.reindex(key).values

@numba.njit(cache=True)
def core(c, tms, adx5, adx30, momA, momB, momM, rsiV, rsi7, btcFill, ahfPct, dOi, dPx, dOi12, dPx12, bz6, bz12, bPx12, trendSh, P, TK):
    n = len(c)
    (aAdx30, aAdx5, aRise, aMom, bAdx30, bRise, bMom, rsiLo, rsiEx, rsiSL, rsiMax, mMom, mAdx30, mRise, mHold, minHold,
     trimX, trimY, trimBars, trimCont, ahfK, ahfSL, ahfBars, shMax, cbLoss, cbAmt, cbDays, liqPx, liqOi, liqSL, liqBars, liqExit,
     j4Px, j4Oi, j4Bz, j2Px, j2Oi, j2Bz, j2Btc, j2SL, j2Bars, useJ) = P
    vV2 = 0; vV2b = 0; vV3 = 0; vV3b = 0; vR = 0; vREp = 0.0; vRb = 0; vMid = 0; vMidb = 0; vLiq = 0; vLiqEp = 0.0; vLiqb = 0
    vJ2 = 0; vJ2Ep = 0.0; vJ2b = 0; midRun = 0; midC = 0; prevMid = 0; prevTarget = 0
    shDir = 0; shEp = np.nan; stMode = 0; trimPx = 0.0; trimBar = 0; trimSide = 0; cntDir = 0; cntPx = 0.0; cntBar = 0
    cbSt = 0; cbSum = 0.0; cbUntil = 0; cbPaused = False; cbWait = False; cbLast = 0
    pos = 0; ep = 0.0; q = 0.0; ei = 0; real = 0.0; pend = 0.0; hasPend = False
    eq = np.zeros(n); T = np.zeros((n // 2, 6)); nt = 0
    for i in range(1, n):
        # kapanmış işlem (önceki barın emri) → kesici sayacı
        if hasPend:
            if pend < 0: cbSt += 1; cbSum -= pend
            else: cbSt = 0; cbSum = 0.0
            hasPend = False
        oiOk = not np.isnan(dOi[i])
        liqL = oiOk and dPx[i] <= -liqPx and dOi[i] <= -liqOi; liqS = oiOk and dPx[i] >= liqPx and dOi[i] <= -liqOi
        # likidasyon sanal motoru
        if vLiq != 0:
            lr = (c[i] / vLiqEp - 1.0) * vLiq
            if lr <= -liqSL / 100.0 or i - vLiqb >= liqBars or (vLiq > 0 and rsi7[i] > liqExit) or (vLiq < 0 and rsi7[i] < 100.0 - liqExit): vLiq = 0
            elif vLiq > 0 and liqS: vLiq = -1; vLiqEp = c[i]; vLiqb = i
            elif vLiq < 0 and liqL: vLiq = 1; vLiqEp = c[i]; vLiqb = i
        else:
            if liqL: vLiq = 1; vLiqEp = c[i]; vLiqb = i
            elif liqS: vLiq = -1; vLiqEp = c[i]; vLiqb = i
        okA = not (np.isnan(adx5[i]) or np.isnan(adx30[i]) or np.isnan(adx5[i - int(aRise)]))
        aG = okA and adx30[i] > aAdx30 and adx5[i] > aAdx5 and adx5[i] > adx5[i - int(aRise)]
        aL = aG and momA[i] >= aMom; aS = aG and momA[i] <= -aMom
        okB = not (np.isnan(adx5[i]) or np.isnan(adx30[i]) or np.isnan(adx5[i - int(bRise)]))
        bG = okB and adx30[i] > bAdx30 and adx5[i] > adx5[i - int(bRise)]
        bL = bG and momB[i] >= bMom; bS = bG and momB[i] <= -bMom
        canV2 = vV2 == 0 or i - vV2b >= minHold
        if aL and vV2 <= 0 and canV2: vV2 = 1; vV2b = i
        elif aS and vV2 >= 0 and canV2: vV2 = -1; vV2b = i
        canV3 = vV3 == 0 or i - vV3b >= minHold
        if (aL or bL) and vV3 <= 0 and canV3: vV3 = 1; vV3b = i
        elif (aS or bS) and vV3 >= 0 and canV3: vV3 = -1; vV3b = i
        if not np.isnan(rsiV[i]):
            if vR != 0:
                rr = (c[i] / vREp - 1.0) * vR
                if rr <= -rsiSL / 100.0 or i - vRb >= rsiMax or (vR > 0 and rsiV[i] > rsiEx) or (vR < 0 and rsiV[i] < 100.0 - rsiEx): vR = 0
            else:
                if rsiV[i] < rsiLo: vR = 1; vREp = c[i]; vRb = i
                elif rsiV[i] > 100.0 - rsiLo: vR = -1; vREp = c[i]; vRb = i
        okM = not (np.isnan(adx5[i]) or np.isnan(adx30[i]) or np.isnan(adx5[i - int(mRise)]))
        mG = okM and adx30[i] > mAdx30 and adx5[i] > adx5[i - int(mRise)]
        mL = mG and momM[i] >= mMom; mS = mG and momM[i] <= -mMom
        canM = vMid == 0 or i - vMidb >= mHold
        if mL and vMid <= 0 and canM: vMid = 1; vMidb = i
        elif mS and vMid >= 0 and canM: vMid = -1; vMidb = i
        j2L = useJ and not np.isnan(dOi12[i]) and dPx12[i] >= j2Px and dOi12[i] >= j2Oi and bz12[i] >= j2Bz and bPx12[i] >= j2Btc
        j2S = useJ and not np.isnan(dOi12[i]) and dPx12[i] <= -j2Px and dOi12[i] >= j2Oi and bz12[i] >= j2Bz and bPx12[i] <= -j2Btc
        if vJ2 != 0:
            jr = (c[i] / vJ2Ep - 1.0) * vJ2
            if jr <= -j2SL / 100.0 or i - vJ2b >= j2Bars: vJ2 = 0
            elif vJ2 > 0 and j2S: vJ2 = -1; vJ2Ep = c[i]; vJ2b = i
            elif vJ2 < 0 and j2L: vJ2 = 1; vJ2Ep = c[i]; vJ2b = i
        else:
            if j2L: vJ2 = 1; vJ2Ep = c[i]; vJ2b = i
            elif j2S: vJ2 = -1; vJ2Ep = c[i]; vJ2b = i
        midRun = midRun + 1 if vMid == prevMid else 1
        prevMid = vMid
        if midRun >= 3: midC = vMid
        s = vV2 + vV3; collide = 1 if s > 0 else (-1 if s < 0 else 0)
        target = vLiq if vLiq != 0 else (vR if vR != 0 else (vJ2 if vJ2 != 0 else (collide if collide != 0 else (btcFill[i] if btcFill[i] != 0 else midC))))
        coreEvt = target != prevTarget; prevTarget = target
        exJ = 0
        if useJ and oiOk and not np.isnan(bz6[i]):
            if dPx[i] >= j4Px and dOi[i] <= -j4Oi and bz6[i] <= -j4Bz: exJ = 1
            elif dPx[i] <= -j4Px and dOi[i] <= -j4Oi and bz6[i] <= -j4Bz: exJ = -1
        posDir = shDir; ahfThr = ahfK * (0.0 if np.isnan(ahfPct[i]) else ahfPct[i]) / 100.0; wantDir = posDir
        if coreEvt:
            stMode = 0; wantDir = target
        elif stMode == 0:
            wantDir = target
            if posDir != 0 and posDir == target:
                adv = (c[i] / shEp - 1.0) * (-posDir)
                if ahfThr > 0 and adv >= ahfThr:
                    wantDir = -posDir; stMode = 1; cntDir = posDir; cntPx = c[i]; cntBar = i
                elif exJ == posDir:
                    wantDir = 0; stMode = 2; trimPx = c[i]; trimBar = i; trimSide = posDir
                elif ((posDir > 0 and rsi7[i] >= trimX) or (posDir < 0 and rsi7[i] <= 100.0 - trimX)) and trendSh[i] * posDir < shMax:
                    wantDir = 0; stMode = 2; trimPx = c[i]; trimBar = i; trimSide = posDir
        elif stMode == 1:
            cg = (c[i] / cntPx - 1.0) * (-cntDir)
            if cg <= -ahfSL / 100.0 or i - cntBar >= ahfBars: stMode = 0; wantDir = target
            else: wantDir = -cntDir
        else:
            if trimSide > 0: bi = rsi7[i] <= trimY or i - trimBar >= trimBars or c[i] >= trimPx * (1.0 + trimCont / 100.0)
            else: bi = rsi7[i] >= 100.0 - trimY or i - trimBar >= trimBars or c[i] <= trimPx * (1.0 - trimCont / 100.0)
            if bi: stMode = 0; wantDir = target
            else: wantDir = 0
        if wantDir != shDir:
            shDir = wantDir; shEp = c[i] + TK * wantDir if wantDir != 0 else np.nan
        realWant = wantDir
        if cbPaused and tms[i] > cbUntil:
            cbPaused = False; cbSt = 0; cbSum = 0.0; cbWait = True; cbLast = wantDir
        if cbPaused: realWant = 0
        elif cbSt >= cbLoss and cbSum >= cbAmt:
            realWant = 0; cbPaused = True; cbUntil = tms[i] + int(round(cbDays * 86400000))
        elif cbWait:
            if wantDir != cbLast: cbWait = False
            else: realWant = 0
        # emirler (kapanışta)
        if realWant != pos:
            if pos != 0:
                xp = c[i] - TK * pos if realWant == 0 else c[i] + TK * realWant
                pnl = q * (xp - ep) * pos - FEE * q * (ep + xp); real += pnl
                T[nt, 0] = ei; T[nt, 1] = i; T[nt, 2] = pos; T[nt, 3] = ep; T[nt, 4] = xp; T[nt, 5] = pnl; nt += 1
                pend = pnl; hasPend = True; pos = 0
            if realWant != 0:
                pos = realWant; ep = c[i] + TK * pos; q = CAP / c[i]; ei = i
        eq[i] = real + (q * (c[i] - ep) * pos - FEE * q * ep if pos != 0 else 0.0)
    return eq, T[:nt]

PEPE_P = (28.0, 20.0, 12, 3.0, 25.0, 6, 2.5, 10.0, 50.0, 8.0, 144, 4.0, 18.0, 3, 12, 24, 90.0, 60.0, 12, 3.0, 4.5, 2.0, 144, 2.0,
          2, 100.0, 1.0, 4.0, 4.0, 8.0, 72, 55.0, 3.0, 3.0, 1.5, 4.0, 4.0, 1.0, 0.5, 3.0, 48, True)

def build(sym_df1m, oi1m, btc1m, btcoi5):
    """sym_df1m: 1m OHLC (index UTC), oi1m: 1m OI serisi (aynı index), btc1m: BTC 1m close/high/low (index), btcoi5: BTC OI 5m (bitiş etiketli)."""
    d5 = sym_df1m.resample('5min', label='left', closed='left').agg({'o': 'first', 'h': 'max', 'l': 'min', 'c': 'last'}).dropna()
    o, h, l, c = (d5[k].values for k in 'ohlc'); t = d5.index; tms = np.asarray((t - pd.Timestamp('1970-01-01')) // pd.Timedelta('1ms'), dtype=np.int64)
    _, _, adx5 = dmi(h, l, c)
    adx30 = htf(d5, '30min', lambda H: dmi(H.h.values, H.l.values, H.c.values)[2])
    ahf = htf(d5, '60min', lambda H: atr_tr(H.h.values, H.l.values, H.c.values, 14) / H.c.values * 100)
    b5 = btc1m.resample('5min', label='left', closed='left').agg({'o': 'first', 'h': 'max', 'l': 'min', 'c': 'last'}).dropna().reindex(t).ffill()
    def bf(H):
        p_, m_, a_ = dmi(H.h.values, H.l.values, H.c.values); return np.where(a_ > 30, np.sign(p_ - m_), 0.0)
    btcFill = np.nan_to_num(htf(b5, '240min', bf)).astype(np.float64)
    mom = lambda k: np.r_[np.full(k, np.nan), c[k:] / c[:-k] - 1] * 100
    oiR = oi1m.reindex(t).values; oiP = np.r_[np.nan, oiR[:-1]]
    sh = lambda x, k: np.r_[np.full(k, np.nan), x[:-k]]
    dOi = np.where((~np.isnan(oiP)) & (sh(oiP, 6) > 0), (oiP / sh(oiP, 6) - 1) * 100, np.nan)
    dOi12 = np.where((~np.isnan(oiP)) & (sh(oiP, 12) > 0), (oiP / sh(oiP, 12) - 1) * 100, np.nan)
    bo = btcoi5.reindex(t).values; bP = np.r_[np.nan, bo[:-1]]
    bD6 = np.log(bP) - np.log(sh(bP, 6)); bD12 = np.log(bP) - np.log(sh(bP, 12))
    sd = lambda x: pd.Series(np.nan_to_num(x)).rolling(864).std(ddof=0).values
    bz6 = np.where(~np.isnan(bD6) & (sd(bD6) > 0), bD6 / sd(bD6), 0.0); bz12 = np.where(~np.isnan(bD12) & (sd(bD12) > 0), bD12 / sd(bD12), 0.0)
    bc = b5.c.values; bPx12 = np.nan_to_num(np.where(sh(bc, 12) > 0, (bc / sh(bc, 12) - 1) * 100, 0.0))
    br = np.r_[np.nan, c[1:] / c[:-1] - 1]; s_ = pd.Series(br); trendSh = np.nan_to_num((s_.rolling(864).mean() / s_.rolling(864).std(ddof=1) * np.sqrt(864)).values)
    A = dict(c=c, tms=tms, adx5=adx5, adx30=adx30, momA=mom(4), momB=mom(48), momM=mom(24), rsiV=rsi(c, 14), rsi7=np.nan_to_num(rsi(c, 7), nan=50.0),
             btcFill=btcFill, ahfPct=ahf, dOi=dOi, dPx=mom(6), dOi12=dOi12, dPx12=mom(12), bz6=bz6, bz12=bz12, bPx12=bPx12, trendSh=trendSh)
    return d5, A

def run(A, P=PEPE_P, tick=0.01):
    return core(*(np.ascontiguousarray(A[k], dtype=np.int64 if k == 'tms' else np.float64) for k in
                  ('c', 'tms', 'adx5', 'adx30', 'momA', 'momB', 'momM', 'rsiV', 'rsi7', 'btcFill', 'ahfPct', 'dOi', 'dPx', 'dOi12', 'dPx12', 'bz6', 'bz12', 'bPx12', 'trendSh')), P, tick)
