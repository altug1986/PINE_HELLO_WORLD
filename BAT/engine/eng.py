import numpy as np, numba as nb, pandas as pd
NAN = np.nan
TICK = 1e-5; CASH = 500.0; COMM = 0.0004

@nb.njit(cache=True)
def rma(x, n):
    out = np.full(len(x), np.nan); s = np.nan; cnt = 0; acc = 0.0
    for i in range(len(x)):
        v = x[i]
        if np.isnan(s):
            if np.isnan(v):
                cnt = 0; acc = 0.0
            else:
                cnt += 1; acc += v
                if cnt >= n:
                    s = acc / n  # (seed approx: sum of last n consecutive)
                    out[i] = s
        else:
            if not np.isnan(v):
                s = (v + (n - 1) * s) / n
            out[i] = s
    return out

@nb.njit(cache=True)
def trr(h, l, c, hna):
    n = len(c); tr = np.full(n, np.nan)
    for i in range(n):
        if i == 0:
            tr[i] = h[i] - l[i] if hna else np.nan
        else:
            tr[i] = max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1]))
    return tr

@nb.njit(cache=True)
def dmi(h, l, c, dl, al):
    n = len(c); pdm = np.full(n, np.nan); mdm = np.full(n, np.nan)
    for i in range(1, n):
        up = h[i] - h[i-1]; dn = l[i-1] - l[i]
        pdm[i] = up if (up > dn and up > 0) else 0.0
        mdm[i] = dn if (dn > up and dn > 0) else 0.0
    tr = rma(trr(h, l, c, False), dl); rp = rma(pdm, dl); rm = rma(mdm, dl)
    p = np.full(n, np.nan); m = np.full(n, np.nan); lp = np.nan; lm = np.nan
    for i in range(n):
        if not np.isnan(rp[i]) and not np.isnan(tr[i]) and tr[i] != 0:
            lp = 100 * rp[i] / tr[i]; lm = 100 * rm[i] / tr[i]
        p[i] = lp; m[i] = lm
    dx = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(p[i]):
            s = p[i] + m[i]
            dx[i] = abs(p[i] - m[i]) / (1.0 if s == 0 else s)
    return p, m, 100 * rma(dx, al)

@nb.njit(cache=True)
def rsi(c, n):
    L = len(c); u = np.full(L, np.nan); d = np.full(L, np.nan)
    for i in range(1, L):
        ch = c[i] - c[i-1]; u[i] = max(ch, 0.0); d[i] = max(-ch, 0.0)
    ru = rma(u, n); rd = rma(d, n); out = np.full(L, np.nan)
    for i in range(L):
        if not np.isnan(ru[i]):
            out[i] = 100.0 if rd[i] == 0 else (0.0 if ru[i] == 0 else 100 - 100 / (1 + ru[i] / rd[i]))
    return out

def htf_map(vals, hk, bmap):
    # lookahead_on + expr[1]: value of the HTF bar before the one containing the chart bar
    j = np.searchsorted(hk, bmap, side='right') - 1
    exact = (j >= 0) & (hk[np.clip(j, 0, None)] == bmap)
    jj = np.where(exact, j - 1, j)
    return np.where(jj >= 0, vals[np.clip(jj, 0, None)], np.nan)

def roll_sharpe(c, n):
    r = np.r_[np.nan, c[1:] / c[:-1] - 1]
    s = pd.Series(r)
    return np.nan_to_num((s.rolling(n).mean() / s.rolling(n).std(ddof=1) * np.sqrt(n)).to_numpy(), nan=0.0)

def load(tf):
    z = dict(np.load(f'd{tf}.npz'))
    o, h, l, c = z['o'], z['h'], z['l'], z['c']
    D = dict(t=z['t'], o=o, h=h, l=l, c=c, v=z['v'], xc=z['xc'], oi=z['oi'])
    D['adx5'] = dmi(h, l, c, 14, 14)[2]
    b = z['b30']; D['adx30'] = htf_map(dmi(b[1], b[2], b[3], 14, 14)[2], z['b30_k'], z['b30_map'])
    b = z['x240']; p, m, a = dmi(b[1], b[2], b[3], 14, 14)
    D['badx4'] = htf_map(a, z['x240_k'], z['x240_map']); D['bdi4'] = htf_map(p - m, z['x240_k'], z['x240_map'])
    b = z['b60']; at = rma(trr(b[1], b[2], b[3], True), 14) / b[3] * 100
    D['atr1h'] = htf_map(at, z['b60_k'], z['b60_map'])
    D['rsi14'] = rsi(c, 14); D['rsi7'] = rsi(c, 7)
    D['tsh'] = roll_sharpe(c, 864)
    return D

# ---------------- PARAMS ----------------
PN = ['aA30','aA5','aRise','aLb','aMom','bA30','bRise','bLb','bMom','minHold',
      'rLo','rExit','rSL','rBars','mLb','mMom','mA30','mRise','mHold','mConf','btcAdx',
      'trX','trY','trBars','trCont','ahfK','ahfSL','ahfBars','shMax','cbLoss','cbAmt','cbDays',
      'oiLen','liqPx','liqOi','liqSL','liqBars','liqExit']
PEPE = dict(aA30=28,aA5=20,aRise=12,aLb=4,aMom=3,bA30=25,bRise=6,bLb=48,bMom=2.5,minHold=24,
      rLo=10,rExit=50,rSL=8,rBars=144,mLb=24,mMom=4,mA30=18,mRise=3,mHold=12,mConf=3,btcAdx=30,
      trX=90,trY=60,trBars=12,trCont=3,ahfK=4.5,ahfSL=2,ahfBars=144,shMax=2.0,cbLoss=2,cbAmt=100,cbDays=1,
      oiLen=6,liqPx=4,liqOi=4,liqSL=8,liqBars=72,liqExit=55)
def P(**kw):
    d = dict(PEPE); d.update(kw); return np.array([d[k] for k in PN], float)
# flags: useLiq,useR,useCol,useBtc,useMid,useTrim,useCnt,useCb,colMode(0 collide,1 V2,2 V3)
FULL = np.array([1,1,1,1,1,1,1,1,0], np.int64)

@nb.njit(cache=True)
def run(t, h, l, c, adx5, adx30, badx4, bdi4, atr1h, rsi14, rsi7, tsh, oiP, P, F, tick):
    n = len(c)
    aA30,aA5,aRise,aLb,aMom = P[0],P[1],int(P[2]),int(P[3]),P[4]
    bA30,bRise,bLb,bMom,minHold = P[5],int(P[6]),int(P[7]),P[8],int(P[9])
    rLo,rExit,rSL,rBars = P[10],P[11],P[12],int(P[13])
    mLb,mMom,mA30,mRise,mHold,mConf = int(P[14]),P[15],P[16],int(P[17]),int(P[18]),int(P[19])
    btcAdx = P[20]; trX,trY,trBars,trCont = P[21],P[22],int(P[23]),P[24]
    ahfK,ahfSL,ahfBars,shMax = P[25],P[26],int(P[27]),P[28]
    cbLoss,cbAmt,cbDays = int(P[29]),P[30],P[31]
    oiLen,liqPx,liqOi,liqSL,liqBars,liqExit = int(P[32]),P[33],P[34],P[35],int(P[36]),P[37]
    uL,uR,uC,uB,uM,uT,uK,uCb,cm = F[0],F[1],F[2],F[3],F[4],F[5],F[6],F[7],F[8]
    tgtA = np.zeros(n, np.int64); wantA = np.zeros(n, np.int64); realA = np.zeros(n, np.int64)
    # trades
    eb = np.zeros(n, np.int64); xb = np.zeros(n, np.int64); dr = np.zeros(n, np.int64)
    ep = np.zeros(n); xp = np.zeros(n); qt = np.zeros(n); pl = np.zeros(n); nt = 0
    pos = 0; pEp = 0.0; pQ = 0.0; pEb = 0
    vLiq = 0; vLiqEp = 0.0; vLiqBar = 0; vV2 = 0; vV2Bar = 0; vV3 = 0; vV3Bar = 0
    vR = 0; vREp = 0.0; vRBar = 0; vMid = 0; vMidBar = 0; vMidPrev = 0; midRun = 0; midC = 0
    tgtPrev = 0; shDir = 0; shEp = np.nan; stMode = 0; trimPx = 0.0; trimBar = 0; trimSide = 0
    cntDir = 0; cntPx = 0.0; cntBar = 0
    cbSt = 0; cbSum = 0.0; cbSeen = 0; cbUntil = 0; cbPaused = False; cbWait = False; cbLast = 0
    for i in range(n):
        ci = c[i]
        # ---- liquidation reversal (OI)
        if uL == 1 and i >= oiLen:
            o0 = oiP[i]; o1 = oiP[i - oiLen]
            ok = (not np.isnan(o0)) and (not np.isnan(o1)) and o1 > 0
            dOi = (o0 / o1 - 1) * 100 if ok else 0.0
            dPx = (ci / c[i - oiLen] - 1) * 100
            liqL = ok and dPx <= -liqPx and dOi <= -liqOi
            liqS = ok and dPx >= liqPx and dOi <= -liqOi
            r7 = rsi7[i]
            if vLiq != 0:
                lRet = (ci / vLiqEp - 1) * vLiq
                if lRet <= -liqSL / 100 or i - vLiqBar >= liqBars or (vLiq > 0 and r7 > liqExit) or (vLiq < 0 and r7 < 100 - liqExit):
                    vLiq = 0
                elif vLiq > 0 and liqS:
                    vLiq = -1; vLiqEp = ci; vLiqBar = i
                elif vLiq < 0 and liqL:
                    vLiq = 1; vLiqEp = ci; vLiqBar = i
            else:
                if liqL:
                    vLiq = 1; vLiqEp = ci; vLiqBar = i
                elif liqS:
                    vLiq = -1; vLiqEp = ci; vLiqBar = i
        # ---- collision motors
        a30 = adx30[i]; a5 = adx5[i]
        aL = False; aS = False; bL = False; bS = False
        if i >= aRise and i >= aLb and not np.isnan(a5) and not np.isnan(a30) and not np.isnan(adx5[i - aRise]):
            if a30 > aA30 and a5 > aA5 and a5 > adx5[i - aRise]:
                mA = (ci / c[i - aLb] - 1) * 100
                aL = mA >= aMom; aS = mA <= -aMom
        if i >= bRise and i >= bLb and not np.isnan(a5) and not np.isnan(a30) and not np.isnan(adx5[i - bRise]):
            if a30 > bA30 and a5 > adx5[i - bRise]:
                mB = (ci / c[i - bLb] - 1) * 100
                bL = mB >= bMom; bS = mB <= -bMom
        canV2 = vV2 == 0 or i - vV2Bar >= minHold
        if aL and vV2 <= 0 and canV2:
            vV2 = 1; vV2Bar = i
        elif aS and vV2 >= 0 and canV2:
            vV2 = -1; vV2Bar = i
        canV3 = vV3 == 0 or i - vV3Bar >= minHold
        if (aL or bL) and vV3 <= 0 and canV3:
            vV3 = 1; vV3Bar = i
        elif (aS or bS) and vV3 >= 0 and canV3:
            vV3 = -1; vV3Bar = i
        # ---- RSI extreme
        rv = rsi14[i]
        if not np.isnan(rv):
            if vR != 0:
                rRet = (ci / vREp - 1) * vR
                if rRet <= -rSL / 100 or i - vRBar >= rBars or (vR > 0 and rv > rExit) or (vR < 0 and rv < 100 - rExit):
                    vR = 0
            else:
                if rv < rLo:
                    vR = 1; vREp = ci; vRBar = i
                elif rv > 100 - rLo:
                    vR = -1; vREp = ci; vRBar = i
        # ---- mid regime
        if i >= mRise and i >= mLb and not np.isnan(a5) and not np.isnan(a30) and not np.isnan(adx5[i - mRise]):
            if a30 > mA30 and a5 > adx5[i - mRise]:
                mM = (ci / c[i - mLb] - 1) * 100
                canMid = vMid == 0 or i - vMidBar >= mHold
                if mM >= mMom and vMid <= 0 and canMid:
                    vMid = 1; vMidBar = i
                elif mM <= -mMom and vMid >= 0 and canMid:
                    vMid = -1; vMidBar = i
        midRun = midRun + 1 if vMid == vMidPrev else 1
        vMidPrev = vMid
        if midRun >= mConf:
            midC = vMid
        # ---- BTC 4H
        bf = 0
        if not np.isnan(badx4[i]) and not np.isnan(bdi4[i]) and badx4[i] > btcAdx:
            bf = 1 if bdi4[i] > 0 else (-1 if bdi4[i] < 0 else 0)
        if cm == 0:
            s = vV2 + vV3; col = 1 if s > 0 else (-1 if s < 0 else 0)
        elif cm == 1:
            col = vV2
        else:
            col = vV3
        # ---- target
        if uL == 1 and vLiq != 0: tg = vLiq
        elif uR == 1 and vR != 0: tg = vR
        elif uC == 1 and col != 0: tg = col
        elif uB == 1 and bf != 0: tg = bf
        elif uM == 1: tg = midC
        else: tg = 0
        tgtA[i] = tg
        # ---- state machine (shadow)
        posDir = shDir; coreEvt = tg != tgtPrev; tgtPrev = tg
        ahfThr = ahfK * (0.0 if np.isnan(atr1h[i]) else atr1h[i]) / 100
        r7t = 50.0 if np.isnan(rsi7[i]) else rsi7[i]
        if coreEvt:
            stMode = 0; want = tg
        elif stMode == 0:
            want = tg
            if posDir != 0 and posDir == tg:
                adv = (ci / shEp - 1) * (-posDir)
                if uK == 1 and ahfThr > 0 and adv >= ahfThr:
                    want = -posDir; stMode = 1; cntDir = posDir; cntPx = ci; cntBar = i
                elif uT == 1 and ((posDir > 0 and r7t >= trX) or (posDir < 0 and r7t <= 100 - trX)) and tsh[i] * posDir < shMax:
                    want = 0; stMode = 2; trimPx = ci; trimBar = i; trimSide = posDir
        elif stMode == 1:
            cG = (ci / cntPx - 1) * (-cntDir)
            if cG <= -ahfSL / 100 or i - cntBar >= ahfBars:
                stMode = 0; want = tg
            else:
                want = -cntDir
        else:
            if trimSide > 0:
                bi = r7t <= trY or i - trimBar >= trBars or ci >= trimPx * (1 + trCont / 100)
            else:
                bi = r7t >= 100 - trY or i - trimBar >= trBars or ci <= trimPx * (1 - trCont / 100)
            if bi:
                stMode = 0; want = tg
            else:
                want = 0
        if want != shDir:
            shDir = want; shEp = ci + tick * want if want != 0 else np.nan
        wantA[i] = want
        # ---- circuit breaker on real closed trades (closed before this bar)
        while cbSeen < nt:
            if pl[cbSeen] < 0:
                cbSt += 1; cbSum -= pl[cbSeen]
            else:
                cbSt = 0; cbSum = 0.0
            cbSeen += 1
        rw = want
        if uCb == 1:
            if cbPaused and t[i] > cbUntil:
                cbPaused = False; cbSt = 0; cbSum = 0.0; cbWait = True; cbLast = want
            if cbPaused:
                rw = 0
            elif cbSt >= cbLoss and cbSum >= cbAmt:
                rw = 0; cbPaused = True; cbUntil = t[i] + int(round(cbDays * 86400000))
            elif cbWait:
                if want != cbLast:
                    cbWait = False
                else:
                    rw = 0
        realA[i] = rw
        # ---- fills at close +- tick
        if rw != pos:
            if pos != 0:
                x = ci - tick * pos
                eb[nt] = pEb; xb[nt] = i; dr[nt] = pos; ep[nt] = pEp; xp[nt] = x; qt[nt] = pQ
                pl[nt] = (x - pEp) * pQ * pos - COMM * pQ * (pEp + x); nt += 1
            if rw != 0:
                pEp = ci + tick * rw; pQ = CASH / ci; pEb = i
            pos = rw
    return tgtA, wantA, realA, eb[:nt], xb[:nt], dr[:nt], ep[:nt], xp[:nt], qt[:nt], pl[:nt], pos, pEb, pEp, pQ

def oi_prev(D, lag=0):
    o = D['oi']; out = np.full(len(o), np.nan); k = 1 + lag
    out[k:] = o[:-k]; return out

def go(D, Pv, F, lag=0):
    r = run(D['t'], D['h'], D['l'], D['c'], D['adx5'], D['adx30'], D['badx4'], D['bdi4'], D['atr1h'],
            D['rsi14'], D['rsi7'], D['tsh'], oi_prev(D, lag), Pv, F, TICK)
    return dict(tgt=r[0], want=r[1], real=r[2], eb=r[3], xb=r[4], dir=r[5], ep=r[6], xp=r[7], q=r[8], pnl=r[9],
                open=(r[10], r[11], r[12], r[13]))

@nb.njit(cache=True)
def equity(n, eb, xb, dr, ep, q, pl, h, l, c, opos, oeb, oep, oq):
    cl = np.zeros(n); mtm = np.zeros(n); worst = np.zeros(n)
    k = 0; cum = 0.0; m = len(eb)
    for i in range(n):
        while k < m and xb[k] <= i:
            cum += pl[k]; k += 1
        cl[i] = cum; op = 0.0; wo = 0.0
        if k < m and eb[k] < i:  # open trade k (entered before i, exits after i)
            d = dr[k]; op = (c[i] - ep[k]) * q[k] * d
            wo = ((l[i] if d > 0 else h[i]) - ep[k]) * q[k] * d
        elif k >= m and opos != 0 and oeb < i:
            op = (c[i] - oep) * oq * opos; wo = ((l[i] if opos > 0 else h[i]) - oep) * oq * opos
        mtm[i] = cum + op; worst[i] = cum + wo
    return cl, mtm, worst
