# Anlık mikro patlamada gösterge seviyesinde milim eşik indirimi. RL/RS: long/short giriş gevşetme oranı, XL/XS: long/short pozisyondan çıkış gevşetme oranı.
import numpy as np
from numba import njit
from p1m import hmap, st_dir, TICK, TV0
@njit(cache=True)
def core2(c, d288, z48, z96, r14, r7, okO, dOi6, dPx6, dO288, vr, rr, br, ar, nrv, dmSig, adx15, adx30, adx1h, trDir, stDir, GL, GS, m5, tsB, RL, RS, XL, XS, gF, gW, tick):
    n = len(c); out = np.zeros(n, np.int64)
    W = np.array([3., 3., 4., 0., 0., 1., 1., 3., .5, 1., 4., .5, 2., .5, 1.]); den = W.sum()
    eV = np.zeros(5, np.int64); eE = np.zeros(5); eB = np.zeros(5, np.int64); oV = np.zeros(3, np.int64); oE = np.zeros(3); oB = np.zeros(3, np.int64)
    rV = np.zeros(2, np.int64); rH = np.full(2, np.nan); rL = np.full(2, np.nan)
    OLO = np.array([-4.0, -3.5, 20.0]); OHI = np.array([4.0, 3.5, 80.0]); OXL = np.array([0.5, 0.0, 45.0]); OXS = np.array([-0.5, 0.0, 55.0]); OMID = np.array([0.0, 0.0, 50.0])
    OSL = np.array([8.0, 8.0, 3.0]); OMB = np.array([288.0, 288.0, 144.0]); RP = np.array([3.0, 8.0])
    EH = np.array([648.0, 144.0, 288.0, 288.0, 288.0]); ESL = np.array([3.0, 10.0, 10.0, 10.0, 10.0])
    vLiq = 0; vLiqEp = 0.0; vLiqB = 0; vBL = 0; vBLb = 0; vDM = 0; pos = 0; realDir = 0; rEp = 0.0; lockDir = 0; lockExt = 0.0; prevT = 0; rEb = 0
    P = np.zeros(15)
    for i in range(n):
        x = c[i]; d = d288[i]; okd = not np.isnan(d); fl = RL[i] * gF; fs = RS[i] * gF; wl = RL[i] * gW; ws_ = RS[i] * gW; xl = XL[i]; xs = XS[i]
        # likidasyon (dönüş grubu)
        liqL = okO[i] and dPx6[i] <= -1.5 * (1 - fl) and dOi6[i] <= -2.0 * (1 - fl); liqS = okO[i] and dPx6[i] >= 1.5 * (1 - fs) and dOi6[i] <= -2.0 * (1 - fs)
        if vLiq != 0:
            xx = xl if vLiq > 0 else xs
            if (x / vLiqEp - 1.0) * vLiq <= -0.05 * (1 - xx) or i - vLiqB >= 144 * m5 or (vLiq > 0 and r7[i] > 50 + 15 * (1 - xl)) or (vLiq < 0 and r7[i] < 50 - 15 * (1 - xs)): vLiq = 0
            elif vLiq > 0 and liqS: vLiq = -1; vLiqEp = x; vLiqB = i
            elif vLiq < 0 and liqL: vLiq = 1; vLiqEp = x; vLiqB = i
        elif liqL: vLiq = 1; vLiqEp = x; vLiqB = i
        elif liqS: vLiq = -1; vLiqEp = x; vLiqB = i
        # BTC lider (takip grubu)
        if vBL != 0 and i - vBLb >= 24 * m5: vBL = 0
        if not np.isnan(br[i]) and not np.isnan(ar[i]):
            if br[i] >= 2.5 * (1 - wl) and ar[i] < br[i]: vBL = 1; vBLb = i
            elif br[i] <= -2.5 * (1 - ws_) and ar[i] > br[i]: vBL = -1; vBLb = i
        if dmSig[i] != 0: vDM = dmSig[i]
        sV1 = 0
        if okd:
            if d <= -3.5 * (1 - fl): sV1 = 1
            elif d >= 6.0 * (1 - fs): sV1 = -1
        sOIK = 0; sHVX = 0; s15 = 0; sREL = 0
        if okd and d < 0:
            if not np.isnan(dO288[i]) and -d >= 4.0 * (1 - fl) and dO288[i] >= 6.0 * (1 - fl): sOIK = 1
            if not np.isnan(vr[i]) and -d >= 5.0 * (1 - fl) and vr[i] >= 1.5 * (1 - fl): sHVX = 1
            if nrv[i] >= 1.5 * (1 - fl) and -d >= 4.0 * (1 - fl): s15 = 1
        elif okd and d > 0:
            if not np.isnan(dO288[i]) and d >= 4.0 * (1 - fs) and dO288[i] >= 6.0 * (1 - fs): sOIK = -1
            if not np.isnan(vr[i]) and d >= 5.0 * (1 - fs) and vr[i] >= 1.5 * (1 - fs): sHVX = -1
            if nrv[i] >= 1.5 * (1 - fs) and d >= 4.0 * (1 - fs): s15 = -1
        if not np.isnan(rr[i]):
            if rr[i] <= -6.0 * (1 - fl): sREL = 1
            elif rr[i] >= 6.0 * (1 - fs): sREL = -1
        # Ent bileşenleri (stop çıkışı gevşer)
        SG = np.array([sV1, sOIK, sHVX, sREL, s15]); PI = np.array([0, 7, 8, 9, 14])
        for k in range(5):
            if eV[k] != 0:
                xx = xl if eV[k] > 0 else xs
                if (x / eE[k] - 1.0) * eV[k] <= -ESL[k] * (1 - xx) / 100.0 or i - eB[k] >= EH[k] * m5: eV[k] = 0
            s = SG[k]
            if s != 0:
                if eV[k] != s: eV[k] = s; eE[k] = x; eB[k] = i
                else: eB[k] = i
            P[PI[k]] = eV[k]
        # Osc bileşenleri (z48, z96, RSI14)
        XO = np.array([z48[i], z96[i], r14[i]]); OP = np.array([1, 2, 5])
        for k in range(3):
            xv = XO[k]
            if not np.isnan(xv):
                mid = OMID[k]; span = abs(OLO[k] - mid)
                if oV[k] != 0:
                    xx = xl if oV[k] > 0 else xs
                    if (x / oE[k] - 1.0) * oV[k] <= -OSL[k] * (1 - xx) / 100.0 or i - oB[k] >= OMB[k] * m5 or (oV[k] > 0 and xv > OXL[k] - span * xl) or (oV[k] < 0 and xv < OXS[k] + span * xs): oV[k] = 0
                elif xv < mid + (OLO[k] - mid) * (1 - fl): oV[k] = 1; oE[k] = x; oB[k] = i
                elif xv > mid + (OHI[k] - mid) * (1 - fs): oV[k] = -1; oE[k] = x; oB[k] = i
            P[OP[k]] = oV[k]
        P[6] = vLiq; P[10] = vBL; P[13] = vDM
        # %3 / %8 dönüş takibi (takip grubu): kısa yöne dönüş = short girişi + long çıkışı
        for k in range(2):
            if np.isnan(rH[k]): rH[k] = x; rL[k] = x
            if rV[k] >= 0:
                rH[k] = max(rH[k], x); g = max(ws_, xl if rV[k] > 0 else 0.0)
                if x <= rH[k] * (1.0 - RP[k] * (1 - g) / 100.0): rV[k] = -1; rL[k] = x
            if rV[k] <= 0:
                rL[k] = min(rL[k], x); g = max(wl, xs if rV[k] < 0 else 0.0)
                if x >= rL[k] * (1.0 + RP[k] * (1 - g) / 100.0): rV[k] = 1; rH[k] = x
            P[11 + k] = rV[k]
        s = (W * P).sum() / den
        onL = 0.15 if (trDir[i] == -1 and adx1h[i] >= 20.0) else 0.04; onS = 0.15 if (trDir[i] == 1 and adx1h[i] >= 20.0) else 0.04
        sH = stDir[i] if adx30[i] >= 25.0 else 0; sT = stDir[i] if adx30[i] >= 30.0 else 0
        offNow = -0.15 if (pos != 0 and sH == pos) else 0.0
        if sT == -1: onL = max(onL, 0.12)
        if sT == 1: onS = max(onS, 0.12)
        want = pos
        if pos != 0 and s * pos < offNow: want = 0
        if s >= onL: want = 1
        elif s <= -onS: want = -1
        pos = want; tgt = pos
        if tgt != prevT: lockDir = 0; prevT = tgt
        wD = tgt
        if realDir != 0 and wD == realDir:
            pl = (x / rEp - 1.0) * realDir * 100.0; xx = xl if realDir > 0 else xs
            if pl <= -7.0 * (1 - xx): lockDir = realDir; lockExt = x; wD = 0
            elif i - rEb >= tsB and pl <= -1.0 and adx15[i] < 25.0: lockDir = realDir; lockExt = x; wD = 0
        if lockDir != 0 and wD == lockDir:
            lockExt = min(lockExt, x) if lockDir > 0 else max(lockExt, x)
            if (x / lockExt - 1.0) * lockDir * 100.0 >= 4.0: lockDir = 0
            else: wD = 0
        if wD != 0 and wD != realDir and (nrv[i] < 0.1 or (wD == 1 and GL[i]) or (wD == -1 and GS[i])): wD = 0
        if wD != realDir: realDir = wD; rEp = x + tick * wD; rEb = i
        out[i] = realDir
    return out
_ST = {}
def run2(Dd, RL=None, RS=None, XL=None, XS=None, gF=1.0, gW=1.0, whale='tam'):
    n = len(Dd['c']); Z = np.zeros(n)
    key = id(Dd)
    if key not in _ST:
        H90 = Dd['H90']; _ST[key] = np.nan_to_num(hmap(st_dir(H90[2], H90[3], H90[4], 10, 2.5).astype(float), H90[0], Dd['t'], 90)).astype(np.int64)
    ws = Dd['ws'].copy()
    if whale == 'tv': ws[Dd['t'] < TV0] = 0
    GL = (ws == 1) | (ws == 3); GS = (ws == 2) | (ws == 3)
    return core2(Dd['c'], Dd['d288'], Dd['z48'], Dd['z96'], Dd['r14'], Dd['r7'], Dd['okO'], Dd['dOi6'], Dd['dPx6'], Dd['dO288'], Dd['vr'], Dd['rr'], Dd['br'], Dd['ar'], Dd['nrv'],
                 Dd['dmSig'], Dd['adx15'], Dd['adx30'], Dd['adx1h'], Dd['trDir'], _ST[key], GL, GS, float(Dd['m5']), float(round(288 * Dd['m5'])),
                 Z if RL is None else RL, Z if RS is None else RS, Z if XL is None else XL, Z if XS is None else XS, gF, gW, TICK)
