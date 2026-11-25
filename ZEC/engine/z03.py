"""1 dk: R2 + SIÇRAMA BEKÇİSİ + SIÇRAMA LONG motoru."""
import numpy as np
from numba import njit
from z01 import FEE, TICK, CAP


@njit(cache=True)
def sim7(o, h, l, c, atr, aL, aS, ats, atH, bL, bts, btrl, btH, spk, blkN):
    """A = R2 (öncelikli; SL ats·ATR, süre atH). B = sıçrama long (yalnız boşken; ilk SL bts·ATR, iz tepe − btrl·ATR, süre btH).
    spk[i] True → açık SHORT bar kapanışında ANINDA kapanır ve blkN bar yeni short açılmaz (sıçrama bekçisi)."""
    n = len(c); eq = np.zeros(n); real = 0.0; pos = 0; own = 0; ep = 0.0; q = 0.0; stop = 0.0; ei = 0; ti = 0; mae = 0.0; pk = 0.0; noS = -1
    T = np.zeros((n // 2 + 10, 9)); nt = 0
    for i in range(1, n):
        if pos != 0:
            if (pos == 1 and l[i] <= stop) or (pos == -1 and h[i] >= stop):
                xp = (min(o[i], stop) - TICK) if pos == 1 else (max(o[i], stop) + TICK)
                pnl = q * (xp - ep) * pos - FEE * q * (ep + xp); real += pnl
                mae = max(mae, (ep - l[i]) / ep if pos == 1 else (h[i] - ep) / ep)
                T[nt, 0] = ei; T[nt, 1] = i; T[nt, 2] = pos; T[nt, 3] = ep; T[nt, 4] = xp; T[nt, 5] = pnl; T[nt, 6] = mae; T[nt, 7] = own; T[nt, 8] = 0; nt += 1
                pos = 0; own = 0
            else:
                mae = max(mae, (ep - l[i]) / ep if pos == 1 else (h[i] - ep) / ep)
        if spk[i]:
            noS = i + blkN
        wA = 1 if aL[i] else (-1 if (aS[i] and i > noS) else 0); wB = 1 if bL[i] else 0
        guard = pos == -1 and spk[i]
        tout = pos != 0 and ((own == 1 and i - ti >= atH) or (own == 2 and i - ti >= btH))
        ex = pos != 0 and (guard or (wA != 0 and wA != pos) or tout)
        blk = pos if (tout and (wA == pos or wB == pos)) else 0
        if ex:
            xp = c[i] - TICK * pos; pnl = q * (xp - ep) * pos - FEE * q * (ep + xp); real += pnl
            T[nt, 0] = ei; T[nt, 1] = i; T[nt, 2] = pos; T[nt, 3] = ep; T[nt, 4] = xp; T[nt, 5] = pnl; T[nt, 6] = mae; T[nt, 7] = own; T[nt, 8] = 1 if guard else 0; nt += 1
            pos = 0; own = 0
        if pos != 0 and own == 2 and wA == pos:
            own = 1; ti = i; s2 = c[i] - ats * atr[i]; stop = max(stop, s2)
        if pos == 0 and not np.isnan(atr[i]):
            if wA != 0 and wA != blk:
                pos = wA; own = 1; ep = c[i] + TICK * pos; q = CAP / ep; ei = i; ti = i; mae = 0.0; stop = ep - pos * ats * atr[i]
            elif wB != 0 and wB != blk:
                pos = 1; own = 2; ep = c[i] + TICK; q = CAP / ep; ei = i; ti = i; mae = 0.0; pk = c[i]; stop = ep - bts * atr[i]
        if pos == 1 and own == 2:
            pk = max(pk, c[i]); stop = max(stop, pk - btrl * atr[i])
        eq[i] = real + (q * (c[i] - ep) * pos - FEE * q * ep if pos != 0 else 0.0)
    return eq, T[:nt]


@njit(cache=True)
def sim8(o, h, l, c, atr, aL, aS, ats, atH, spk, blkN, upX, dnX, lockK, keep):
    """R3 + sıçrama bekçisi + SIÇRAMA KÂR KİLİDİ: açık long'da upX (lehe sıçrama) olursa kilit açılır; kilitliyken
    stop = max(stop, tepe − lockK·ATR, giriş + keep·(tepe − giriş)). Short için ayna (dnX). Kayıt: [.., mfe, kilit]."""
    n = len(c); eq = np.zeros(n); real = 0.0; pos = 0; ep = 0.0; q = 0.0; stop = 0.0; ei = 0; mae = 0.0; mfe = 0.0; pk = 0.0; noS = -1; lk = False
    T = np.zeros((n // 2 + 10, 10)); nt = 0
    for i in range(1, n):
        if pos != 0:
            mae = max(mae, (ep - l[i]) / ep if pos == 1 else (h[i] - ep) / ep); mfe = max(mfe, (h[i] - ep) / ep if pos == 1 else (ep - l[i]) / ep)
            if (pos == 1 and l[i] <= stop) or (pos == -1 and h[i] >= stop):
                xp = (min(o[i], stop) - TICK) if pos == 1 else (max(o[i], stop) + TICK)
                pnl = q * (xp - ep) * pos - FEE * q * (ep + xp); real += pnl
                T[nt, 0] = ei; T[nt, 1] = i; T[nt, 2] = pos; T[nt, 3] = ep; T[nt, 4] = xp; T[nt, 5] = pnl; T[nt, 6] = mae; T[nt, 7] = 0; T[nt, 8] = mfe; T[nt, 9] = 1 if lk else 0; nt += 1
                pos = 0
        if spk[i]:
            noS = i + blkN
        wA = 1 if aL[i] else (-1 if (aS[i] and i > noS) else 0)
        guard = pos == -1 and spk[i]
        tout = pos != 0 and i - ei >= atH
        ex = pos != 0 and (guard or (wA != 0 and wA != pos) or tout)
        blk = pos if (tout and wA == pos) else 0
        if ex:
            xp = c[i] - TICK * pos; pnl = q * (xp - ep) * pos - FEE * q * (ep + xp); real += pnl
            T[nt, 0] = ei; T[nt, 1] = i; T[nt, 2] = pos; T[nt, 3] = ep; T[nt, 4] = xp; T[nt, 5] = pnl; T[nt, 6] = mae; T[nt, 7] = 1 if guard else 0; T[nt, 8] = mfe; T[nt, 9] = 1 if lk else 0; nt += 1
            pos = 0
        if pos == 0 and wA != 0 and wA != blk and not np.isnan(atr[i]):
            pos = wA; ep = c[i] + TICK * pos; q = CAP / ep; ei = i; mae = 0.0; mfe = 0.0; stop = ep - pos * ats * atr[i]; pk = c[i]; lk = False
        if pos != 0:
            pk = max(pk, c[i]) if pos == 1 else min(pk, c[i])
            if (pos == 1 and upX[i]) or (pos == -1 and dnX[i]):
                lk = True
            if lk and (pk - ep) * pos > 0:
                if pos == 1:
                    stop = max(stop, max(pk - lockK * atr[i], ep + keep * (pk - ep)))
                else:
                    stop = min(stop, min(pk + lockK * atr[i], ep - keep * (ep - pk)))
        eq[i] = real + (q * (c[i] - ep) * pos - FEE * q * ep if pos != 0 else 0.0)
    return eq, T[:nt]


@njit(cache=True)
def sim10(o, h, l, c, atr, aL, aS, ats, atH, spk, blkN, RG, D4, mode):
    """R5 + REJİMDE TAŞI: long iken RG==+1 (short iken RG==−1) olduğu sürece süre çıkışı yok; zarar kes → iz (en iyi kapanış − D4).
    Rejim bitince normal kurala döner. mode: 0 kapalı, 1 yalnız long, 2 iki yön."""
    n = len(c); eq = np.zeros(n); real = 0.0; pos = 0; ep = 0.0; q = 0.0; stop = 0.0; ei = 0; ti = 0; mae = 0.0; noS = -1; best = 0.0
    T = np.zeros((n // 2 + 10, 8)); nt = 0
    for i in range(1, n):
        if pos != 0:
            mae = max(mae, (ep - l[i]) / ep if pos == 1 else (h[i] - ep) / ep)
            if (pos == 1 and l[i] <= stop) or (pos == -1 and h[i] >= stop):
                xp = (min(o[i], stop) - TICK) if pos == 1 else (max(o[i], stop) + TICK)
                pnl = q * (xp - ep) * pos - FEE * q * (ep + xp); real += pnl
                T[nt, 0] = ei; T[nt, 1] = i; T[nt, 2] = pos; T[nt, 3] = ep; T[nt, 4] = xp; T[nt, 5] = pnl; T[nt, 6] = mae; T[nt, 7] = 0; nt += 1
                pos = 0
        if spk[i]:
            noS = i + blkN
        wA = 1 if aL[i] else (-1 if (aS[i] and i > noS) else 0)
        hold = pos != 0 and mode > 0 and ((pos == 1 and RG[i] == 1) or (pos == -1 and RG[i] == -1 and mode == 2)) and not np.isnan(D4[i])
        guard = pos == -1 and spk[i]
        tout = pos != 0 and (not hold) and i - ti >= atH
        ex = pos != 0 and (guard or (wA != 0 and wA != pos and not hold) or tout)
        blk = pos if (tout and wA == pos) else 0
        if ex:
            xp = c[i] - TICK * pos; pnl = q * (xp - ep) * pos - FEE * q * (ep + xp); real += pnl
            T[nt, 0] = ei; T[nt, 1] = i; T[nt, 2] = pos; T[nt, 3] = ep; T[nt, 4] = xp; T[nt, 5] = pnl; T[nt, 6] = mae; T[nt, 7] = 1 if guard else 0; nt += 1
            pos = 0
        if pos != 0 and wA == pos:
            ti = i                                            # aynı yönde yeni oy → süre yenilenir
        if pos == 0 and wA != 0 and wA != blk and not np.isnan(atr[i]):
            pos = wA; ep = c[i] + TICK * pos; q = CAP / ep; ei = i; ti = i; mae = 0.0; stop = ep - pos * ats * atr[i]; best = c[i]
        if pos != 0:
            best = max(best, c[i]) if pos == 1 else min(best, c[i])
            if hold:
                stop = max(stop, best - D4[i]) if pos == 1 else min(stop, best + D4[i])
        eq[i] = real + (q * (c[i] - ep) * pos - FEE * q * ep if pos != 0 else 0.0)
    return eq, T[:nt]
