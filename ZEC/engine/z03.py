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


@njit(cache=True)
def sim11(o, h, l, c, atr, aL, aS, eL, eS, ats, atH, spk, blkN):
    """R motoru (sim10 mod 0 ile aynı yön/çıkış mantığı) + ayrı GİRİŞ İZNİ: oy yönü aL/aS, giriş ancak eL/eS izinliyse.
    Oy sürdükçe izin gelene kadar bekler (fitil/dikkat kapısı); çıkışlar izinden bağımsız."""
    n = len(c); eq = np.zeros(n); real = 0.0; pos = 0; ep = 0.0; q = 0.0; stop = 0.0; ei = 0; ti = 0; mae = 0.0; noS = -1
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
        guard = pos == -1 and spk[i]
        tout = pos != 0 and i - ti >= atH
        ex = pos != 0 and (guard or (wA != 0 and wA != pos) or tout)
        blk = pos if (tout and wA == pos) else 0
        if ex:
            xp = c[i] - TICK * pos; pnl = q * (xp - ep) * pos - FEE * q * (ep + xp); real += pnl
            T[nt, 0] = ei; T[nt, 1] = i; T[nt, 2] = pos; T[nt, 3] = ep; T[nt, 4] = xp; T[nt, 5] = pnl; T[nt, 6] = mae; T[nt, 7] = 1 if guard else 0; nt += 1
            pos = 0
        if pos != 0 and wA == pos:
            ti = i
        if pos == 0 and wA != 0 and wA != blk and not np.isnan(atr[i]) and ((wA == 1 and eL[i]) or (wA == -1 and eS[i])):
            pos = wA; ep = c[i] + TICK * pos; q = CAP / ep; ei = i; ti = i; mae = 0.0; stop = ep - pos * ats * atr[i]
        eq[i] = real + (q * (c[i] - ep) * pos - FEE * q * ep if pos != 0 else 0.0)
    return eq, T[:nt]


@njit(cache=True)
def sim12(o, h, l, c, atr, aL1, aS1, eL1, eS1, ts1, tH1, aL2, aS2, eL2, eS2, ts2, tH2, spk, blkN):
    """İki motor, tek 500 $ birim: ANA motor (1) öncelikli; ana boştayken YEDEK motor (2) işlem açar.
    Yedek pozisyondayken ana aynı yönü isterse devralır (pozisyon sürer, stop ana genişliğine), ters isterse kapatıp ana girer.
    T[:,7] = sahibi (1/2)."""
    n = len(c); eq = np.zeros(n); real = 0.0; pos = 0; own = 0; ep = 0.0; q = 0.0; stop = 0.0; ei = 0; ti = 0; mae = 0.0; noS = -1; ae = 0.0
    T = np.zeros((n // 2 + 10, 8)); nt = 0
    for i in range(1, n):
        if pos != 0:
            mae = max(mae, (ep - l[i]) / ep if pos == 1 else (h[i] - ep) / ep)
            if (pos == 1 and l[i] <= stop) or (pos == -1 and h[i] >= stop):
                xp = (min(o[i], stop) - TICK) if pos == 1 else (max(o[i], stop) + TICK)
                pnl = q * (xp - ep) * pos - FEE * q * (ep + xp); real += pnl
                T[nt, 0] = ei; T[nt, 1] = i; T[nt, 2] = pos; T[nt, 3] = ep; T[nt, 4] = xp; T[nt, 5] = pnl; T[nt, 6] = mae; T[nt, 7] = own; nt += 1
                pos = 0; own = 0
        if spk[i]:
            noS = i + blkN
        w1 = 1 if aL1[i] else (-1 if (aS1[i] and i > noS) else 0)
        w2 = 1 if aL2[i] else (-1 if (aS2[i] and i > noS) else 0)
        guard = pos == -1 and spk[i]
        ex = False; blk = 0
        if pos != 0 and own == 2 and w1 == pos and not guard:
            own = 1; ti = i; stop = ep - pos * ts1 * ae
        if pos != 0 and own == 1:
            tout = i - ti >= tH1
            ex = guard or (w1 != 0 and w1 != pos) or tout
            blk = pos if (tout and w1 == pos) else 0
        elif pos != 0 and own == 2:
            tout = i - ti >= tH2
            ex = guard or (w1 != 0 and w1 != pos) or (w2 != 0 and w2 != pos) or tout
            blk = pos if (tout and w2 == pos) else 0
        if ex:
            xp = c[i] - TICK * pos; pnl = q * (xp - ep) * pos - FEE * q * (ep + xp); real += pnl
            T[nt, 0] = ei; T[nt, 1] = i; T[nt, 2] = pos; T[nt, 3] = ep; T[nt, 4] = xp; T[nt, 5] = pnl; T[nt, 6] = mae; T[nt, 7] = own; nt += 1
            pos = 0; own = 0
        if pos != 0 and ((own == 1 and w1 == pos) or (own == 2 and w2 == pos)):
            ti = i
        if pos == 0 and not np.isnan(atr[i]):
            if w1 != 0 and w1 != blk and ((w1 == 1 and eL1[i]) or (w1 == -1 and eS1[i])):
                pos = w1; own = 1; ep = c[i] + TICK * pos; q = CAP / ep; ei = i; ti = i; mae = 0.0; ae = atr[i]; stop = ep - pos * ts1 * ae
            elif w2 != 0 and w2 != blk and ((w2 == 1 and eL2[i]) or (w2 == -1 and eS2[i])):
                pos = w2; own = 2; ep = c[i] + TICK * pos; q = CAP / ep; ei = i; ti = i; mae = 0.0; ae = atr[i]; stop = ep - pos * ts2 * ae
        eq[i] = real + (q * (c[i] - ep) * pos - FEE * q * ep if pos != 0 else 0.0)
    return eq, T[:nt]


@njit(cache=True)
def simN(o, h, l, c, atr, AL, AS, EL, ES, TS, TH, spk, blkN):
    """N motor öncelik sırasıyla (0 en yüksek), tek 500 $ birim. Pozisyon sahibi k iken daha öncelikli j aynı yönü isterse devralır,
    ters isterse kapanır ve j girer. Boşta: en öncelikli izinli motor girer. T[:,7] = sahip motor."""
    K, n = AL.shape; eq = np.zeros(n); real = 0.0; pos = 0; own = -1; ep = 0.0; q = 0.0; stop = 0.0; ei = 0; ti = 0; mae = 0.0; noS = -1; ae = 0.0
    T = np.zeros((n // 2 + 10, 8)); nt = 0; w = np.zeros(K, np.int64)
    for i in range(1, n):
        if pos != 0:
            mae = max(mae, (ep - l[i]) / ep if pos == 1 else (h[i] - ep) / ep)
            if (pos == 1 and l[i] <= stop) or (pos == -1 and h[i] >= stop):
                xp = (min(o[i], stop) - TICK) if pos == 1 else (max(o[i], stop) + TICK)
                pnl = q * (xp - ep) * pos - FEE * q * (ep + xp); real += pnl
                T[nt, 0] = ei; T[nt, 1] = i; T[nt, 2] = pos; T[nt, 3] = ep; T[nt, 4] = xp; T[nt, 5] = pnl; T[nt, 6] = mae; T[nt, 7] = own; nt += 1
                pos = 0; own = -1
        if spk[i]:
            noS = i + blkN
        for k in range(K):
            w[k] = 1 if AL[k, i] else (-1 if (AS[k, i] and i > noS) else 0)
        guard = pos == -1 and spk[i]
        if pos != 0 and not guard:
            for j in range(own):
                if w[j] == pos:
                    own = j; ti = i; stop = ep - pos * TS[j] * ae
                    break
        ex = False; blk = 0
        if pos != 0:
            tout = i - ti >= TH[own]
            opp = False
            for j in range(own + 1):
                if w[j] != 0 and w[j] != pos:
                    opp = True
            ex = guard or opp or tout
            blk = pos if (tout and w[own] == pos) else 0
        if ex:
            xp = c[i] - TICK * pos; pnl = q * (xp - ep) * pos - FEE * q * (ep + xp); real += pnl
            T[nt, 0] = ei; T[nt, 1] = i; T[nt, 2] = pos; T[nt, 3] = ep; T[nt, 4] = xp; T[nt, 5] = pnl; T[nt, 6] = mae; T[nt, 7] = own; nt += 1
            pos = 0; own = -1
        if pos != 0 and w[own] == pos:
            ti = i
        if pos == 0 and not np.isnan(atr[i]):
            for k in range(K):
                if w[k] != 0 and w[k] != blk and ((w[k] == 1 and EL[k, i]) or (w[k] == -1 and ES[k, i])):
                    pos = w[k]; own = k; ep = c[i] + TICK * pos; q = CAP / ep; ei = i; ti = i; mae = 0.0; ae = atr[i]; stop = ep - pos * TS[k] * ae
                    break
        eq[i] = real + (q * (c[i] - ep) * pos - FEE * q * ep if pos != 0 else 0.0)
    return eq, T[:nt]


@njit(cache=True)
def simN2(o, h, l, c, atr, AL, AS, EL, ES, TS, TH, spk, blkN):
    """[simN2] simN + giriş kuralı: daha öncelikli motor ters yön istiyorsa alt motor GİREMEZ (giriş-çıkış zinciri biter).
    N motor öncelik sırasıyla (0 en yüksek), tek 500 $ birim. Pozisyon sahibi k iken daha öncelikli j aynı yönü isterse devralır,
    ters isterse kapanır ve j girer. Boşta: en öncelikli izinli motor girer. T[:,7] = sahip motor."""
    K, n = AL.shape; eq = np.zeros(n); real = 0.0; pos = 0; own = -1; ep = 0.0; q = 0.0; stop = 0.0; ei = 0; ti = 0; mae = 0.0; noS = -1; ae = 0.0
    T = np.zeros((n // 2 + 10, 8)); nt = 0; w = np.zeros(K, np.int64)
    for i in range(1, n):
        if pos != 0:
            mae = max(mae, (ep - l[i]) / ep if pos == 1 else (h[i] - ep) / ep)
            if (pos == 1 and l[i] <= stop) or (pos == -1 and h[i] >= stop):
                xp = (min(o[i], stop) - TICK) if pos == 1 else (max(o[i], stop) + TICK)
                pnl = q * (xp - ep) * pos - FEE * q * (ep + xp); real += pnl
                T[nt, 0] = ei; T[nt, 1] = i; T[nt, 2] = pos; T[nt, 3] = ep; T[nt, 4] = xp; T[nt, 5] = pnl; T[nt, 6] = mae; T[nt, 7] = own; nt += 1
                pos = 0; own = -1
        if spk[i]:
            noS = i + blkN
        for k in range(K):
            w[k] = 1 if AL[k, i] else (-1 if (AS[k, i] and i > noS) else 0)
        guard = pos == -1 and spk[i]
        if pos != 0 and not guard:
            for j in range(own):
                if w[j] == pos:
                    own = j; ti = i; stop = ep - pos * TS[j] * ae
                    break
        ex = False; blk = 0
        if pos != 0:
            tout = i - ti >= TH[own]
            opp = False
            for j in range(own + 1):
                if w[j] != 0 and w[j] != pos:
                    opp = True
            ex = guard or opp or tout
            blk = pos if (tout and w[own] == pos) else 0
        if ex:
            xp = c[i] - TICK * pos; pnl = q * (xp - ep) * pos - FEE * q * (ep + xp); real += pnl
            T[nt, 0] = ei; T[nt, 1] = i; T[nt, 2] = pos; T[nt, 3] = ep; T[nt, 4] = xp; T[nt, 5] = pnl; T[nt, 6] = mae; T[nt, 7] = own; nt += 1
            pos = 0; own = -1
        if pos != 0 and w[own] == pos:
            ti = i
        if pos == 0 and not np.isnan(atr[i]):
            for k in range(K):
                ok = True
                for jj in range(k):
                    if w[jj] != 0 and w[jj] != w[k]:
                        ok = False
                if ok and w[k] != 0 and w[k] != blk and ((w[k] == 1 and EL[k, i]) or (w[k] == -1 and ES[k, i])):
                    pos = w[k]; own = k; ep = c[i] + TICK * pos; q = CAP / ep; ei = i; ti = i; mae = 0.0; ae = atr[i]; stop = ep - pos * TS[k] * ae
                    break
        eq[i] = real + (q * (c[i] - ep) * pos - FEE * q * ep if pos != 0 else 0.0)
    return eq, T[:nt]


@njit(cache=True)
def simN3(o, h, l, c, atr, AL, AS, EL, ES, TS, TH, spk, blkN):
    """[simN3] simN2 + süre dolunca aynı yönü isteyen herhangi bir motor pozisyonu DEVRALIR (kapat-aynı yöne yeniden aç yok).
    [simN2] simN + giriş kuralı: daha öncelikli motor ters yön istiyorsa alt motor GİREMEZ (giriş-çıkış zinciri biter).
    N motor öncelik sırasıyla (0 en yüksek), tek 500 $ birim. Pozisyon sahibi k iken daha öncelikli j aynı yönü isterse devralır,
    ters isterse kapanır ve j girer. Boşta: en öncelikli izinli motor girer. T[:,7] = sahip motor."""
    K, n = AL.shape; eq = np.zeros(n); real = 0.0; pos = 0; own = -1; ep = 0.0; q = 0.0; stop = 0.0; ei = 0; ti = 0; mae = 0.0; noS = -1; ae = 0.0
    T = np.zeros((n // 2 + 10, 8)); nt = 0; w = np.zeros(K, np.int64)
    for i in range(1, n):
        if pos != 0:
            mae = max(mae, (ep - l[i]) / ep if pos == 1 else (h[i] - ep) / ep)
            if (pos == 1 and l[i] <= stop) or (pos == -1 and h[i] >= stop):
                xp = (min(o[i], stop) - TICK) if pos == 1 else (max(o[i], stop) + TICK)
                pnl = q * (xp - ep) * pos - FEE * q * (ep + xp); real += pnl
                T[nt, 0] = ei; T[nt, 1] = i; T[nt, 2] = pos; T[nt, 3] = ep; T[nt, 4] = xp; T[nt, 5] = pnl; T[nt, 6] = mae; T[nt, 7] = own; nt += 1
                pos = 0; own = -1
        if spk[i]:
            noS = i + blkN
        for k in range(K):
            w[k] = 1 if AL[k, i] else (-1 if (AS[k, i] and i > noS) else 0)
        guard = pos == -1 and spk[i]
        if pos != 0 and not guard:
            for j in range(own):
                if w[j] == pos:
                    own = j; ti = i; stop = ep - pos * TS[j] * ae
                    break
        ex = False; blk = 0
        if pos != 0 and not guard and i - ti >= TH[own]:
            for kk in range(K):
                if w[kk] == pos:
                    own = kk; ti = i; stop = ep - pos * TS[kk] * ae
                    break
        if pos != 0:
            tout = i - ti >= TH[own]
            opp = False
            for j in range(own + 1):
                if w[j] != 0 and w[j] != pos:
                    opp = True
            ex = guard or opp or tout
            blk = pos if (tout and w[own] == pos) else 0
        if ex:
            xp = c[i] - TICK * pos; pnl = q * (xp - ep) * pos - FEE * q * (ep + xp); real += pnl
            T[nt, 0] = ei; T[nt, 1] = i; T[nt, 2] = pos; T[nt, 3] = ep; T[nt, 4] = xp; T[nt, 5] = pnl; T[nt, 6] = mae; T[nt, 7] = own; nt += 1
            pos = 0; own = -1
        if pos != 0 and w[own] == pos:
            ti = i
        if pos == 0 and not np.isnan(atr[i]):
            for k in range(K):
                ok = True
                for jj in range(k):
                    if w[jj] != 0 and w[jj] != w[k]:
                        ok = False
                if ok and w[k] != 0 and w[k] != blk and ((w[k] == 1 and EL[k, i]) or (w[k] == -1 and ES[k, i])):
                    pos = w[k]; own = k; ep = c[i] + TICK * pos; q = CAP / ep; ei = i; ti = i; mae = 0.0; ae = atr[i]; stop = ep - pos * TS[k] * ae
                    break
        eq[i] = real + (q * (c[i] - ep) * pos - FEE * q * ep if pos != 0 else 0.0)
    return eq, T[:nt]


@njit(cache=True)
def simN4(o, h, l, c, atr, AL, AS, EL, ES, TS, TH, spk, blkN):
    """[simN4] simN3 + stop sonrası kilit: stop yiyen yöne, o yöndeki oylar sönene kadar yeniden girilmez.
    [simN3] simN2 + süre dolunca aynı yönü isteyen herhangi bir motor pozisyonu DEVRALIR (kapat-aynı yöne yeniden aç yok).
    [simN2] simN + giriş kuralı: daha öncelikli motor ters yön istiyorsa alt motor GİREMEZ (giriş-çıkış zinciri biter).
    N motor öncelik sırasıyla (0 en yüksek), tek 500 $ birim. Pozisyon sahibi k iken daha öncelikli j aynı yönü isterse devralır,
    ters isterse kapanır ve j girer. Boşta: en öncelikli izinli motor girer. T[:,7] = sahip motor."""
    K, n = AL.shape; eq = np.zeros(n); real = 0.0; pos = 0; own = -1; ep = 0.0; q = 0.0; stop = 0.0; ei = 0; ti = 0; mae = 0.0; noS = -1; ae = 0.0
    T = np.zeros((n // 2 + 10, 8)); nt = 0; w = np.zeros(K, np.int64); lockD = 0
    for i in range(1, n):
        if pos != 0:
            mae = max(mae, (ep - l[i]) / ep if pos == 1 else (h[i] - ep) / ep)
            if (pos == 1 and l[i] <= stop) or (pos == -1 and h[i] >= stop):
                xp = (min(o[i], stop) - TICK) if pos == 1 else (max(o[i], stop) + TICK)
                pnl = q * (xp - ep) * pos - FEE * q * (ep + xp); real += pnl
                T[nt, 0] = ei; T[nt, 1] = i; T[nt, 2] = pos; T[nt, 3] = ep; T[nt, 4] = xp; T[nt, 5] = pnl; T[nt, 6] = mae; T[nt, 7] = own; nt += 1
                lockD = pos; pos = 0; own = -1
        if spk[i]:
            noS = i + blkN
        for k in range(K):
            w[k] = 1 if AL[k, i] else (-1 if (AS[k, i] and i > noS) else 0)
        if lockD != 0:
            still = False
            for kk in range(K):
                if w[kk] == lockD:
                    still = True
            if not still:
                lockD = 0
        guard = pos == -1 and spk[i]
        if pos != 0 and not guard:
            for j in range(own):
                if w[j] == pos:
                    own = j; ti = i; stop = ep - pos * TS[j] * ae
                    break
        ex = False; blk = 0
        if pos != 0 and not guard and i - ti >= TH[own]:
            for kk in range(K):
                if w[kk] == pos:
                    own = kk; ti = i; stop = ep - pos * TS[kk] * ae
                    break
        if pos != 0:
            tout = i - ti >= TH[own]
            opp = False
            for j in range(own + 1):
                if w[j] != 0 and w[j] != pos:
                    opp = True
            ex = guard or opp or tout
            blk = pos if (tout and w[own] == pos) else 0
        if ex:
            xp = c[i] - TICK * pos; pnl = q * (xp - ep) * pos - FEE * q * (ep + xp); real += pnl
            T[nt, 0] = ei; T[nt, 1] = i; T[nt, 2] = pos; T[nt, 3] = ep; T[nt, 4] = xp; T[nt, 5] = pnl; T[nt, 6] = mae; T[nt, 7] = own; nt += 1
            pos = 0; own = -1
        if pos != 0 and w[own] == pos:
            ti = i
        if pos == 0 and not np.isnan(atr[i]):
            for k in range(K):
                ok = True
                for jj in range(k):
                    if w[jj] != 0 and w[jj] != w[k]:
                        ok = False
                if ok and w[k] != 0 and w[k] != blk and w[k] != lockD and ((w[k] == 1 and EL[k, i]) or (w[k] == -1 and ES[k, i])):
                    pos = w[k]; own = k; ep = c[i] + TICK * pos; q = CAP / ep; ei = i; ti = i; mae = 0.0; ae = atr[i]; stop = ep - pos * TS[k] * ae
                    break
        eq[i] = real + (q * (c[i] - ep) * pos - FEE * q * ep if pos != 0 else 0.0)
    return eq, T[:nt]


@njit(cache=True)
def simN5(o, h, l, c, atr, AL, AS, EL, ES, TS, TH, spk, blkN, conflictExit):
    """ÇARPIŞMA = FLAT: motorlar aynı anda long ve short isterse pozisyon kapanır / girilmez ("dur, nefes al").
    Net yön netleşince: boşsa izinli motorla girer, ters pozisyondaysa doğrudan döner. Süre: o yönü isteyen en öncelikli motorun TH'ı."""
    K, n = AL.shape; eq = np.zeros(n); real = 0.0; pos = 0; own = -1; ep = 0.0; q = 0.0; stop = 0.0; ei = 0; ti = 0; mae = 0.0; noS = -1
    T = np.zeros((n // 2 + 10, 8)); nt = 0; w = np.zeros(K, np.int64)
    for i in range(1, n):
        if pos != 0:
            mae = max(mae, (ep - l[i]) / ep if pos == 1 else (h[i] - ep) / ep)
            if (pos == 1 and l[i] <= stop) or (pos == -1 and h[i] >= stop):
                xp = (min(o[i], stop) - TICK) if pos == 1 else (max(o[i], stop) + TICK)
                pnl = q * (xp - ep) * pos - FEE * q * (ep + xp); real += pnl
                T[nt, 0] = ei; T[nt, 1] = i; T[nt, 2] = pos; T[nt, 3] = ep; T[nt, 4] = xp; T[nt, 5] = pnl; T[nt, 6] = mae; T[nt, 7] = own; nt += 1
                pos = 0; own = -1
        if spk[i]:
            noS = i + blkN
        anyL = False; anyS = False; kL = -1; kS = -1
        for k in range(K):
            w[k] = 1 if AL[k, i] else (-1 if (AS[k, i] and i > noS) else 0)
            if w[k] == 1:
                anyL = True
                if kL < 0 and EL[k, i]:
                    kL = k
            elif w[k] == -1:
                anyS = True
                if kS < 0 and ES[k, i]:
                    kS = k
        conflict = anyL and anyS
        D = 0 if conflict else (1 if anyL else (-1 if anyS else 0))
        guard = pos == -1 and spk[i]
        if pos != 0:
            if D == pos:
                ti = i
                for k in range(K):
                    if w[k] == pos:
                        own = k
                        break
            tout = i - ti >= TH[own]
            if guard or (conflict and conflictExit) or D == -pos or tout:
                xp = c[i] - TICK * pos; pnl = q * (xp - ep) * pos - FEE * q * (ep + xp); real += pnl
                T[nt, 0] = ei; T[nt, 1] = i; T[nt, 2] = pos; T[nt, 3] = ep; T[nt, 4] = xp; T[nt, 5] = pnl; T[nt, 6] = mae; T[nt, 7] = own; nt += 1
                pos = 0; own = -1
        if pos == 0 and D != 0 and not np.isnan(atr[i]):
            kk = kL if D == 1 else kS
            if kk >= 0:
                pos = D; own = kk; ep = c[i] + TICK * pos; q = CAP / ep; ei = i; ti = i; mae = 0.0; stop = ep - pos * TS[kk] * atr[i]
        eq[i] = real + (q * (c[i] - ep) * pos - FEE * q * ep if pos != 0 else 0.0)
    return eq, T[:nt]


@njit(cache=True)
def simN6(o, h, l, c, atr, AL, AS, EL, ES, TS, TH, spk, blkN, mPers):
    """[simN6] simN3 + ÇARPIŞMA GİRİŞ KURALI: girişte hiçbir motor ters yön istememeli (oybirliği) ve bu durum mPers bar sürmeli.
    [simN3] simN2 + süre dolunca aynı yönü isteyen herhangi bir motor pozisyonu DEVRALIR (kapat-aynı yöne yeniden aç yok).
    [simN2] simN + giriş kuralı: daha öncelikli motor ters yön istiyorsa alt motor GİREMEZ (giriş-çıkış zinciri biter).
    N motor öncelik sırasıyla (0 en yüksek), tek 500 $ birim. Pozisyon sahibi k iken daha öncelikli j aynı yönü isterse devralır,
    ters isterse kapanır ve j girer. Boşta: en öncelikli izinli motor girer. T[:,7] = sahip motor."""
    K, n = AL.shape; eq = np.zeros(n); real = 0.0; pos = 0; own = -1; ep = 0.0; q = 0.0; stop = 0.0; ei = 0; ti = 0; mae = 0.0; noS = -1; ae = 0.0
    T = np.zeros((n // 2 + 10, 8)); nt = 0; w = np.zeros(K, np.int64); runD = 0; runN = 0
    for i in range(1, n):
        if pos != 0:
            mae = max(mae, (ep - l[i]) / ep if pos == 1 else (h[i] - ep) / ep)
            if (pos == 1 and l[i] <= stop) or (pos == -1 and h[i] >= stop):
                xp = (min(o[i], stop) - TICK) if pos == 1 else (max(o[i], stop) + TICK)
                pnl = q * (xp - ep) * pos - FEE * q * (ep + xp); real += pnl
                T[nt, 0] = ei; T[nt, 1] = i; T[nt, 2] = pos; T[nt, 3] = ep; T[nt, 4] = xp; T[nt, 5] = pnl; T[nt, 6] = mae; T[nt, 7] = own; nt += 1
                pos = 0; own = -1
        if spk[i]:
            noS = i + blkN
        for k in range(K):
            w[k] = 1 if AL[k, i] else (-1 if (AS[k, i] and i > noS) else 0)
        aL_ = False; aS_ = False
        for kk in range(K):
            if w[kk] == 1:
                aL_ = True
            elif w[kk] == -1:
                aS_ = True
        Dn = 0 if (aL_ and aS_) else (1 if aL_ else (-1 if aS_ else 0))
        if Dn != 0 and Dn == runD:
            runN += 1
        else:
            runD = Dn; runN = 1 if Dn != 0 else 0
        guard = pos == -1 and spk[i]
        if pos != 0 and not guard:
            for j in range(own):
                if w[j] == pos:
                    own = j; ti = i; stop = ep - pos * TS[j] * ae
                    break
        ex = False; blk = 0
        if pos != 0 and not guard and i - ti >= TH[own]:
            for kk in range(K):
                if w[kk] == pos:
                    own = kk; ti = i; stop = ep - pos * TS[kk] * ae
                    break
        if pos != 0:
            tout = i - ti >= TH[own]
            opp = False
            for j in range(own + 1):
                if w[j] != 0 and w[j] != pos:
                    opp = True
            ex = guard or opp or tout
            blk = pos if (tout and w[own] == pos) else 0
        if ex:
            xp = c[i] - TICK * pos; pnl = q * (xp - ep) * pos - FEE * q * (ep + xp); real += pnl
            T[nt, 0] = ei; T[nt, 1] = i; T[nt, 2] = pos; T[nt, 3] = ep; T[nt, 4] = xp; T[nt, 5] = pnl; T[nt, 6] = mae; T[nt, 7] = own; nt += 1
            pos = 0; own = -1
        if pos != 0 and w[own] == pos:
            ti = i
        if pos == 0 and not np.isnan(atr[i]):
            for k in range(K):
                ok = runD == w[k] and runN >= mPers
                if ok and w[k] != 0 and w[k] != blk and ((w[k] == 1 and EL[k, i]) or (w[k] == -1 and ES[k, i])):
                    pos = w[k]; own = k; ep = c[i] + TICK * pos; q = CAP / ep; ei = i; ti = i; mae = 0.0; ae = atr[i]; stop = ep - pos * TS[k] * ae
                    break
        eq[i] = real + (q * (c[i] - ep) * pos - FEE * q * ep if pos != 0 else 0.0)
    return eq, T[:nt]



@njit(cache=True)
def simN7(o, h, l, c, atr, AL, AS, EL, ES, TS, TH, spk, blkN, mPers, byp):
    """[simN7] simN6 + byp[k]=True olan motorlar (olay motorları, ör. POMPA) oybirliği/kalıcılık şartı olmadan girer.
    [simN6] simN3 + ÇARPIŞMA GİRİŞ KURALI: girişte hiçbir motor ters yön istememeli (oybirliği) ve bu durum mPers bar sürmeli.
    [simN3] simN2 + süre dolunca aynı yönü isteyen herhangi bir motor pozisyonu DEVRALIR (kapat-aynı yöne yeniden aç yok).
    [simN2] simN + giriş kuralı: daha öncelikli motor ters yön istiyorsa alt motor GİREMEZ (giriş-çıkış zinciri biter).
    N motor öncelik sırasıyla (0 en yüksek), tek 500 $ birim. Pozisyon sahibi k iken daha öncelikli j aynı yönü isterse devralır,
    ters isterse kapanır ve j girer. Boşta: en öncelikli izinli motor girer. T[:,7] = sahip motor."""
    K, n = AL.shape; eq = np.zeros(n); real = 0.0; pos = 0; own = -1; ep = 0.0; q = 0.0; stop = 0.0; ei = 0; ti = 0; mae = 0.0; noS = -1; ae = 0.0
    T = np.zeros((n // 2 + 10, 8)); nt = 0; w = np.zeros(K, np.int64); runD = 0; runN = 0
    for i in range(1, n):
        if pos != 0:
            mae = max(mae, (ep - l[i]) / ep if pos == 1 else (h[i] - ep) / ep)
            if (pos == 1 and l[i] <= stop) or (pos == -1 and h[i] >= stop):
                xp = (min(o[i], stop) - TICK) if pos == 1 else (max(o[i], stop) + TICK)
                pnl = q * (xp - ep) * pos - FEE * q * (ep + xp); real += pnl
                T[nt, 0] = ei; T[nt, 1] = i; T[nt, 2] = pos; T[nt, 3] = ep; T[nt, 4] = xp; T[nt, 5] = pnl; T[nt, 6] = mae; T[nt, 7] = own; nt += 1
                pos = 0; own = -1
        if spk[i]:
            noS = i + blkN
        for k in range(K):
            w[k] = 1 if AL[k, i] else (-1 if (AS[k, i] and i > noS) else 0)
        aL_ = False; aS_ = False
        for kk in range(K):
            if w[kk] == 1:
                aL_ = True
            elif w[kk] == -1:
                aS_ = True
        Dn = 0 if (aL_ and aS_) else (1 if aL_ else (-1 if aS_ else 0))
        if Dn != 0 and Dn == runD:
            runN += 1
        else:
            runD = Dn; runN = 1 if Dn != 0 else 0
        guard = pos == -1 and spk[i]
        if pos != 0 and not guard:
            for j in range(own):
                if w[j] == pos:
                    own = j; ti = i; stop = ep - pos * TS[j] * ae
                    break
        ex = False; blk = 0
        if pos != 0 and not guard and i - ti >= TH[own]:
            for kk in range(K):
                if w[kk] == pos:
                    own = kk; ti = i; stop = ep - pos * TS[kk] * ae
                    break
        if pos != 0:
            tout = i - ti >= TH[own]
            opp = False
            for j in range(own + 1):
                if w[j] != 0 and w[j] != pos:
                    opp = True
            ex = guard or opp or tout
            blk = pos if (tout and w[own] == pos) else 0
        if ex:
            xp = c[i] - TICK * pos; pnl = q * (xp - ep) * pos - FEE * q * (ep + xp); real += pnl
            T[nt, 0] = ei; T[nt, 1] = i; T[nt, 2] = pos; T[nt, 3] = ep; T[nt, 4] = xp; T[nt, 5] = pnl; T[nt, 6] = mae; T[nt, 7] = own; nt += 1
            pos = 0; own = -1
        if pos != 0 and w[own] == pos:
            ti = i
        if pos == 0 and not np.isnan(atr[i]):
            for k in range(K):
                ok = byp[k] or (runD == w[k] and runN >= mPers)
                if ok and w[k] != 0 and w[k] != blk and ((w[k] == 1 and EL[k, i]) or (w[k] == -1 and ES[k, i])):
                    pos = w[k]; own = k; ep = c[i] + TICK * pos; q = CAP / ep; ei = i; ti = i; mae = 0.0; ae = atr[i]; stop = ep - pos * TS[k] * ae
                    break
        eq[i] = real + (q * (c[i] - ep) * pos - FEE * q * ep if pos != 0 else 0.0)
    return eq, T[:nt]



@njit(cache=True)
def simN8(o, h, l, c, atr, AL, AS, EL, ES, TS, TH, spk, blkN, mPers, byp, cap, rev, FL, FS):
    """[simN8] simN7 + ZARAR TAVANI: stop mesafesi = min(TS×ATR, cap×giriş). rev=1 ise tavan stopunda FL/FS izin veriyorsa
    aynı fiyattan TERS yöne MANEVRA (sahip = satır 0, boş satır; yalnız stop/süre ile çıkar).
    [simN7] simN6 + byp[k]=True olan motorlar (olay motorları, ör. POMPA) oybirliği/kalıcılık şartı olmadan girer.
    [simN6] simN3 + ÇARPIŞMA GİRİŞ KURALI: girişte hiçbir motor ters yön istememeli (oybirliği) ve bu durum mPers bar sürmeli.
    [simN3] simN2 + süre dolunca aynı yönü isteyen herhangi bir motor pozisyonu DEVRALIR (kapat-aynı yöne yeniden aç yok).
    [simN2] simN + giriş kuralı: daha öncelikli motor ters yön istiyorsa alt motor GİREMEZ (giriş-çıkış zinciri biter).
    N motor öncelik sırasıyla (0 en yüksek), tek 500 $ birim. Pozisyon sahibi k iken daha öncelikli j aynı yönü isterse devralır,
    ters isterse kapanır ve j girer. Boşta: en öncelikli izinli motor girer. T[:,7] = sahip motor."""
    K, n = AL.shape; eq = np.zeros(n); real = 0.0; pos = 0; own = -1; ep = 0.0; q = 0.0; stop = 0.0; ei = 0; ti = 0; mae = 0.0; noS = -1; ae = 0.0
    T = np.zeros((n // 2 + 10, 8)); nt = 0; w = np.zeros(K, np.int64); runD = 0; runN = 0
    for i in range(1, n):
        if pos != 0:
            mae = max(mae, (ep - l[i]) / ep if pos == 1 else (h[i] - ep) / ep)
            if (pos == 1 and l[i] <= stop) or (pos == -1 and h[i] >= stop):
                xp = (min(o[i], stop) - TICK) if pos == 1 else (max(o[i], stop) + TICK)
                pnl = q * (xp - ep) * pos - FEE * q * (ep + xp); real += pnl
                T[nt, 0] = ei; T[nt, 1] = i; T[nt, 2] = pos; T[nt, 3] = ep; T[nt, 4] = xp; T[nt, 5] = pnl; T[nt, 6] = mae; T[nt, 7] = own; nt += 1
                op = pos; pos = 0; own = -1
                if rev and cap > 0.0 and ((op == 1 and FS[i]) or (op == -1 and FL[i])):
                    pos = -op; own = 0; ep = xp; q = CAP / ep; ei = i; ti = i; mae = 0.0; stop = ep - pos * cap * ep
        if spk[i]:
            noS = i + blkN
        for k in range(K):
            w[k] = 1 if AL[k, i] else (-1 if (AS[k, i] and i > noS) else 0)
        aL_ = False; aS_ = False
        for kk in range(K):
            if w[kk] == 1:
                aL_ = True
            elif w[kk] == -1:
                aS_ = True
        Dn = 0 if (aL_ and aS_) else (1 if aL_ else (-1 if aS_ else 0))
        if Dn != 0 and Dn == runD:
            runN += 1
        else:
            runD = Dn; runN = 1 if Dn != 0 else 0
        guard = pos == -1 and spk[i]
        if pos != 0 and not guard:
            for j in range(own):
                if w[j] == pos:
                    own = j; ti = i; stop = ep - pos * (min(TS[j] * ae, cap * ep) if cap > 0.0 else TS[j] * ae)
                    break
        ex = False; blk = 0
        if pos != 0 and not guard and i - ti >= TH[own]:
            for kk in range(K):
                if w[kk] == pos:
                    own = kk; ti = i; stop = ep - pos * (min(TS[kk] * ae, cap * ep) if cap > 0.0 else TS[kk] * ae)
                    break
        if pos != 0:
            tout = i - ti >= TH[own]
            opp = False
            for j in range(own + 1):
                if w[j] != 0 and w[j] != pos:
                    opp = True
            ex = guard or opp or tout
            blk = pos if (tout and w[own] == pos) else 0
        if ex:
            xp = c[i] - TICK * pos; pnl = q * (xp - ep) * pos - FEE * q * (ep + xp); real += pnl
            T[nt, 0] = ei; T[nt, 1] = i; T[nt, 2] = pos; T[nt, 3] = ep; T[nt, 4] = xp; T[nt, 5] = pnl; T[nt, 6] = mae; T[nt, 7] = own; nt += 1
            pos = 0; own = -1
        if pos != 0 and w[own] == pos:
            ti = i
        if pos == 0 and not np.isnan(atr[i]):
            for k in range(K):
                ok = byp[k] or (runD == w[k] and runN >= mPers)
                if ok and w[k] != 0 and w[k] != blk and ((w[k] == 1 and EL[k, i]) or (w[k] == -1 and ES[k, i])):
                    pos = w[k]; own = k; ep = c[i] + TICK * pos; q = CAP / ep; ei = i; ti = i; mae = 0.0; ae = atr[i]; stop = ep - pos * (min(TS[k] * ae, cap * ep) if cap > 0.0 else TS[k] * ae)
                    break
        eq[i] = real + (q * (c[i] - ep) * pos - FEE * q * ep if pos != 0 else 0.0)
    return eq, T[:nt]



@njit(cache=True)
def simN9(o, h, l, c, atr, AL, AS, EL, ES, TS, TH, spk, blkN, mPers, byp, cap, rev, FL, FS, fN, fM, fR):
    """[simN9] simN8 + BASKI/ULAŞAMAMA: girişten fN bar sonra (kapanışta) pozisyon yönünde fM×ATR mesafeye hiç ulaşılmadıysa
    kapat + aynı kapanışta TERS yöne aç (sahip satır 0; stop cap, süre TH[0]). fR=0 ise yalnız kapat.
    [simN8] simN7 + ZARAR TAVANI: stop mesafesi = min(TS×ATR, cap×giriş). rev=1 ise tavan stopunda FL/FS izin veriyorsa
    aynı fiyattan TERS yöne MANEVRA (sahip = satır 0, boş satır; yalnız stop/süre ile çıkar).
    [simN7] simN6 + byp[k]=True olan motorlar (olay motorları, ör. POMPA) oybirliği/kalıcılık şartı olmadan girer.
    [simN6] simN3 + ÇARPIŞMA GİRİŞ KURALI: girişte hiçbir motor ters yön istememeli (oybirliği) ve bu durum mPers bar sürmeli.
    [simN3] simN2 + süre dolunca aynı yönü isteyen herhangi bir motor pozisyonu DEVRALIR (kapat-aynı yöne yeniden aç yok).
    [simN2] simN + giriş kuralı: daha öncelikli motor ters yön istiyorsa alt motor GİREMEZ (giriş-çıkış zinciri biter).
    N motor öncelik sırasıyla (0 en yüksek), tek 500 $ birim. Pozisyon sahibi k iken daha öncelikli j aynı yönü isterse devralır,
    ters isterse kapanır ve j girer. Boşta: en öncelikli izinli motor girer. T[:,7] = sahip motor."""
    K, n = AL.shape; eq = np.zeros(n); real = 0.0; pos = 0; own = -1; ep = 0.0; q = 0.0; stop = 0.0; ei = 0; ti = 0; mae = 0.0; noS = -1; ae = 0.0
    hs = 0.0; ls = 0.0; lastE = -1; T = np.zeros((n // 2 + 10, 8)); nt = 0; w = np.zeros(K, np.int64); runD = 0; runN = 0
    for i in range(1, n):
        if pos != 0:
            mae = max(mae, (ep - l[i]) / ep if pos == 1 else (h[i] - ep) / ep)
            if ei != lastE:
                hs = h[i]; ls = l[i]; lastE = ei
            else:
                hs = max(hs, h[i]); ls = min(ls, l[i])
            if (pos == 1 and l[i] <= stop) or (pos == -1 and h[i] >= stop):
                xp = (min(o[i], stop) - TICK) if pos == 1 else (max(o[i], stop) + TICK)
                pnl = q * (xp - ep) * pos - FEE * q * (ep + xp); real += pnl
                T[nt, 0] = ei; T[nt, 1] = i; T[nt, 2] = pos; T[nt, 3] = ep; T[nt, 4] = xp; T[nt, 5] = pnl; T[nt, 6] = mae; T[nt, 7] = own; nt += 1
                op = pos; pos = 0; own = -1
                if rev and cap > 0.0 and ((op == 1 and FS[i]) or (op == -1 and FL[i])):
                    pos = -op; own = 0; ep = xp; q = CAP / ep; ei = i; ti = i; mae = 0.0; stop = ep - pos * cap * ep
        if fN > 0 and pos != 0 and own != 0 and i - ei == fN:
            if (pos == 1 and hs < ep + fM * ae) or (pos == -1 and ls > ep - fM * ae):
                xp = c[i] - TICK * pos; pnl = q * (xp - ep) * pos - FEE * q * (ep + xp); real += pnl
                T[nt, 0] = ei; T[nt, 1] = i; T[nt, 2] = pos; T[nt, 3] = ep; T[nt, 4] = xp; T[nt, 5] = pnl; T[nt, 6] = mae; T[nt, 7] = own; nt += 1
                op = pos; pos = 0; own = -1
                if fR:
                    pos = -op; own = 0; ep = xp; q = CAP / ep; ei = i; ti = i; mae = 0.0; stop = ep - pos * cap * ep
        if spk[i]:
            noS = i + blkN
        for k in range(K):
            w[k] = 1 if AL[k, i] else (-1 if (AS[k, i] and i > noS) else 0)
        aL_ = False; aS_ = False
        for kk in range(K):
            if w[kk] == 1:
                aL_ = True
            elif w[kk] == -1:
                aS_ = True
        Dn = 0 if (aL_ and aS_) else (1 if aL_ else (-1 if aS_ else 0))
        if Dn != 0 and Dn == runD:
            runN += 1
        else:
            runD = Dn; runN = 1 if Dn != 0 else 0
        guard = pos == -1 and spk[i]
        if pos != 0 and not guard:
            for j in range(own):
                if w[j] == pos:
                    own = j; ti = i; stop = ep - pos * (min(TS[j] * ae, cap * ep) if cap > 0.0 else TS[j] * ae)
                    break
        ex = False; blk = 0
        if pos != 0 and not guard and i - ti >= TH[own]:
            for kk in range(K):
                if w[kk] == pos:
                    own = kk; ti = i; stop = ep - pos * (min(TS[kk] * ae, cap * ep) if cap > 0.0 else TS[kk] * ae)
                    break
        if pos != 0:
            tout = i - ti >= TH[own]
            opp = False
            for j in range(own + 1):
                if w[j] != 0 and w[j] != pos:
                    opp = True
            ex = guard or opp or tout
            blk = pos if (tout and w[own] == pos) else 0
        if ex:
            xp = c[i] - TICK * pos; pnl = q * (xp - ep) * pos - FEE * q * (ep + xp); real += pnl
            T[nt, 0] = ei; T[nt, 1] = i; T[nt, 2] = pos; T[nt, 3] = ep; T[nt, 4] = xp; T[nt, 5] = pnl; T[nt, 6] = mae; T[nt, 7] = own; nt += 1
            pos = 0; own = -1
        if pos != 0 and w[own] == pos:
            ti = i
        if pos == 0 and not np.isnan(atr[i]):
            for k in range(K):
                ok = byp[k] or (runD == w[k] and runN >= mPers)
                if ok and w[k] != 0 and w[k] != blk and ((w[k] == 1 and EL[k, i]) or (w[k] == -1 and ES[k, i])):
                    pos = w[k]; own = k; ep = c[i] + TICK * pos; q = CAP / ep; ei = i; ti = i; mae = 0.0; ae = atr[i]; stop = ep - pos * (min(TS[k] * ae, cap * ep) if cap > 0.0 else TS[k] * ae)
                    break
        eq[i] = real + (q * (c[i] - ep) * pos - FEE * q * ep if pos != 0 else 0.0)
    return eq, T[:nt]



@njit(cache=True)
def simN10(o, h, l, c, atr, AL, AS, EL, ES, TS, TH, spk, blkN, mPers, byp, cap, rev, FL, FS, fN, fM, mode):
    """[simN10] simN8 + GÖLGE BASKI TESTİ (arşiv Fikir 2): giriş sinyali gerçek pozisyon açmaz, gölge test başlatır.
    i>shT barlarında yön hedefi (shP ± fM×ATR) bar içinde görülürse: mode 1 → hedef fiyattan (stop-giriş) orijinal yöne gir; mode 2 → girme.
    fN bar dolup ulaşılamazsa → o barın kapanışında TERS yöne gir (sahip satır 0, stop cap, süre TH[0]). Atlayan (byp) motorlar normal girer.
    [simN8] simN7 + ZARAR TAVANI: stop mesafesi = min(TS×ATR, cap×giriş). rev=1 ise tavan stopunda FL/FS izin veriyorsa
    aynı fiyattan TERS yöne MANEVRA (sahip = satır 0, boş satır; yalnız stop/süre ile çıkar).
    [simN7] simN6 + byp[k]=True olan motorlar (olay motorları, ör. POMPA) oybirliği/kalıcılık şartı olmadan girer.
    [simN6] simN3 + ÇARPIŞMA GİRİŞ KURALI: girişte hiçbir motor ters yön istememeli (oybirliği) ve bu durum mPers bar sürmeli.
    [simN3] simN2 + süre dolunca aynı yönü isteyen herhangi bir motor pozisyonu DEVRALIR (kapat-aynı yöne yeniden aç yok).
    [simN2] simN + giriş kuralı: daha öncelikli motor ters yön istiyorsa alt motor GİREMEZ (giriş-çıkış zinciri biter).
    N motor öncelik sırasıyla (0 en yüksek), tek 500 $ birim. Pozisyon sahibi k iken daha öncelikli j aynı yönü isterse devralır,
    ters isterse kapanır ve j girer. Boşta: en öncelikli izinli motor girer. T[:,7] = sahip motor."""
    K, n = AL.shape; eq = np.zeros(n); real = 0.0; pos = 0; own = -1; ep = 0.0; q = 0.0; stop = 0.0; ei = 0; ti = 0; mae = 0.0; noS = -1; ae = 0.0
    shD = 0; shP = 0.0; shA = 0.0; shT = 0; shK = 0; tgt = 0.0; T = np.zeros((n // 2 + 10, 8)); nt = 0; w = np.zeros(K, np.int64); runD = 0; runN = 0
    for i in range(1, n):
        if pos != 0:
            mae = max(mae, (ep - l[i]) / ep if pos == 1 else (h[i] - ep) / ep)
            if (pos == 1 and l[i] <= stop) or (pos == -1 and h[i] >= stop):
                xp = (min(o[i], stop) - TICK) if pos == 1 else (max(o[i], stop) + TICK)
                pnl = q * (xp - ep) * pos - FEE * q * (ep + xp); real += pnl
                T[nt, 0] = ei; T[nt, 1] = i; T[nt, 2] = pos; T[nt, 3] = ep; T[nt, 4] = xp; T[nt, 5] = pnl; T[nt, 6] = mae; T[nt, 7] = own; nt += 1
                op = pos; pos = 0; own = -1
                if rev and cap > 0.0 and ((op == 1 and FS[i]) or (op == -1 and FL[i])):
                    pos = -op; own = 0; ep = xp; q = CAP / ep; ei = i; ti = i; mae = 0.0; stop = ep - pos * cap * ep
        if spk[i]:
            noS = i + blkN
        for k in range(K):
            w[k] = 1 if AL[k, i] else (-1 if (AS[k, i] and i > noS) else 0)
        aL_ = False; aS_ = False
        for kk in range(K):
            if w[kk] == 1:
                aL_ = True
            elif w[kk] == -1:
                aS_ = True
        Dn = 0 if (aL_ and aS_) else (1 if aL_ else (-1 if aS_ else 0))
        if Dn != 0 and Dn == runD:
            runN += 1
        else:
            runD = Dn; runN = 1 if Dn != 0 else 0
        guard = pos == -1 and spk[i]
        if pos != 0 and not guard:
            for j in range(own):
                if w[j] == pos:
                    own = j; ti = i; stop = ep - pos * (min(TS[j] * ae, cap * ep) if cap > 0.0 else TS[j] * ae)
                    break
        ex = False; blk = 0
        if pos != 0 and not guard and i - ti >= TH[own]:
            for kk in range(K):
                if w[kk] == pos:
                    own = kk; ti = i; stop = ep - pos * (min(TS[kk] * ae, cap * ep) if cap > 0.0 else TS[kk] * ae)
                    break
        if pos != 0:
            tout = i - ti >= TH[own]
            opp = False
            for j in range(own + 1):
                if w[j] != 0 and w[j] != pos:
                    opp = True
            ex = guard or opp or tout
            blk = pos if (tout and w[own] == pos) else 0
        if ex:
            xp = c[i] - TICK * pos; pnl = q * (xp - ep) * pos - FEE * q * (ep + xp); real += pnl
            T[nt, 0] = ei; T[nt, 1] = i; T[nt, 2] = pos; T[nt, 3] = ep; T[nt, 4] = xp; T[nt, 5] = pnl; T[nt, 6] = mae; T[nt, 7] = own; nt += 1
            pos = 0; own = -1
        if pos != 0 and w[own] == pos:
            ti = i
        if pos != 0:
            shD = 0
        if pos == 0 and shD != 0 and i > shT:
            tgt = shP + shD * fM * shA
            if (shD == 1 and h[i] >= tgt) or (shD == -1 and l[i] <= tgt):
                if mode == 1:
                    pos = shD; own = shK; ep = (max(o[i], tgt) + TICK) if pos == 1 else (min(o[i], tgt) - TICK); q = CAP / ep; ei = i; ti = i; mae = 0.0; ae = shA
                    stop = ep - pos * (min(TS[shK] * ae, cap * ep) if cap > 0.0 else TS[shK] * ae)
                shD = 0
            elif i - shT >= fN:
                pos = -shD; own = 0; ep = c[i] + TICK * pos; q = CAP / ep; ei = i; ti = i; mae = 0.0; ae = shA; stop = ep - pos * cap * ep
                shD = 0
        if pos == 0 and not np.isnan(atr[i]):
            for k in range(K):
                if shD != 0 and not byp[k]:
                    continue
                ok = byp[k] or (runD == w[k] and runN >= mPers)
                if ok and w[k] != 0 and w[k] != blk and ((w[k] == 1 and EL[k, i]) or (w[k] == -1 and ES[k, i])):
                    if fN > 0 and not byp[k]:
                        shD = w[k]; shP = c[i]; shA = atr[i]; shT = i; shK = k
                    else:
                        pos = w[k]; own = k; ep = c[i] + TICK * pos; q = CAP / ep; ei = i; ti = i; mae = 0.0; ae = atr[i]; stop = ep - pos * (min(TS[k] * ae, cap * ep) if cap > 0.0 else TS[k] * ae)
                        shD = 0
                    break
        eq[i] = real + (q * (c[i] - ep) * pos - FEE * q * ep if pos != 0 else 0.0)
    return eq, T[:nt]
