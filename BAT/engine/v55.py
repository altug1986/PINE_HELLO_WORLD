import numpy as np, numba as nb, pandas as pd
from feat import *            # D, c, t, n, F, ADX15, DI15, NR, BADX, BDI
from v54 import target, S0, RA0, TICK, BQ
from lib import simR
from met import metrics, SPLIT
BV = pd.Series(BQ).rolling(6).sum(); BV = np.nan_to_num((BV / BV.rolling(1728).mean()).to_numpy(), nan=1.0)   # BTC 30 dk hacim ÷ 6 gün ort.
HR = ((t // 3600000) % 24).astype(np.int64)
A15 = np.nan_to_num(ADX15); D15 = np.nan_to_num(DI15); VR = np.nan_to_num(F['volr'], nan=0.5)
MOM = {k: np.nan_to_num(np.r_[np.full(k, np.nan), (c[k:] / c[:-k] - 1) * 100]) for k in (12, 48, 144, 288)}
TG = target(1.0, 1.5, 4.0)
TSG = np.zeros(n, np.int64)
ZC = np.zeros(n, np.int64)
STP = np.zeros(n)
GL = np.zeros(n, np.int64)
GS = np.zeros(n, np.int64)
DLL = np.zeros(n)
DHH = np.full(n, 1e18)
from eng import dmi, htf_map
z5 = np.load('d5.npz')
def _hz(b, k, mp): p_, m_, a_ = dmi(b[1], b[2], b[3], 14, 14); return np.nan_to_num(htf_map(a_, k, mp)), np.nan_to_num(htf_map(p_ - m_, k, mp))
_k4 = t // 14400000; _h4, _st = np.unique(_k4, return_index=True); _en = np.r_[_st[1:], n]
_b4 = np.stack([D['o'][_st], np.maximum.reduceat(D['h'], _st), np.minimum.reduceat(D['l'], _st), c[_en - 1]])
TRF = dict(a15=(A15, D15), a30=_hz(z5['b30'], z5['b30_k'], z5['b30_map']), a60=_hz(z5['b60'], z5['b60_k'], z5['b60_map']), a240=_hz(_b4, _h4, _k4),
           x60=(BADX, BDI), x240=_hz(z5['x240'], z5['x240_k'], z5['x240_map']))
# q: 0 S,1 rA,2 L,3 a,4 vk,5 cS,6 N | 7 eC,8 aE | 9 H,10 xH,11 rH | 12 aT,13 dT,14 sT | 15 bvk,16 mk,17 sB | 18 nrMin,19 vrMin | 20 hmask | 21 mode-entry-dir(0 any)
@nb.njit(cache=True)
def layer(c, tg, BADX, BDI, NR, A15, D15, BV, MO, VR, HR, TRD, TDI, TSG, ZC, STP, GL, GS, DLL, DHH, q, tick):
    n = len(c); out = np.zeros(n, np.int64); why = np.zeros(n, np.int8)
    sh = 0; ep = 0.0; eb = 0; lock = 0; lext = 0.0; lockR = 0.0; prevT = 0; cnt = 0; cPx = 0.0; cBar = 0
    S, rA, L, a, vk, cS, N = q[0], q[1], q[2], q[3], q[4], q[5], q[6]
    eC, aE, H, xH, rH, aT, dT, sT, bvk, mk, sB, nrMin, vrMin = q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19]
    hm = np.int64(q[20]); tpP, tpG, rTP = q[21], q[22], q[23]; pk = 0.0
    kS, xS, wS, aH = q[24], q[25], q[26], q[27]; tS, tA, tD, beP, beL, beT = q[28], q[29], q[30], q[31], q[32], q[33]; gm = np.int64(q[34]); zK1, zK2, zX, zM, zC, zAct = q[35], q[36], q[37], q[38], q[39], np.int64(q[40]); cOrig = 0; dOn, dMin = q[41], q[42]; stN = 0; stS = 0.0; pU = -1; need = 0
    for i in range(n):
        t_ = tg[i]; r = 1
        if t_ != prevT:
            lock = 0; cnt = 0; prevT = t_
            if i >= pU: need = 0
            if t_ != 0 and eC > 0 and A15[i] >= aE:
                lock = t_; lext = c[i]; lockR = eC
        want = t_
        if cnt != 0:
            g = (c[i] / cPx - 1) * cnt * 100
            if g <= -cS or i - cBar >= N:
                cnt = 0; r = 4
                if zAct == 3 and cOrig != 0: lock = cOrig; lext = c[i]; lockR = rA
            else: want = cnt
        if cnt == 0 and zAct > 0 and sh != 0 and want == sh and i - eb >= zK1 and i - eb <= zK2:
            plz = (c[i] / ep - 1) * sh * 100
            if plz <= -zX and (pk / ep - 1) * sh * 100 < zM and (zC == 0 or ZC[i] == -sh):
                if zAct == 1:
                    lock = sh; lext = c[i]; lockR = rA; want = 0; r = 12
                else:
                    cOrig = sh; cnt = -sh; cPx = c[i]; cBar = i; want = cnt; r = 13
        if cnt == 0:
            if sh != 0 and want == sh and L < 50:
                loss = (c[i] / ep - 1) * sh * 100
                if loss <= -L and BADX[i] > a and BDI[i] * sh < 0 and NR[i] >= vk:
                    cnt = -sh; cPx = c[i]; cBar = i; want = cnt; r = 3
            if cnt == 0:
                if sh != 0 and want == sh:
                    pl = (c[i] / ep - 1) * sh * 100
                    Se = S if STP[i] <= 0 else STP[i]; rr = 2
                    trd = TRD[i] >= tA and (tD == 0 or TDI[i] * sh < 0)
                    if tS > 0 and trd: Se = tS; rr = 9
                    if beP > 0 and (pk / ep - 1) * sh * 100 >= beP and (beT == 0 or trd) and -beL < Se: Se = -beL; rr = 10
                    if pl <= -Se:
                        lock = sh; lext = c[i]; lockR = rA; want = 0; r = rr
                    elif gm == 4 and TSG[i] == -sh:
                        lock = sh; lext = c[i]; lockR = rA; want = 0; r = 11
                    elif H > 0 and i - eb >= H and pl <= xH and (aH <= 0 or A15[i] < aH):
                        lock = sh; lext = c[i]; lockR = rH; want = 0; r = 5
                    elif aT > 0 and A15[i] >= aT and D15[i] * sh <= -dT and pl <= -sT:
                        lock = sh; lext = c[i]; lockR = rA; want = 0; r = 6
                    elif bvk > 0 and BV[i] >= bvk and MO[i] * sh <= -mk and pl <= -sB:
                        lock = sh; lext = c[i]; lockR = rA; want = 0; r = 7
                    elif tpP > 0 and (pk / ep - 1) * sh * 100 >= tpP and (c[i] / pk - 1) * sh * 100 <= -tpG:
                        lock = sh; lext = c[i]; lockR = rTP; want = 0; r = 8
                if lock != 0 and want == lock:
                    if lock > 0: lext = min(lext, c[i])
                    else: lext = max(lext, c[i])
                    if (c[i] / lext - 1) * lock * 100 >= lockR: lock = 0
                    else: want = 0
        if kS > 0 and (i < pU or need == 1):
            want = 0; cnt = 0
        if want != 0 and want != sh and cnt == 0:
            if NR[i] < nrMin or VR[i] < vrMin or (hm >> HR[i]) & 1 == 1:
                want = 0
            elif (gm == 1 or gm == 4) and TSG[i] == -want:
                want = 0
            elif gm == 2 and TSG[i] == 0:
                want = 0
            elif gm == 3 and TSG[i] == want:
                want = 0
            elif gm == 5 and TSG[i] == -want:
                want = TSG[i]
            elif (want == 1 and GL[i] == 1) or (want == -1 and GS[i] == 1):
                want = 0
        if dOn > 0 and sh != 0 and want != sh and r == 1 and lock == 0 and (c[i] / ep - 1) * sh * 100 >= dMin:
            if (sh > 0 and c[i] > DLL[i]) or (sh < 0 and c[i] < DHH[i]):
                want = sh
        if want != sh:
            if sh != 0:
                why[i] = r
                if kS > 0:
                    plx = ((c[i] - tick * sh) / ep - 1) * sh * 100 - 0.08
                    if plx < 0:
                        stN += 1; stS += plx
                        if stN >= kS and stS <= -xS:
                            pU = i + np.int64(wS); need = 1; stN = 0; stS = 0.0
                    else:
                        stN = 0; stS = 0.0
            sh = want; ep = c[i] + tick * want; eb = i; pk = c[i]
        if sh > 0: pk = max(pk, c[i])
        elif sh < 0: pk = min(pk, c[i])
        out[i] = sh
    return out, why
Q0 = dict(S=S0, rA=RA0, L=5.0, a=40.0, vk=2.0, cS=5.0, N=288, eC=0.0, aE=0.0, H=0, xH=0.0, rH=99.0, aT=0.0, dT=0.0, sT=0.0, bvk=0.0, mk=0.0, sB=0.0, nrMin=0.0, vrMin=0.0, hm=0, tpP=0.0, tpG=0.0, rTP=99.0, kS=0, xS=0.0, wS=0, aH=0.0, tS=0.0, tA=999.0, tD=0, beP=0.0, beL=0.0, beT=0, gm=0, zK1=10, zK2=50, zX=99.0, zM=99.0, zC=0, zAct=0, dOn=0, dMin=0.0, mo=48, tf='a15')
KEYS = [k for k in Q0 if k not in ('mo', 'tf')]
def run(p):
    pp = dict(Q0); pp.update(p); q = np.array([float(pp[k]) for k in KEYS])
    TA, TD_ = TRF[pp['tf']]
    return layer(c, TG, BADX, BDI, NR, A15, D15, BV, MOM[int(pp['mo'])], VR, HR, TA, TD_, TSG, ZC, STP, GL, GS, DLL, DHH, q, TICK)
k0 = np.searchsorted(t, SPLIT); km = np.searchsorted(t, np.int64(pd.Timestamp('2022-06-15').value // 10**6))
def cutD(a_, b_): return {kk: (v[a_:b_] if isinstance(v, np.ndarray) and len(v) == n else v) for kk, v in D.items()}
HV = [cutD(0, km), cutD(km, k0)]
def blk2(R):
    x = pd.to_datetime(t[R['xb']], unit='ms'); b = pd.Series(R['pnl']).groupby(((x.year * 12 + x.month - 1) // 2).values).sum()
    return dict(bmin=round(b.min()), bstd=round(b.std()), bneg=int((b < 0).sum()), b50=int((b < 50).sum()))
def evalp(p, full=True):
    tg, why = run(p); out = {}
    for j, ((a_, b_), Dh) in enumerate(zip([(0, km), (km, k0)], HV)):
        m = metrics(Dh, simR(Dh, tg[a_:b_])); out[f'h{j}'] = round(m['net'] - 1.5 * m['p98'])
    out['sc'] = min(out['h0'], out['h1'])
    if full:
        R = simR(D, tg); m = metrics(D, R); out.update({k: m[k] for k in ['net', 'n', 'pf', 'sharpe', 'calmar', 'p95', 'p98', 'p99', 'mddc', 'mddi', 'ulcer', 'IS', 'OOS', 'posM', 'last12']}); out.update(blk2(R)); xt_ = t[R['xb']]; out['L12'] = round(R['pnl'][xt_ >= t[-1] - 365 * 86400000].sum()); l12 = R['pnl'][xt_ >= t[-1] - 365 * 86400000]; out['L200'] = round(R['pnl'][xt_ >= t[-1] - 200 * 86400000].sum()); out['P200'] = round(R['pnl'][(xt_ >= t[-1] - 400 * 86400000) & (xt_ < t[-1] - 200 * 86400000)].sum()); out['L12x'] = round(l12.sum() - np.sort(l12)[-3:].sum()); out['L24'] = round(R['pnl'][(xt_ >= t[-1] - 730 * 86400000) & (xt_ < t[-1] - 365 * 86400000)].sum())
    return out
if __name__ == '__main__':
    import time; t0 = time.time(); e = evalp({}); print(e, f'{time.time()-t0:.1f}s')
