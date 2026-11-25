# ZEC'in kendi düşüşünü küçültme (sabit 500 $ emir): oynaklık kapısı, azami tutma, günlük zarar sınırı, özsermaye kesicisi → portföy ΔSC4
import sys; sys.path.insert(0, '/home/claude/zecwork'); sys.argv = ['x', 'yok']
import pfsearch as P; import numpy as np, pandas as pd, numba, itertools, time
from pfsearch import o, h, l, c, posA, bEp, bXp, bXd, R10, tms, n, TICK, ix, S
atrp = P.a / c * 100; day = (tms // 86400000).astype(np.int64)
@numba.njit(cache=True)
def ov6(o, h, l, c, posA, bEp, bXp, bXd, R10, atrp, volMax, volAct, maxH, dayLoss, ddBrk, pauseB, day, tick):
    n = len(c); pos = 0; ep = 0.0; q = 0.0; real = 0.0; mode = 0; prevD = 0; tPx = 0.0; tBar = 0; tSide = 0; rStp = 0.0; eb = 0
    eq = np.zeros(n); pk = 0.0; pauseUntil = -1; d0 = day[0]; dStart = 0.0; dayOff = False; holdOff = False
    for i in range(1, n):
        if day[i] != d0: d0 = day[i]; dStart = eq[i - 1]; dayOff = False
        if pos != 0:
            xp = -1.0
            if mode == 2 and pos == -tSide:
                if (pos == 1 and l[i] <= rStp) or (pos == -1 and h[i] >= rStp):
                    xp = (min(o[i], rStp) - tick) if pos == 1 else (max(o[i], rStp) + tick); mode = 0
            elif not np.isnan(bXp[i]) and pos == bXd[i]:
                xp = bXp[i]
            if xp > 0:
                real += q * (xp - ep) * pos - 0.0004 * q * (ep + xp); pos = 0
        D = posA[i]; want = D
        if D != prevD:
            mode = 0; holdOff = False
        elif D != 0:
            if mode == 0:
                if (D > 0 and R10[i] >= 94) or (D < 0 and R10[i] <= 6):
                    want = -D; mode = 2; tPx = c[i]; tBar = i; tSide = D
            else:
                bi = (R10[i] <= 70 or i - tBar >= 240 or c[i] >= tPx * 1.03) if tSide > 0 else (R10[i] >= 30 or i - tBar >= 240 or c[i] <= tPx * 0.97)
                if bi: mode = 0
                else: want = -tSide if pos == -tSide else 0
        prevD = D
        cur = real + (q * (c[i] - ep) * pos - 0.0004 * q * ep if pos != 0 else 0.0)
        if cur > pk: pk = cur
        if ddBrk > 0 and pk - cur >= ddBrk and pauseUntil < i: pauseUntil = i + pauseB; pk = cur
        if dayLoss > 0 and cur - dStart <= -dayLoss: dayOff = True
        if maxH > 0 and pos != 0 and i - eb >= maxH: holdOff = True
        if volMax > 0 and atrp[i] > volMax:
            if want != 0 and pos != want: want = pos if volAct == 1 else 0
            if volAct == 2: want = 0
        if i < pauseUntil or dayOff or holdOff: want = 0
        if want != pos:
            if pos != 0:
                xp = c[i] - tick * pos; real += q * (xp - ep) * pos - 0.0004 * q * (ep + xp); pos = 0
            if want != 0:
                pos = want; ep = c[i] + tick * pos; q = 500.0 / c[i]; eb = i
                if mode == 2 and want == -tSide: rStp = ep * (1.0 - want * 0.02)
        eq[i] = real + (q * (c[i] - ep) * pos - 0.0004 * q * ep if pos != 0 else 0.0)
    return eq
rows = []; t0 = time.time(); qs = np.nanpercentile(atrp, [70, 80, 90, 95])
print('ATR% yüzdelik 70/80/90/95:', np.round(qs, 3), flush=True)
def go(lab, volMax=0.0, volAct=1, maxH=0, dayLoss=0.0, ddBrk=0.0, pauseB=0):
    e = ov6(o, h, l, c, posA, bEp, bXp, bXd, R10, atrp, volMax, volAct, maxH, dayLoss, ddBrk, pauseB, day, TICK); r = P.pmetrics(e, lab); rows.append(r); return r
go('V2.9')
for v_, act in itertools.product(qs, (1, 2)): go(f'oynaklık ATR% > {v_:.3f}: ' + ('yeni giriş yok' if act == 1 else 'pozisyon da kapat'), volMax=v_, volAct=act)
for mh in (60, 120, 240, 480, 960): go(f'azami tutma {mh} bar (aynı yönde yeni sinyale kadar flat)', maxH=mh)
for dl in (25, 50, 75, 100): go(f'günlük zarar sınırı {dl} $ (gün sonuna kadar flat)', dayLoss=dl)
for db, pb in itertools.product((75, 100, 150, 200), (1440, 4320)): go(f'özsermaye kesicisi {db} $ düşüş → {pb // 1440} gün flat', ddBrk=db, pauseB=pb)
for v_, mh, db in itertools.product(qs[1:3], (240, 480), (100, 150)): go(f'BİRLEŞİK ATR%>{v_:.3f} + tutma {mh} + kesici {db}$/1g', volMax=v_, volAct=1, maxH=mh, ddBrk=db, pauseB=1440)
R = pd.DataFrame(rows); R.to_pickle('/home/claude/zecwork/risk1.pkl'); pd.set_option('display.width', 330); pd.set_option('display.max_colwidth', 62)
c_ = ['model', 'SEÇİM ΔSC4', 'SEÇİM ΔMaxDD%', 'DOĞRULAMA ΔSC4', 'DOĞRULAMA ΔCalmar', 'DOĞRULAMA ΔMaxDD%', 'DOĞRULAMA ΔP99%', 'DOĞRULAMA çukur$', 'DOĞRULAMA çukur%+', 'DOĞRULAMA zayıfAy+', 'DOĞRULAMA korAy', 'net', 'MaxDD', 'MaxDDnom', 'std1s']
print(R[c_].round(2).to_string(index=False)); print(f'{time.time() - t0:.0f} sn')
