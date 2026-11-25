# PF ve SUPER-CALMAR filtreleri: V2.9 sanal (filtresiz) işlem/MTM akışından kayan PF / Calmar / Super-Calmar hesapla, kötüyken GERÇEK emri kes.
import sys; sys.path.insert(0, '/home/claude/zecwork'); sys.argv = ['x', 'yok']
import pfsearch as P; import numpy as np, pandas as pd, itertools, time
from numba import njit
from pfsearch import o, h, l, c, posA, bXp, bXd, R10, tms, n, TICK, ix, S
from sc4 import sc4
Z = np.load('/home/claude/zecwork/base28.npz'); mid = int(Z['mid']); TB = Z['T']; T0 = time.time()
@njit(cache=False)
def run29(o, h, l, c, posA, bXp, bXd, R10, tick, vpos, vxp, vnew, TR):
    n = len(c); pos = 0; ep = 0.0; q = 0.0; real = 0.0; mode = 0; prevD = 0; tPx = 0.0; tBar = 0; tSide = 0; rStp = 0.0; eb = 0; k = 0; eq = np.zeros(n)
    for i in range(1, n):
        if pos != 0:
            xp = -1.0
            if mode == 2 and pos == -tSide:
                if (pos == 1 and l[i] <= rStp) or (pos == -1 and h[i] >= rStp): xp = (min(o[i], rStp) - tick) if pos == 1 else (max(o[i], rStp) + tick); mode = 0
            elif mode == 0 and not np.isnan(bXp[i]) and pos == bXd[i]: xp = bXp[i]
            if xp > 0:
                pn = q * (xp - ep) * pos - 0.0004 * q * (ep + xp); real += pn; TR[k, 0] = eb; TR[k, 1] = i; TR[k, 2] = pos; TR[k, 3] = pn; k += 1; pos = 0; vxp[i] = xp
        D = posA[i]; want = D
        if D != prevD: mode = 0
        elif D != 0:
            if mode == 0:
                if (D > 0 and R10[i] >= 94.0) or (D < 0 and R10[i] <= 6.0): want = -D; mode = 2; tPx = c[i]; tBar = i; tSide = D
            else:
                bi = (R10[i] <= 70 or i - tBar >= 240 or c[i] >= tPx * 1.03) if tSide > 0 else (R10[i] >= 30 or i - tBar >= 240 or c[i] <= tPx * 0.97)
                if bi: mode = 0
                else: want = -tSide if pos == -tSide else 0
        prevD = D
        if want != pos:
            if pos != 0:
                xp = c[i] - tick * pos; pn = q * (xp - ep) * pos - 0.0004 * q * (ep + xp); real += pn; TR[k, 0] = eb; TR[k, 1] = i; TR[k, 2] = pos; TR[k, 3] = pn; k += 1; pos = 0
            if want != 0:
                pos = want; ep = c[i] + tick * pos; q = 500.0 / c[i]; eb = i; vnew[i] = True
                if mode == 2 and want == -tSide: rStp = ep * (1.0 - want * 0.02)
        vpos[i] = pos; eq[i] = real + (q * (c[i] - ep) * pos - 0.0004 * q * ep if pos != 0 else 0.0)
    return eq, k
@njit(cache=False)
def follow(o, c, tick, vpos, vxp, vnew, F, md):
    """md 1: kötüyken kapat, iyileşince hemen katıl | 2: kötüyken kapat, iyileşince YENİ sanal girişi bekle | 3: açık işleme dokunma, sadece yeni girişi kes"""
    n = len(c); real = 0; ep = 0.0; q = 0.0; rz = 0.0; wait = False; eq = np.zeros(n); nt = 0; gp = 0.0; gl = 0.0; inm = 0
    for i in range(1, n):
        if real != 0 and real == vpos[i - 1] and vxp[i] > 0:
            pn = q * (vxp[i] - ep) * real - 0.0004 * q * (ep + vxp[i]); rz += pn; nt += 1; gp += max(pn, 0.0); gl += max(-pn, 0.0); real = 0
        v = vpos[i]
        if md == 1: tg = v if F[i] else 0
        elif md == 2:
            if not F[i]: tg = 0; wait = True
            else:
                if wait and vnew[i]: wait = False
                tg = 0 if wait else v
        else:
            if vnew[i]: tg = v if F[i] else 0
            else: tg = real if real == v else 0
        if tg != real:
            if real != 0:
                xp = c[i] - tick * real; pn = q * (xp - ep) * real - 0.0004 * q * (ep + xp); rz += pn; nt += 1; gp += max(pn, 0.0); gl += max(-pn, 0.0); real = 0
            if tg != 0: real = tg; ep = c[i] + tick * real; q = 500.0 / c[i]
        if real != 0: inm += 1
        eq[i] = rz + (q * (c[i] - ep) * real - 0.0004 * q * ep if real != 0 else 0.0)
    return eq, nt, gp / max(gl, 1e-9), inm / n
vpos = np.zeros(n, np.int8); vxp = np.zeros(n); vnew = np.zeros(n, np.bool_); TR = np.zeros((20000, 4))
eq0, K = run29(o, h, l, c, posA, bXp, bXd, R10, TICK, vpos, vxp, vnew, TR); TR = TR[:K]
e1, *_ = follow(o, c, TICK, vpos, vxp, vnew, np.ones(n, np.bool_), 1)
print(f'sanal V2.9 net {eq0[-1]:.1f} işlem {K} | filtresiz takip {e1[-1]:.1f} (eşit olmalı)')
xi = TR[:, 1].astype(np.int64); pnl = TR[:, 3]; ei = TR[:, 0].astype(np.int64)
own_bar = np.zeros(n, np.int8)
for r in TB: own_bar[int(r[0]):int(r[1])] = int(r[7])
own = np.where(TR[:, 2] != posA[ei], 6, own_bar[ei])                    # 6 = uç TERS işlemi
# --- 1) OTOKORELASYON (playbook §7-1): kötü blok → sonraki blok?
def pf(x): g = x[x > 0].sum(); z = -x[x < 0].sum(); return g / z if z > 0 else 9.0
print('\nOTOKORELASYON (ardışık bloklar):')
for B in (20, 50, 100):
    m = K // B; bp = np.array([pf(pnl[j * B:(j + 1) * B]) for j in range(m)]); bn = np.array([pnl[j * B:(j + 1) * B].sum() for j in range(m)])
    lo = bn[:-1] <= np.percentile(bn[:-1], 33); hi = bn[:-1] >= np.percentile(bn[:-1], 67)
    print(f'  {B} işlemlik blok ({m}): PF kor {np.corrcoef(bp[:-1], bp[1:])[0, 1]:+.2f} | net kor {np.corrcoef(bn[:-1], bn[1:])[0, 1]:+.2f} | kötü 1/3 sonrası ort {bn[1:][lo].mean():.0f} $, iyi 1/3 sonrası {bn[1:][hi].mean():.0f} $, genel {bn.mean():.0f} $')
e30 = eq0[::30]; tt = ix[::30]; mo = S(e30, tt).resample('ME').last().diff().dropna().values; wk = S(e30, tt).resample('W').last().diff().dropna().values
print(f'  ay net kor {np.corrcoef(mo[:-1], mo[1:])[0, 1]:+.2f} | hafta net kor {np.corrcoef(wk[:-1], wk[1:])[0, 1]:+.2f}')
# --- 2) FİLTRE SERİLERİ (hepsi nedensel: sadece o ana kadar kapanmış sanal işlem / geçmiş MTM)
bi_tr = np.searchsorted(xi, np.arange(n), side='right') - 1           # her barda son kapanmış sanal işlem
dts = pd.date_range(ix[0].normalize() + pd.Timedelta('1D'), ix[-1].normalize(), freq='D'); dbar = np.searchsorted(ix.values, dts.values)
bi_day = np.searchsorted(dbar, np.arange(n), side='right') - 1           # her barın günü (gün başı güncellenir)
def trF(val): return np.where(bi_tr >= 0, val[np.maximum(bi_tr, 0)], np.nan)
def dayF(val): return np.where(bi_day >= 0, val[np.maximum(bi_day, 0)], np.nan)
cs = np.r_[0, np.cumsum(np.maximum(pnl, 0))]; cl = np.r_[0, np.cumsum(np.maximum(-pnl, 0))]
MET = {}
for N in (10, 15, 20, 30, 50, 75, 100):
    j = np.arange(K); a = np.maximum(j - N + 1, 0); g = cs[j + 1] - cs[a]; z = cl[j + 1] - cl[a]; v = np.where(z > 0, g / np.maximum(z, 1e-9), 9.0); v[j < N - 1] = np.nan
    MET[f'PF son {N} işlem'] = trF(v)
xt = tms[xi]
for Dd in (7, 14, 21, 30, 45, 60, 90):
    t1 = (dts - pd.Timestamp('1970-01-01')) // pd.Timedelta('1ms'); k1 = np.searchsorted(xt, t1.values, side='left'); k0 = np.searchsorted(xt, t1.values - Dd * 86400000, side='left')
    g = cs[k1] - cs[k0]; z = cl[k1] - cl[k0]; v = np.where(z > 0, g / np.maximum(z, 1e-9), np.where(g > 0, 9.0, np.nan))
    MET[f'PF son {Dd} gün'] = dayF(v)
e30d = S(e30, tt); i30 = np.searchsorted(tt.values, dts.values)
for W in (7, 14, 30, 60, 90):
    w = W * 48; cal = np.full(len(dts), np.nan); scv = np.full(len(dts), np.nan)
    for j, b in enumerate(i30):
        if b - w < 0: continue
        x = e30[b - w:b]; dd = np.maximum.accumulate(x) - x; mdd = max(dd.max(), 5.0); net = (x[-1] - x[0]) * 365 / W
        cal[j] = net / mdd
        if W >= 14:
            r2 = np.corrcoef(np.arange(w), x)[0, 1] ** 2 if x.std() > 0 else 0.0
            scv[j] = net / (mdd + np.percentile(dd, 95)) * (r2 if net > 0 else 1.0)
    MET[f'Calmar son {W} gün'] = dayF(cal)
    if W >= 14: MET[f'Super-Calmar son {W} gün'] = dayF(scv)
print(f'filtre serileri hazır {time.time() - T0:.0f} sn')
TH = {'PF': (0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35, 1.5), 'Cal': (-1, 0, 0.5, 1, 2, 3, 5), 'Sup': (-0.5, 0, 0.25, 0.5, 1, 2)}
yrs = (tms[-1] - tms[0]) / (365.25 * 86400000); t23 = int(np.searchsorted(tms, (pd.Timestamp('2023-02-06') - pd.Timestamp('1970-01-01')) // pd.Timedelta('1ms'))); t300 = int(np.searchsorted(tms, tms[-1] - 300 * 86400000))
def ev(e, nt, pfr, inm, **kw):
    x = e[::30]; d = np.maximum.accumulate(x) - x
    return dict(**kw, net=e[-1], IS=e[mid], OOS=e[-1] - e[mid], yakın=e[-1] - e[t23], son300=e[-1] - e[t300], MaxDD=d.max(), DD95=np.percentile(d, 95), DD99=np.percentile(d, 99),
                Ulcer=np.sqrt(((d / 5) ** 2).mean()), Calmar=e[-1] / yrs / d.max(), işlem=nt, PF=pfr, piyasada=inm * 100)
rows = [ev(*follow(o, c, TICK, vpos, vxp, vnew, np.ones(n, np.bool_), 1), filtre='YOK (V2.9)', eşik=np.nan, yön='', mod=0)]
MODS = {1: 'kötüde kapat, iyide hemen katıl', 2: 'kötüde kapat, iyide yeni girişi bekle', 3: 'açığa dokunma, yeni girişi kes'}
for nm, v in MET.items():
    for th in TH['PF' if nm.startswith('PF') else 'Cal' if nm.startswith('Calmar') else 'Sup']:
        good = np.where(np.isnan(v), True, v >= th)
        for yn, F in (('iyiyken işlem', good), ('kötüyken işlem (ters)', np.where(np.isnan(v), True, v < th))):
            for md in (1, 2, 3): rows.append(ev(*follow(o, c, TICK, vpos, vxp, vnew, F, md), filtre=nm, eşik=th, yön=yn, mod=md))
# motor bazlı PF kapısı: her motorun (E0..E3, POMPA, uç TERS) kendi son N kapanmış sanal işleminin PF'i < eşik → o motorun yeni girişi gerçek emir olmaz
for N, th in itertools.product((5, 10, 20, 30), (0.7, 0.9, 1.0, 1.2)):
    gate = np.ones(n, np.bool_); gi = np.ones(n, np.bool_)
    for u in np.unique(own):
        ku = np.flatnonzero(own == u); pu = pnl[ku]; xu = xi[ku]; eu = ei[ku]
        for jj, kk in enumerate(ku):
            prev = np.flatnonzero(xu[:jj] <= eu[jj])[-N:]
            if len(prev) == N: g_ = pf(pu[prev]) >= th; gate[eu[jj]] = g_; gi[eu[jj]] = not g_
    for yn, F in (('iyiyken işlem', gate), ('kötüyken işlem (ters)', gi)):
        rows.append(ev(*follow(o, c, TICK, vpos, vxp, vnew, F, 3), filtre=f'motor PF son {N} işlem', eşik=th, yön=yn, mod=3))
R = pd.DataFrame(rows); b = R.iloc[0]
for k in ('net', 'IS', 'OOS', 'MaxDD', 'DD99', 'Ulcer', 'Calmar'): R['d' + k] = R[k] - b[k]
R['modAd'] = R['mod'].map(MODS); R.to_pickle('/home/claude/zecwork/pffilt.pkl'); np.save('/home/claude/zecwork/pff_vpos.npy', vpos)
print(f'{len(R) - 1} varyant {time.time() - T0:.0f} sn')
