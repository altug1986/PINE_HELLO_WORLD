# TRIM sırtı: uçta düz kâr al (flat) vs kısa ters dönüş (flip); RSI10/94, RSI14/90, RSI21/86-88 bölgesi
import sys, time, itertools, numpy as np, pandas as pd, numba
src = open('/home/claude/zecwork/overlay2.py').read().split("F0 = np.zeros(n, np.bool_); JOBS = []")[0].replace("W = int(sys.argv[1]); NW = int(sys.argv[2])", "").replace('cache=True', 'cache=False'); exec(src)
@numba.njit
def ov3(o, h, l, c, posA, bEp, bXp, bXd, trL, trS, riL, riS, tb, cont, shMax, trSh, flip, fSL, tick):
    n = len(c); pos = 0; ep = 0.0; q = 0.0; real = 0.0; mode = 0; prevD = 0; tPx = 0.0; tBar = 0; tSide = 0; nt = 0; st = 0.0
    eq = np.zeros(n); P = np.zeros(n)
    for i in range(1, n):
        D = posA[i]
        if not np.isnan(bXp[i]) and pos != 0 and pos == bXd[i] and mode == 0:
            xp = bXp[i]; pnl = q * (xp - ep) * pos - 0.0004 * q * (ep + xp); real += pnl; P[i] += pnl; pos = 0
        if mode == 2 and flip and pos != 0 and pos == -tSide and ((pos == 1 and l[i] <= st) or (pos == -1 and h[i] >= st)):   # ters pozisyon stopu
            xp = (min(o[i], st) - tick) if pos == 1 else (max(o[i], st) + tick); pnl = q * (xp - ep) * pos - 0.0004 * q * (ep + xp); real += pnl; P[i] += pnl; pos = 0; mode = 0
        want = D
        if D != prevD: mode = 0
        elif D != 0:
            if mode == 0:
                if ((D > 0 and trL[i]) or (D < 0 and trS[i])) and trSh[i] * D < shMax:
                    want = -D if flip else 0; mode = 2; tPx = c[i]; tBar = i; tSide = D; nt += 1
            else:
                bi = (riL[i] or i - tBar >= tb or c[i] >= tPx * (1.0 + cont)) if tSide > 0 else (riS[i] or i - tBar >= tb or c[i] <= tPx * (1.0 - cont))
                if bi: mode = 0
                else: want = -tSide if (flip and pos != 0 or (flip and pos == 0 and False)) else (0 if not flip else (-tSide if pos == -tSide else 0))
        prevD = D
        if want != pos:
            if pos != 0:
                xp = c[i] - tick * pos; pnl = q * (xp - ep) * pos - 0.0004 * q * (ep + xp); real += pnl; P[i] += pnl; pos = 0
            if want != 0:
                pos = want; ep = c[i] + tick * pos; q = 500.0 / c[i]; st = ep * (1.0 - pos * fSL)
        eq[i] = real + (q * (c[i] - ep) * pos - 0.0004 * q * ep if pos != 0 else 0.0)
    return eq, P, nt
def M(lab, eq, P, nt):
    e30 = eq[::30]; dd = np.maximum.accumulate(e30) - e30; dl = P - base_c; d = np.diff(e30[::48])
    r = dict(v=lab, net=eq[-1], nTrim=nt, dIS=dl[:mid].sum(), dOOS=dl[mid:].sum(), dCalm=dl[isCalm].sum(), d300=dl[tms >= t300].sum(), MaxDD=dd.max(), DD99=np.percentile(dd, 99), DD95=np.percentile(dd, 95), Sh=d.mean() / d.std() * np.sqrt(365))
    for y in (2021, 2022, 2023, 2024, 2025, 2026): r[f'd{y}'] = dl[yr == y].sum()
    return r
F0 = np.zeros(n, np.bool_); RS = {N: np.nan_to_num(rsi(c, N), nan=50.0) for N in (10, 14, 21)}
eq, P, nt = ov3(o, h, l, c, posA, bEp, bXp, bXd, F0, F0, F0, F0, 60, 0.03, 99.0, trSh, False, 0.02, TICK); C = M('kontrol', eq, P, nt); rows = [C]
for (N, X), Y, tb, cont, sh, fl, fsl in itertools.product(((10, 94), (14, 90), (21, 86), (21, 88)), (50, 60, 70), (60, 240), (0.03, 0.05), (2.0, 99.0), (False, True), (0.01, 0.02)):
    if not fl and fsl == 0.02: continue
    R_ = RS[N]; eq, P, nt = ov3(o, h, l, c, posA, bEp, bXp, bXd, R_ >= X, R_ <= 100 - X, R_ <= Y, R_ >= 100 - Y, tb, cont, sh, trSh, fl, fsl, TICK)
    rows.append(M(f'RSI{N} X{X} Y{Y} geri{tb} devam%{cont * 100:g} sh<{sh} ' + (f'TERS sl%{fsl * 100:g}' if fl else 'FLAT'), eq, P, nt))
R = pd.DataFrame(rows)
for k in ('dIS', 'dOOS', 'dCalm', 'd300') + tuple(f'd{y}' for y in range(2021, 2027)): R[k] = R[k] - C[k]
R['dNet'] = R.net - C['net']; R.to_pickle('/home/claude/zecwork/ov3.pkl'); pd.set_option('display.width', 330); pd.set_option('display.max_colwidth', 64)
R['tip'] = np.where(R.v.str.contains('TERS'), 'TERS', 'FLAT')
print('kontrol: net', round(C['net'], 1), 'MaxDD', round(C['MaxDD'], 1), 'DD99', round(C['DD99'], 1), 'DD95', round(C['DD95'], 1), 'Sharpe', round(C['Sh'], 2))
print(R.iloc[1:].groupby('tip').agg(var=('v', 'size'), artı=('dNet', lambda s: (s > 0).mean()), ikiYarı=('dIS', lambda s: 0), medNet=('dNet', 'median'), medDD=('MaxDD', 'median')).round(2).to_string())
print('iki yarı artı:', R.iloc[1:].assign(b=(R.dIS > 0) & (R.dOOS > 0)).groupby('tip').b.mean().round(2).to_dict())
cols = ['v', 'nTrim', 'dNet', 'dIS', 'dOOS', 'dCalm', 'd300', 'MaxDD', 'DD99', 'Sh', 'd2023', 'd2024', 'd2025', 'd2026']
print(R.iloc[1:].sort_values('dIS', ascending=False)[cols].head(14).round(1).to_string(index=False))
