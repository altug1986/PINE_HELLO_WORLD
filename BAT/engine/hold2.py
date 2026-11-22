import numpy as np, pandas as pd, v55
from numba import njit
from wh8 import ST, A30, st5, D, SC, ADX60, AGREE, GLb, GSb, GL0, GS0, q0, TS, simR, run, evalp, m30
c = D['c']
@njit
def vote_h(sc, adx, ag, stH, stT, cl, aT, on, onT, onCS, offW, xl):     # tutma yalnız pozisyon zararı ≤ %xl iken
    m = len(sc); tg = np.zeros(m, np.int64); pos = 0; ep = 0.0
    for i in range(m):
        s = sc[i]; want = pos; pl = (cl[i] / ep - 1) * 100 * pos if pos != 0 else 0.0
        off = offW if (pos != 0 and stH[i] == pos and pl >= -xl) else 0.0
        if pos != 0 and s * pos < off: want = 0
        oL = onT if (ag[i] == -1 and adx[i] >= aT) else on
        oS = onT if (ag[i] == 1 and adx[i] >= aT) else on
        if stT[i] == 1: oS = max(oS, onCS)
        elif stT[i] == -1: oL = max(oL, onCS)
        if s >= oL: want = 1
        elif s <= -oS: want = -1
        if want != pos and want != 0: ep = cl[i]
        pos = want; tg[i] = pos
    return tg
def ev(stH, stT, xl=999.0, full=True, lab=''):
    v55.TG = vote_h(SC, ADX60, AGREE, stH, stT, c, 20.0, 0.04, 0.15, 0.12, -0.15, xl)
    gl, gs = (GLb, GSb) if full else (GL0, GS0); v55.GL = gl.astype(np.int64); v55.GS = gs.astype(np.int64)
    q = dict(q0, **TS); e = evalp(q); m = m30(simR(D, run(q)[0]))
    return dict(t=lab, net=e['net'], n=e['n'], IS=e['IS'], OOS=e['OOS'], L12=e['L12'], L200=e['L200'], DD=m['MaxDD 30dk mum içi'], Cal=m['Calmar 30dk'], P=m['DD P95/P98/P99 30dk'], SS=m['Sharpe/Sortino 30dk'])
if __name__ == '__main__':
    rows = []
    for full in (True, False):
        tag = 'tam balina' if full else 'TV benzeri'
        rows.append(ev(ST, ST, full=full, lab=f'V6.5 · {tag}'))
        for xl in (0.0, 1.0, 2.0, 3.0, 5.0): rows.append(ev(ST, ST, xl, full, f'tut yalnız zarar ≤ %{xl:g} · {tag}'))
        for a in (30, 35): s = st5(1.5, 10, 3.5); rows.append(ev(s * (A30 >= a), ST, full=full, lab=f'tut ADX ≥ {a} · {tag}'))
        s = st5(1.5, 10, 2.5); rows.append(ev(s * (A30 >= 25), s * (A30 >= 30), full=full, lab=f'seçenek B (2.5, 25/30) · {tag}'))
    r = pd.DataFrame(rows); pd.set_option('display.width', 250); print(r.to_string(index=False))
