import itertools, numpy as np, pandas as pd, v55
from collections import deque
from stlib import st5, D, vote_st2, SC, ADX60, AGREE, GLb, GSb, GL0, GS0, S1, q0, TS, simR, run, evalp, t, m30
from astre_test import P, A, hours, H0, HMS
def whz(N, mn, sh=-300000):                                  # Astre mantığı: tekrar atla, takvim penceresi N saat, en az mn değer
    V, T = deque(), deque(); last = None; z = np.full(len(hours), np.nan)
    for k, H in enumerate(hours):
        p, a = P.get(H + sh), A.get(H + sh)
        while T and T[0] <= H - N * HMS: T.popleft(); V.popleft()
        if p is None or a is None: continue
        dv = np.log(p) - np.log(a)
        if last is not None and dv == last: continue
        last = dv; V.append(dv); T.append(H)
        if len(V) >= mn:
            x = np.fromiter(V, float); sd = x.std(ddof=1); z[k] = (dv - x.mean()) / sd if sd > 0 else np.nan
    return z[(t // HMS * HMS - H0) // HMS]
A30 = np.nan_to_num(D['adx30']); ST = st5(1.5, 10, 3.5) * (A30 >= 25)
def ev(xL=None, xS=None, veto=None, full=True, lab=''):
    stH = ST.copy()
    if veto is not None: stH[veto] = 0
    v55.TG = vote_st2(SC, ADX60, AGREE, stH, ST, 20.0, 0.04, 0.15, 0.12, -0.15)
    gl, gs = (GLb, GSb) if full else (GL0, GS0)
    if xL is not None: gl = gl | (xL if full else xL & S1); gs = gs | (xS if full else xS & S1)
    v55.GL = gl.astype(np.int64); v55.GS = gs.astype(np.int64); q = dict(q0, **TS); e = evalp(q); m = m30(simR(D, run(q)[0]))
    return dict(t=lab, net=e['net'], n=e['n'], IS=e['IS'], OOS=e['OOS'], L12=e['L12'], L200=e['L200'], DD=m['MaxDD 30dk mum içi'], Cal=m['Calmar 30dk'], P=m['DD P95/P98/P99 30dk'], SS=m['Sharpe/Sortino 30dk'])
if __name__ == '__main__':
    Z8 = np.nan_to_num(whz(8, 4)); rows = [ev(lab='V6.5 tam balina'), ev(full=False, lab='V6.5 TV benzeri')]
    print('8s z dolu bar %', round((Z8 != 0).mean() * 100, 1))
    for k in (1.0, 1.25, 1.5, 1.75, 2.0):
        rows.append(dict(ev(Z8 >= k, Z8 <= -k, lab=f'kapı ters yön z8≥{k}'), k=k))         # balina long'a yığıldı → long yok
        rows.append(dict(ev(Z8 <= -k, Z8 >= k, lab=f'kapı aynı yön z8≥{k}'), k=k))         # balina long'a yığıldı → short yok
    for k in (1.0, 1.5, 2.0):
        rows.append(dict(ev(veto=((ST == 1) & (Z8 >= k)) | ((ST == -1) & (Z8 <= -k)), lab=f'tutma iptali ters z8≥{k}'), k=k))
        rows.append(dict(ev(veto=((ST == 1) & (Z8 <= -k)) | ((ST == -1) & (Z8 >= k)), lab=f'tutma iptali aynı z8≥{k}'), k=k))
    r = pd.DataFrame(rows); r.to_pickle('wh8.pkl'); pd.set_option('display.width', 250); print(r.to_string(index=False))
