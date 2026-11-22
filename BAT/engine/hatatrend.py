import itertools, sys, numpy as np, pandas as pd, v55
from numba import njit
from multiprocessing import Pool
from hold2 import st5, A30, D, SC, ADX60, AGREE, GLb, GSb, q0, TS, simR, run, evalp, m30, c
S25 = st5(1.5, 10, 2.5); SH = S25 * (A30 >= 25); STT = S25 * (A30 >= 30); Z = np.zeros(len(c), np.int64)
ZCA = {'yok': Z, 'ST90': S25, 'ST90+ADX': SH, '1S EMA': AGREE.astype(np.int64)}
LST = {(h, f): st5(h, 10, f) for h in (4, 6, 8, 12) for f in (2.0, 3.0, 4.0)}
@njit
def vote_x(sc, adx, ag, stH, stH2, stT, stF, aT, on, onT, onCS, offW):
    m = len(sc); tg = np.zeros(m, np.int64); pos = 0
    for i in range(m):
        s = sc[i]; want = pos; off = offW if (pos != 0 and (stH[i] == pos or stH2[i] == pos)) else 0.0
        if pos != 0 and s * pos < off: want = 0
        oL = onT if (ag[i] == -1 and adx[i] >= aT) else on
        oS = onT if (ag[i] == 1 and adx[i] >= aT) else on
        if stT[i] == 1: oS = max(oS, onCS)
        elif stT[i] == -1: oL = max(oL, onCS)
        if s >= oL: want = 1
        elif s <= -oS: want = -1
        if want == 0 and stF[i] != 0: want = stF[i]
        pos = want; tg[i] = pos
    return tg
def ev(lab, stH2=Z, stT=None, stF=Z, zc='yok', **lay):
    v55.TG = vote_x(SC, ADX60, AGREE, SH, stH2, STT if stT is None else stT, stF, 20.0, 0.04, 0.15, 0.12, -0.15)
    v55.GL = GLb.astype(np.int64); v55.GS = GSb.astype(np.int64); v55.ZC = ZCA[zc]; q = dict(q0, **TS, **lay); e = evalp(q); R = simR(D, run(q)[0]); m = m30(R)
    return dict(t=lab, net=e['net'], n=e['n'], IS=e['IS'], OOS=e['OOS'], L12=e['L12'], L200=e['L200'], DD=m['MaxDD 30dk mum içi'], Cal=m['Calmar 30dk'], P=m['DD P95/P98/P99 30dk'], SS=m['Sharpe/Sortino 30dk'])
def job(a):
    kind, kw = a
    if kind == 'zon': lab = 'HATA→TERS pencere'; o = ev(lab, zc=kw['zc'], zAct=2, zK1=kw['k1'], zK2=kw['k2'], zX=kw['x'], zM=kw['m'], zC=int(kw['zc'] != 'yok'), cS=kw['cS'], N=kw['N'])
    elif kind == 'L': lab = 'HATA→TERS zarar+BTC'; o = ev(lab, L=kw['L'], a=kw['a'], vk=0.0, cS=kw['cS'], N=kw['N'])
    else:
        s = LST[(kw['h'], kw['f'])]; sa = s * (A30 >= 25)
        lab = {'tut': 'TREND tut', 'tut+ADX': 'TREND tut (ADX≥25)', 'ters eşik': 'TREND ters eşik', 'boşta gir': 'TREND boşta gir', 'boşta gir+ADX': 'TREND boşta gir (ADX≥25)'}[kw['mod']]
        o = ev(lab, stH2=s if kw['mod'] == 'tut' else (sa if kw['mod'] == 'tut+ADX' else Z), stT=s if kw['mod'] == 'ters eşik' else None,
               stF=s if kw['mod'] == 'boşta gir' else (sa if kw['mod'] == 'boşta gir+ADX' else Z))
    o.update(kw); return o
if __name__ == '__main__':
    J = [('zon', dict(k1=12, k2=k2, x=x, m=m, zc=zc, cS=cS, N=N)) for k2, x, m, zc, cS, N in itertools.product((48, 144, 288), (2.0, 3.0, 4.0, 5.0), (1.0, 99.0), tuple(ZCA), (3.0, 5.0), (144, 288, 576))]
    J += [('L', dict(L=L, a=a, cS=cS, N=N)) for L, a, cS, N in itertools.product((3.0, 4.0, 5.0, 6.0), (0.0, 20.0, 30.0), (3.0, 5.0), (288, 576))]
    J += [('tr', dict(h=h, f=f, mod=md)) for (h, f), md in itertools.product(LST, ('tut', 'tut+ADX', 'ters eşik', 'boşta gir', 'boşta gir+ADX'))]
    base = ev('V6.6'); print(base, flush=True)
    with Pool(2) as p: rows = p.map(job, J, chunksize=8)
    r = pd.DataFrame([base] + rows); r.to_pickle('hatatrend.pkl'); b = r.iloc[0]; pd.set_option('display.width', 260)
    g = r.iloc[1:].groupby('t').agg(test=('net', 'size'), ort=('net', 'mean'), max=('net', 'max'), IS_ort=('IS', 'mean'), OOS_ort=('OOS', 'mean'), DD_ort=('DD', 'mean'), baz_ustu=('net', lambda v: int((v > b.net).sum())), yuzde6=('net', lambda v: int((v > b.net * 1.06).sum())))
    print(g.round(0).to_string())
    for tk in g.index:
        x = r[r.t == tk].sort_values('IS', ascending=False).head(3); print('--', tk); print(x.drop(columns=['t']).dropna(axis=1, how='all').to_string(index=False))
