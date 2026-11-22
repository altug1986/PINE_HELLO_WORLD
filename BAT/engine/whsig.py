import itertools, numpy as np, pandas as pd, v55
from numba import njit
from hold2 import vote_h, st5, A30, D, SC, ADX60, AGREE, GLb, GSb, q0, TS, simR, run, evalp, m30, c
from astre_test import P, A, hours, H0, HMS
from v55 import t
SPL = (pd.Timestamp('2024-11-01') - pd.Timestamp('1970-01-01')) // pd.Timedelta('1ms')
# saatlik seriler (karar saati H, H−1:55 kaydı), tekrar eden değer = NaN
tp = np.array([np.log(P[H - 300000]) if (H - 300000) in P else np.nan for H in hours]); ta = np.array([np.log(A[H - 300000]) if (H - 300000) in A else np.nan for H in hours])
dv = pd.Series(tp - ta); dv[dv == dv.ffill().shift(1)] = np.nan
tps = pd.Series(tp); tps[tps == tps.ffill().shift(1)] = np.nan
def zz(x, N=720): r = x.rolling(N, min_periods=N // 2); return (x - r.mean()) / r.std()
F = {'seviye z30': zz(dv), 'oran sıçrama': zz(dv - dv.ffill().shift(1)), 'pozisyon sıçrama': zz(tps - tps.ffill().shift(1))}
j = (t // HMS * HMS - H0) // HMS
F = {k: np.nan_to_num(v.to_numpy()[j]) for k, v in F.items()}
@njit
def ent(s, cl, hold, sl):
    m = len(s); out = np.zeros(m, np.int64); v = 0; e = 0.0; vb = 0
    for i in range(m):
        if v != 0 and ((cl[i] / e - 1) * v <= -sl / 100 or i - vb >= hold): v = 0
        if s[i] != 0:
            if v != s[i]: v = s[i]; e = cl[i]; vb = i
            else: vb = i
        out[i] = v
    return out
S = st5(1.5, 10, 2.5); SH = S * (A30 >= 25); STT = S * (A30 >= 30); DEN = 24.5
def ev(p16=None, w=0.0, lab=''):
    sc = globals()["SC"] if p16 is None else (SC * DEN + w * p16) / (DEN + w)
    v55.TG = vote_h(sc, ADX60, AGREE, SH, STT, c, 20.0, 0.04, 0.15, 0.12, -0.15, 999.0)
    v55.GL = GLb.astype(np.int64); v55.GS = GSb.astype(np.int64); q = dict(q0, **TS); e = evalp(q); R = simR(D, run(q)[0]); m = m30(R)
    return dict(t=lab, net=e['net'], n=e['n'], IS=e['IS'], OOS=e['OOS'], L12=e['L12'], L200=e['L200'], DD=m['MaxDD 30dk mum içi'], Cal=m['Calmar 30dk'], P=m['DD P95/P98/P99 30dk'], SS=m['Sharpe/Sortino 30dk'])
if __name__ == '__main__':
    rows = [ev(lab='V6.6')]; print(rows[0])
    for (fn, x), k, yon, hold, w in itertools.product(F.items(), (2.0, 2.5, 3.0), ('ters', 'aynı'), (48, 144, 288), (1.0, 2.0, 4.0)):
        sg = np.where(x >= k, 1, np.where(x <= -k, -1, 0)) * (-1 if yon == 'ters' else 1)
        rows.append(dict(ev(ent(sg.astype(np.int64), c, hold, 10.0), w, f'{fn} {yon}'), k=k, hold=hold, w=w, olay=int(((sg != 0) & (np.r_[0, sg[:-1]] == 0)).sum())))
    r = pd.DataFrame(rows); r.to_pickle('whsig.pkl'); b = r.iloc[0]; pd.set_option('display.width', 250)
    g = r.iloc[1:].groupby('t').agg(ort=('net', 'mean'), max=('net', 'max'), IS_ort=('IS', 'mean'), OOS_ort=('OOS', 'mean'), DD_ort=('DD', 'mean'), baz_ustu=('net', lambda v: int((v > b.net).sum())))
    print(g.round(0).to_string())
    print(r.iloc[1:].sort_values('IS', ascending=False).head(10)[['t', 'k', 'hold', 'w', 'olay', 'net', 'n', 'IS', 'OOS', 'L12', 'L200', 'DD', 'Cal', 'P', 'SS']].to_string(index=False))
