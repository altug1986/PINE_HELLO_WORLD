import numpy as np, numba as nb, pandas as pd, itertools
from zz import g_rev
from eng import load, TICK, CASH, COMM
from lib import simR
from met import metrics, SPLIT
from cp import mae_of

@nb.njit(cache=True)
def pf_gate(t, c, sh, kind, N, thr, inv, tick):
    # gölge (3/3) işlemleri → kayan PF kapısı; kapı gölgenin her dönüşünde değerlendirilir
    # kind 0: son N gölge işlem PF | 1: aynı yön son N işlem PF | 2: son N gün gölge PF
    n = len(c); real = np.zeros(n, np.int64); m = n
    pl = np.zeros(m); pd_ = np.zeros(m, np.int64); pt = np.zeros(m, np.int64); k = 0; ep = 0.0; prev = 0; r = 0
    for i in range(n):
        s = sh[i]
        if s != prev:
            if prev != 0:
                x = c[i] - tick * prev; q = CASH / ep
                pl[k] = (x - ep) * q * prev - COMM * q * (ep + x); pd_[k] = prev; pt[k] = t[i]; k += 1
            ep = c[i] + tick * s; prev = s
            gp = 0.0; gl = 0.0; cnt = 0; j = k - 1
            while j >= 0:
                if kind == 2:
                    if t[i] - pt[j] > N * 86400000: break
                elif cnt >= N: break
                if kind == 1 and pd_[j] != s:
                    j -= 1; continue
                if pl[j] > 0: gp += pl[j]
                else: gl -= pl[j]
                cnt += 1; j -= 1
            if cnt < 3: ok = True
            else:
                pf = gp / gl if gl > 0 else 99.0
                ok = pf >= thr if inv == 0 else pf < thr
            r = s if ok else 0
        real[i] = r
    return real

if __name__ == '__main__':
    D = load(5); c = D['c']; t = D['t']
    # 1) ön test: ardışık 20-işlem blok PF korelasyonu
    for u in [3, 4, 5]:
        p = simR(D, g_rev(c, u, u))['pnl']; b = [p[i:i + 20] for i in range(0, len(p) - 19, 20)]
        pf = np.array([x[x > 0].sum() / max(-x[x < 0].sum(), 1e-9) for x in b]); print(f'{u}/{u} blok PF korelasyonu', round(np.corrcoef(pf[:-1], pf[1:])[0, 1], 3))
    V = []
    for kind, N, thr, inv in [(0, N, th, 0) for N in [10, 20, 30, 50, 100] for th in [0.8, 1.0, 1.2, 1.5]] + [(1, N, th, 0) for N in [10, 20, 50] for th in [1.0, 1.2, 1.5]] + \
                             [(2, N, th, 0) for N in [3, 7, 14, 30] for th in [1.0, 1.2, 1.5]] + [(0, N, th, 1) for N in [20, 50] for th in [0.8, 1.0]]:
        V.append(dict(kind=['son-N-işlem', 'aynı-yön-N', 'son-N-gün'][kind] + ('-TERS' if inv else ''), N=N, thr=thr, _k=kind, _i=inv))
    rows = []
    for u in [3, 4, 5]:
        sh = g_rev(c, u, u); b = metrics(D, simR(D, sh)); rows.append(dict(eşik=u, kapı='YOK (baz)', N=0, thr=0, **{k: b[k] for k in ['net', 'n', 'pf', 'sharpe', 'p98', 'mddc', 'IS', 'OOS', 'last12']}))
        for v in V:
            R = simR(D, pf_gate(t, c, sh, v['_k'], v['N'], v['thr'], v['_i'], TICK)); m = metrics(D, R)
            rows.append(dict(eşik=u, kapı=v['kind'], N=v['N'], thr=v['thr'], **{k: m[k] for k in ['net', 'n', 'pf', 'sharpe', 'p98', 'mddc', 'IS', 'OOS', 'last12']}))
    df = pd.DataFrame(rows); df['skorIS'] = df.IS - 1.5 * df.p98; df.to_pickle('pff.pkl'); pd.set_option('display.width', 250)
    for u in [3, 4, 5]:
        g = df[df.eşik == u]; print(f'== {u}/{u}  IS>0 oranı {(g.IS > 0).mean():.2f} | OOS>0 {(g.OOS > 0).mean():.2f}')
        print(pd.concat([g[g.kapı.str.startswith('YOK')], g[~g.kapı.str.startswith('YOK')].sort_values('IS', ascending=False).head(6)]).drop(columns=['eşik', 'skorIS']).to_string(index=False))
