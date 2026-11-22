import numpy as np, numba as nb, pandas as pd, itertools
from eng import load, TICK
from lib import simR
from met import metrics, SPLIT
@nb.njit(cache=True)
def g_rev(c, up, dn):
    # % dönüş takipçisi: dipten up% yükselince LONG, tepeden dn% düşünce SHORT (her zaman pozisyonda)
    n = len(c); tg = np.zeros(n, np.int64); v = 0; hi = c[0]; lo = c[0]
    for i in range(n):
        x = c[i]
        if v >= 0:
            hi = max(hi, x)
            if x <= hi * (1 - dn / 100): v = -1; lo = x
        if v <= 0:
            lo = min(lo, x)
            if x >= lo * (1 + up / 100): v = 1; hi = x
        tg[i] = v
    return tg
if __name__ == '__main__':
    D = load(5); c = D['c']; t = D['t']; blk = ((pd.to_datetime(t, unit='ms').year - 2020) * 2 + (pd.to_datetime(t, unit='ms').month - 1) // 6).to_numpy()
    G = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]; rows = []
    for u, d in itertools.product(G, G):
        R = simR(D, g_rev(c, u, d)); p = R['pnl']; xt = t[R['xb']]
        rows.append(dict(up=u, dn=d, net=round(p.sum()), n=len(p), pf=round(p[p > 0].sum() / -p[p < 0].sum(), 3), IS=round(p[xt < SPLIT].sum()), OOS=round(p[xt >= SPLIT].sum()),
                         pb=np.bincount(blk[R['xb']], weights=p, minlength=blk.max() + 1)))
    df = pd.DataFrame(rows); df.to_pickle('zz.pkl')
    print('tam pozitif', (df.net > 0).mean().round(2), '| IS poz', (df.IS > 0).mean().round(2), '| OOS poz', (df.OOS > 0).mean().round(2))
    print(df.pivot(index='up', columns='dn', values='net').to_string())
    print(df.pivot(index='up', columns='dn', values='n').to_string())
    PB = np.stack(df.pb.values); wf = [PB[PB[:, :b].sum(1).argmax(), b] for b in range(2, PB.shape[1])]; print('yürüyen pencere net', round(sum(wf)), 'poz blok', sum(np.array(wf) > 0), '/', len(wf))
    print(df.sort_values('IS', ascending=False).head(6).drop(columns='pb').to_string(index=False))
