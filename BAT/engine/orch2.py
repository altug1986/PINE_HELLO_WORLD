import numpy as np, pandas as pd, itertools, time
from orch import *
from ent import g_ent
from lib import g_osc, zsc
from sysB import mot2
def fade(c, H, a, hold, sl):
    dP = (c / np.r_[np.full(H, np.nan), c[:-H]] - 1) * 100
    return g_ent(c, np.nan_to_num(np.where(dP <= -a, 1, np.where(dP >= a, -1, 0))).astype(np.int64), hold, sl)
def instruments(D, F=None):
    c = D['c']
    T = [g_rev(c, u, d) for _, u, d in INST]
    from ent import feats
    F = feats(D) if F is None else F
    M = [g_osc(c, zsc(c, 48), -3.5, 3.5, 0.5, -0.5, 8, 288), fade(c, 48, 6, 48, 10),
         mot2(D, F, dict(fam='PXA', H=288, aL=3.5, aS=5.5, hold=648, sl=3)), fade(c, 864, 12, 1728, 10)]
    return np.stack(T + M)
NAMES = ['TAKİP en kısa 2/2', 'TAKİP kısa 3/3', 'TAKİP orta 5/5', 'TAKİP uzun 8/8', 'DÖNÜŞ en kısa z4s', 'DÖNÜŞ kısa 4s', 'DÖNÜŞ orta 24s', 'DÖNÜŞ uzun 3g']
if __name__ == '__main__':
    D0 = load(5); DI = cut(D0, SPLIT); S_I = instruments(DI); rows = []; t0 = time.time()
    for j, nm in enumerate(NAMES):
        m = metrics(DI, simR(DI, S_I[j])); print('tek', nm, m['net'], m['n'], m['pf'])
    for dirspec, minc, wn in itertools.product([0, 1], [3, 6], WTS):
        SC = scores(DI, S_I, dirspec, minc, WTS[wn])
        for on, r, k, pw in itertools.product([0.05, 0.1, 0.2, 0.3, 0.45], [0.0, 0.5], [1, 3, 6], [1, 2]):
            R = simR(DI, conductor(S_I, SC, on, r * on, k, pw)); m = metrics(DI, R)
            rows.append(dict(dirspec=dirspec, minc=minc, wts=wn, on=on, offr=r, k=k, pw=pw, **{kk: m[kk] for kk in ['net', 'n', 'pf', 'sharpe', 'p98', 'mddc', 'upi', 'posM']}))
    df = pd.DataFrame(rows); df['sc'] = df.net - 1.5 * df.p98; df.to_pickle('orch2_IS.pkl'); print(f'{time.time()-t0:.0f}s')
    ok = df[(df.n >= 1300) & (df.n <= 1800)]
    print('IS kârlı oran', (df.net > 0).mean().round(2), '| 1300–1800 IS işlem:', len(ok), 'kârlı', (ok.net > 0).mean().round(2))
    pd.set_option('display.width', 250); print(ok.sort_values('sc', ascending=False).head(8).to_string(index=False))
    print(df.sort_values('sc', ascending=False).head(4).to_string(index=False))
