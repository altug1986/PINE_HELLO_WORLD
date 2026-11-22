import numpy as np, pandas as pd, itertools, time, json
from orch2 import *
from ent import feats
def inst_set(D, F, names):
    c = D['c']; lib = {}
    for nm, u, d in INST: lib[f'TAKİP {u}/{d}'] = lambda u=u, d=d: g_rev(c, u, d)
    lib['DÖNÜŞ z4s'] = lambda: g_osc(c, zsc(c, 48), -3.5, 3.5, 0.5, -0.5, 8, 288)
    lib['DÖNÜŞ 4s'] = lambda: fade(c, 48, 6, 48, 10)
    lib['DÖNÜŞ 8s'] = lambda: fade(c, 96, 6, 192, 10)
    lib['DÖNÜŞ 24s'] = lambda: mot2(D, F, dict(fam='PXA', H=288, aL=3.5, aS=5.5, hold=648, sl=3))
    lib['DÖNÜŞ 2g'] = lambda: fade(c, 576, 9, 1152, 10)
    lib['DÖNÜŞ 3g'] = lambda: fade(c, 864, 12, 1728, 10)
    return np.stack([lib[n]() for n in names])
BASE8 = ['TAKİP 2/2', 'TAKİP 3/3', 'TAKİP 5/5', 'TAKİP 8/8', 'DÖNÜŞ z4s', 'DÖNÜŞ 4s', 'DÖNÜŞ 24s', 'DÖNÜŞ 3g']
SETS = {'8 (baz)': BASE8, '−4s': [x for x in BASE8 if x != 'DÖNÜŞ 4s'], '−3g': [x for x in BASE8 if x != 'DÖNÜŞ 3g'],
        '−4s −3g': [x for x in BASE8 if x not in ('DÖNÜŞ 4s', 'DÖNÜŞ 3g')], '4s→8s 3g→2g': [x.replace('4s', '8s').replace('3g', '2g') if x in ('DÖNÜŞ 4s', 'DÖNÜŞ 3g') else x for x in BASE8]}
def run_set(D, S, names, floor, on, offr, k, pw, minc=6, SC0=None):
    SC = (scores(D, S, 0, minc, WTS['eşit']) if SC0 is None else SC0).copy()
    if floor > 0: j = names.index('DÖNÜŞ 24s'); SC[j] = np.maximum(SC[j], floor)
    return simR(D, conductor(S, SC, on, offr * on, k, pw)), SC
if __name__ == '__main__':
    D0 = load(5); DI = cut(D0, SPLIT); FI = feats(DI)
    # 1) IS katkı (baz ayar)
    S = inst_set(DI, FI, BASE8); R, SC = run_set(DI, S, BASE8, 0, 0.10, 0.0, 6, 1)
    who = [BASE8[int(np.argmax(np.where(S[:, i] == dd, SC[:, i], -9)))] for i, dd in zip(R['eb'], R['dir'])]
    print('IS katkı:', pd.Series(R['pnl']).groupby(who).sum().round().to_dict(), flush=True)
    rows = []; t0 = time.time()
    for sn, names in SETS.items():
        S = inst_set(DI, FI, names); SC0 = scores(DI, S, 0, 6, WTS['eşit'])
        for floor, on, offr, k, pw in itertools.product([0, 0.1, 0.2, 0.3], [0.05, 0.1, 0.15, 0.2], [0, 0.5], [3, 6, 12], [1, 2]):
            R, _ = run_set(DI, S, names, floor, on, offr, k, pw, SC0=SC0); m = metrics(DI, R)
            rows.append(dict(set=sn, floor=floor, on=on, offr=offr, k=k, pw=pw, **{kk: m[kk] for kk in ['net', 'n', 'pf', 'sharpe', 'p98', 'mddc', 'upi', 'posM']}))
        print(sn, f'{time.time()-t0:.0f}s', flush=True)
    df = pd.DataFrame(rows); df['sc'] = df.net - 1.5 * df.p98; df.to_pickle('orch3_IS.pkl'); ok = df[(df.n >= 1300) & (df.n <= 1800)]
    pd.set_option('display.width', 250)
    print(ok.groupby('set').apply(lambda g: pd.Series(dict(ayar=len(g), kârlı=(g.net > 0).mean().round(2), en_iyi_sc=g.sc.max(), medyan_net=g.net.median())), include_groups=False).to_string())
    print(ok.groupby('floor').sc.median().round().to_dict())
    print(ok.sort_values('sc', ascending=False).head(8).to_string(index=False))
