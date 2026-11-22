import numpy as np, pandas as pd, itertools
from ent import g_ent
from eng import load
from lib import simR, cut
from met import metrics, SPLIT
from scipy.ndimage import uniform_filter
D0 = load(5); A = dict(np.load('alt5.npz'))
def feats2(D, A):
    c = D['c']; t = D['t']; H = 288
    r = lambda a: (a / np.r_[np.full(H, np.nan), a[:-H]] - 1) * 100
    basket = np.nanmedian(np.stack([r(D['xc'])] + [r(v) for v in A.values()]), axis=0)
    day = t // 86400000; first = np.r_[True, day[1:] != day[:-1]]; dopen = pd.Series(np.where(first, D['o'], np.nan)).ffill().to_numpy()
    return dict(raw=r(c), idio=r(c) - basket, gün=(c / dopen - 1) * 100)
def tg(D, x, aL, aS, hold, sl):
    return g_ent(D['c'], np.nan_to_num(np.where(x <= -aL, 1, np.where(x >= aS, -1, 0))).astype(np.int64), hold, sl)
def tg2(D, raw, idio, aL, aS, k, hold, sl):  # ham 24s dönüş + idiyosenkratik pay şartı
    s = np.where((raw <= -aL) & (idio <= -k), 1, np.where((raw >= aS) & (idio >= k), -1, 0))
    return g_ent(D['c'], np.nan_to_num(s).astype(np.int64), hold, sl)
if __name__ == '__main__':
    k0 = np.searchsorted(D0['t'], SPLIT); FX = feats2(D0, A); DI = cut(D0, SPLIT); FI = {k: v[:k0] for k, v in FX.items()}
    G = dict(aL=[2, 2.5, 3, 3.5, 4, 5, 6], aS=[3, 4, 4.5, 5, 5.5, 6, 7], hold=[288, 432, 648], sl=[3, 8])
    res = {}
    for fam in ['raw', 'idio', 'gün']:
        Gf = dict(G)
        if fam == 'gün': Gf['hold'] = [144, 288, 432]
        Arr = np.zeros([len(v) for v in Gf.values()]); N = np.zeros_like(Arr)
        for idx in itertools.product(*[range(len(v)) for v in Gf.values()]):
            p = {k: Gf[k][i] for k, i in zip(Gf, idx)}; m = metrics(DI, simR(DI, tg(DI, FI[fam], **p))); Arr[idx] = m['net'] - 1.5 * m['p98']; N[idx] = m['net']
        Sm = uniform_filter(Arr, size=3, mode='nearest'); i2 = np.unravel_index(Sm.argmax(), Arr.shape); p = {k: Gf[k][j] for k, j in zip(Gf, i2)}
        mf = metrics(D0, simR(D0, tg(D0, FX[fam], **p))); res[fam] = (p, mf)
        print(f'{fam}: IS kârlı oran {(N > 0).mean():.2f} | plato merkezi {p} | IS net {N[i2]:.0f} skor {Arr[i2]:.0f} komşu {Sm[i2]:.0f}', flush=True)
        print('   TAM', {k: mf[k] for k in ['net', 'n', 'pf', 'sharpe', 'p98', 'mddc', 'IS', 'OOS', 'last12', 'posM']}, flush=True)
    # ham + idiyosenkratik şart (V1.0 parametreleri sabit, k ızgarası IS)
    for k in [0, 0.5, 1, 1.5, 2, 3]:
        mi = metrics(DI, simR(DI, tg2(DI, FI['raw'], FI['idio'], 3.5, 5.5, k, 648, 3))); mf = metrics(D0, simR(D0, tg2(D0, FX['raw'], FX['idio'], 3.5, 5.5, k, 648, 3)))
        print(f'V1.0 + idio≥{k}: IS {mi["net"]} P98 {mi["p98"]} skor {mi["net"]-1.5*mi["p98"]:.0f} | TAM {mf["net"]} n {mf["n"]} PF {mf["pf"]} Sharpe {mf["sharpe"]} P98 {mf["p98"]} OOS {mf["OOS"]} son12 {mf["last12"]}', flush=True)
