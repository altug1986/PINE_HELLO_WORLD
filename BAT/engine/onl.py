import numpy as np, numba as nb, pandas as pd, itertools, time
from met import SPLIT
# ÇEVRİMİÇİ 4 BOYUTLU BAĞLAMSAL PF / BEKLENTİ FİLTRESİ (geleceği görmez)
# her 3/3 takip işleminin girişinde: hücre = yön × ekran(24s) × volatilite × 4. boyut
# hiyerarşik büzülme: yön → +ekran → +vol → +4. boyut ; zaman ağırlığı: üstel unutma (tau gün)
# karar: TAKİP (beklenti ≥ θ ve ≥ ters) | TERS | FLAT
X = pd.read_pickle('ctx_trades.pkl'); IS = (X.xt < SPLIT).to_numpy()
EDGE = dict(e24=[-5.8, -1.25, 1.45, 5.8], volr=[0.88, 1.34], saat=[4, 8, 12, 16, 20], e7g=[-15, -4.3, 4.3, 15], aralık=[0.33, 0.49, 0.69, 0.86], e4s=[-1, 0.9, 1.7, 2.5], yok=[])
def bins(f): return np.digitize(X[f].to_numpy(), EDGE[f]) if f != 'yok' else np.zeros(len(X), np.int64)

@nb.njit(cache=True)
def online(et, xt, keys, pn, tr, tau, k, th, inv):
    # keys: (n, L) hiyerarşik hücre anahtarları (0..K-1); seviye 0 = yön
    n, L = keys.shape; K = 4096
    Sf = np.zeros((L, K)); Si = np.zeros((L, K)); N = np.zeros((L, K)); TL = np.zeros((L, K))
    act = np.zeros(n, np.int64); order = np.argsort(xt); p = 0
    for j in range(n):
        while p < n and xt[order[p]] <= et[j]:          # girişten önce (ya da aynı barda) kapanan gölge işlemler
            q = order[p]
            for lv in range(L):
                kk = keys[q, lv]
                if tau > 0:
                    d = np.exp(-(xt[q] - TL[lv, kk]) / (tau * 86400000.0)); Sf[lv, kk] *= d; Si[lv, kk] *= d; N[lv, kk] *= d
                TL[lv, kk] = xt[q]; Sf[lv, kk] += pn[q]; Si[lv, kk] += tr[q]; N[lv, kk] += 1.0
            p += 1
        ef = 0.0; ei = 0.0
        for lv in range(L):
            kk = keys[j, lv]; d = 1.0
            if tau > 0: d = np.exp(-(et[j] - TL[lv, kk]) / (tau * 86400000.0))
            nn = N[lv, kk] * d
            ef = (Sf[lv, kk] * d + k * ef) / (nn + k); ei = (Si[lv, kk] * d + k * ei) / (nn + k)
        if ef >= th and ef >= ei: act[j] = 1
        elif inv == 1 and ei >= th: act[j] = -1
    return act

def keys_for(dims):
    L = []; key = np.zeros(len(X), np.int64)
    for i, d in enumerate(['yön'] + dims):
        b = (X.yön.to_numpy() > 0).astype(np.int64) if d == 'yön' else bins(d)
        key = key * 8 + b; L.append(key.copy())
    return np.stack(L, 1)
if __name__ == '__main__':
    et = X.eb.map(lambda i: i).to_numpy(); from eng import load; T = load(5)['t']; et = T[X.eb.to_numpy()].astype(np.float64); xt = X.xt.to_numpy().astype(np.float64)
    pn, tr = X.pnl.to_numpy(), X.ters.to_numpy(); rows = []; t0 = time.time()
    DIMS = {'2B: yön×ekran': ['e24'], '3B: +vol': ['e24', 'volr'], '4B: +saat': ['e24', 'volr', 'saat'], '4B: +7g': ['e24', 'volr', 'e7g'],
            '4B: +aralık': ['e24', 'volr', 'aralık'], '4B: +4s': ['e24', 'volr', 'e4s']}
    for dn, dims in DIMS.items():
        K_ = keys_for(dims)
        for tau, k, th, inv in itertools.product([0, 90, 180, 365, 730], [5, 20, 50, 100], [0, 0.5, 1, 2], [0, 1]):
            a = online(et, xt, K_, pn, tr, float(tau), float(k), th, inv); r = np.where(a == 1, pn, np.where(a == -1, tr, 0))
            rows.append(dict(boyut=dn, tau=tau, k=k, th=th, ters=inv, IS=r[IS].sum(), OOS=r[~IS].sum(), net=r.sum(), işlem=int((a != 0).sum()), takip=int((a == 1).sum()), tersn=int((a == -1).sum())))
    df = pd.DataFrame(rows); df.to_pickle('onl.pkl'); print(f'{time.time()-t0:.0f}s')
    pd.set_option('display.width', 250)
    print(df.groupby('boyut').apply(lambda g: pd.Series(dict(IS_kârlı=(g.IS > 0).mean().round(2), OOS_kârlı=(g.OOS > 0).mean().round(2), en_iyi_IS=g.IS.max().round(), onun_OOS=g.loc[g.IS.idxmax(), 'OOS'].round(), onun_net=g.loc[g.IS.idxmax(), 'net'].round(), max_net=g.net.max().round())), include_groups=False).to_string())
    print(df.sort_values('IS', ascending=False).head(10).round(0).to_string(index=False))
