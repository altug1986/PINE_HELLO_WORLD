import numpy as np, numba as nb, pandas as pd, itertools
from eng import load
from lib import simR, cut
from met import metrics, SPLIT
from comps import NAMES
@nb.njit(cache=True)
def vote_pos(X, w, on, off, k):
    K, n = X.shape; tg = np.zeros(n, np.int64); pos = 0; run = 0; last = 0; W = np.abs(w).sum(); sc = np.zeros(n)
    for i in range(n):
        s = 0.0
        for j in range(K): s += w[j] * X[j, i]
        s /= W; sc[i] = s; want = pos
        if pos != 0 and s * pos < off: want = 0
        if s >= on: want = 1
        elif s <= -on: want = -1
        if want != 0 and want != pos:
            run = run + 1 if want == last else 1; last = want
            if run < k: want = pos if (pos == 0 or s * pos >= off) else 0
        else: run = 0; last = 0
        pos = want; tg[i] = pos
    return tg, sc
if __name__ == '__main__':
    D0 = load(5); X0 = np.load('comps.npy'); k0 = np.searchsorted(D0['t'], SPLIT); DI = cut(D0, SPLIT); XI = X0[:, :k0]
    st = []
    for j in range(14):
        m = metrics(DI, simR(DI, XI[j])); st.append(dict(net=m['net'], sharpe=max(m['sharpe'], 0), pf=m['pf'], exp=(XI[j] != 0).mean()))
    st = pd.DataFrame(st, index=NAMES); print(st.round(3).to_string())
    WS = {'eşit (kârlı bileşenler)': np.where(st.net > 0, 1.0, 0.0), 'IS Sharpe': st.sharpe.to_numpy(), 'IS PF−1': np.clip(st.pf.to_numpy() - 1, 0, None),
          'maruz kalma başı kâr': np.clip(st.net / np.maximum(st.exp, 0.01), 0, None).to_numpy() / 1000}
    rows = []
    for wn, w in WS.items():
        for on, offr, k in itertools.product([0.05, 0.1, 0.15, 0.2, 0.3, 0.4], [0.0, 0.5], [1, 3, 6]):
            tg, _ = vote_pos(XI, w.astype(np.float64), on, on * offr, k); m = metrics(DI, simR(DI, tg))
            rows.append(dict(ağırlık=wn, on=on, offr=offr, k=k, **{kk: m[kk] for kk in ['net', 'n', 'pf', 'sharpe', 'p98', 'mddc', 'upi']}))
    df = pd.DataFrame(rows); df['sc'] = df.net - 1.5 * df.p98; df.to_pickle('vote_IS.pkl'); pd.set_option('display.width', 250)
    print(df.groupby('ağırlık').apply(lambda g: pd.Series(dict(kârlı=(g.net > 0).mean().round(2), en_iyi_net=g.net.max(), en_iyi_sc=g.sc.max())), include_groups=False).to_string())
    top = df.sort_values('sc', ascending=False).head(6); print(top.to_string(index=False))
    for _, r in top.head(3).iterrows():
        tg, _ = vote_pos(X0, WS[r.ağırlık].astype(np.float64), r.on, r.on * r.offr, int(r.k)); m = metrics(D0, simR(D0, tg))
        print('TAM', r.ağırlık, r.on, r.offr, r.k, {kk: m[kk] for kk in ['net', 'n', 'pf', 'sharpe', 'p98', 'mddc', 'IS', 'OOS', 'last12', 'yr']})
