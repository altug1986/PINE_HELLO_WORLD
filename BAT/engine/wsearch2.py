import numpy as np, pandas as pd, json, time, sys
from vote import vote_pos
from eng import load
from lib import simR, cut
from met import metrics, SPLIT
# SAĞLAM AĞIRLIK ARAMASI: amaç = ilk dönemin iki yarısındaki skorların MİNİMUMU (yarılar arası tutarlılık)
D0 = load(5); X0 = np.load('comps.npy'); k0 = np.searchsorted(D0['t'], SPLIT); MID = np.int64(pd.Timestamp('2022-06-15').value // 10**6)
km = np.searchsorted(D0['t'], MID); DI = cut(D0, SPLIT); XI = X0[:, :k0]
def halves(tg):
    R = simR(DI, tg); p = R['pnl']; xt = DI['t'][R['xb']]
    out = []
    for msk, a, b in [(xt < MID, 0, km), (xt >= MID, km, k0)]:
        Dh = {k: (v[a:b] if isinstance(v, np.ndarray) and len(v) == len(DI['t']) else v) for k, v in DI.items()}
        Rh = simR(Dh, tg[a:b]); m = metrics(Dh, Rh); out.append(m['net'] - 1.5 * m['p98'])
    return min(out), out
WV = [0.0, 0.5, 1.0, 2.0, 3.0]; ON = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]; OFFR = [0.0, 0.5]
if __name__ == "__main__":
    rng = np.random.default_rng(int(sys.argv[1])); best = None; rows = []; t0 = time.time()
    for it in range(1500):
        if best is None or rng.random() < 0.35:
            w = [WV[rng.integers(5)] for _ in range(14)]; on = ON[rng.integers(6)]; offr = OFFR[rng.integers(2)]
        else:
            w, on, offr = list(best[0]), best[1], best[2]; j = rng.integers(16)
            if j < 14: w[j] = WV[rng.integers(5)]
            elif j == 14: on = ON[rng.integers(6)]
            else: offr = OFFR[rng.integers(2)]
        if sum(w) == 0: continue
        tg, _ = vote_pos(XI, np.array(w, np.float64), on, on * offr, 1); s, hs = halves(tg)
        rows.append(dict(w=json.dumps(w), on=on, offr=offr, sc=s, a=hs[0], b=hs[1]))
        if best is None or s > best[3]: best = (w, on, offr, s)
    df = pd.DataFrame(rows); df.to_pickle(f'ws2_{sys.argv[1]}.pkl'); print(f'{time.time()-t0:.0f}s')
    pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 80); print(df.sort_values('sc', ascending=False).drop_duplicates('w').head(3).to_string(index=False))
