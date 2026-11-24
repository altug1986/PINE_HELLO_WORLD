import sys, json, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/claude/pine_hello_world/ZEC/engine')
import numpy as np, pandas as pd
from zec_data import load; import z01, z02
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
B = load(2, src='/home/claude/master/'); F = z01.prep(B); c = F['c']; a = F['atr']
M = np.load('/home/claude/zecwork/votes.npy'); names = json.load(open('/home/claude/zecwork/vote_names.json'))
mid = np.searchsorted(B.index.values, (B.index[0] + (B.index[-1] - B.index[0]) / 2).to_datetime64())
P = [j for j, k in enumerate(names) if k[0] in 'PX']
# HAM üretici: P+X eşit ağırlık, düşük eşik, sabit ufuk (aşırı işlem)
w0 = np.zeros(len(names)); w0[P] = 1
for Tin, H in ((2, 120), (4, 120), (2, 240)):
    eq, t = z02.run3(B, F, M, w0, Tin, -1e9, 8, H); i = t.ei.astype(int).values; sd = t.side.values
    X = pd.DataFrame({k: M[i, j] * sd for j, k in enumerate(names)})
    X['atrP'] = a[i] / c[i] * 100
    for n_ in (60, 720, 2880): X[f'trend{n_}'] = (c[i] / pd.Series(c).ewm(span=n_, adjust=False).mean().values[i] - 1) * 100 * sd
    X['basis'] = B.basis_bps.values[i] * sd; oi = B.oi_zec.values; X['doi1h'] = (oi[i] / oi[np.maximum(i - 30, 0)] - 1) * 100
    bt = B.btc_close.values; X['btc1h'] = (bt[i] / bt[np.maximum(i - 30, 0)] - 1) * 100 * sd
    X['side'] = sd; X['volR'] = F['v'][i] / F['vma'][i]
    portable = [k for k in X.columns if not k.startswith('R')]
    y = (t.pnl.values > 0).astype(int); isk = i < mid
    print(f'HAM Tin{Tin} H{H}: n {len(t)} net {t.pnl.sum():.0f} | IS {t.pnl[isk].sum():.0f} OOS {t.pnl[~isk].sum():.0f} | kazanma %{y.mean()*100:.1f}')
    for lab, cols in (('Pine-taşınır', portable), ('tümü', list(X.columns))):
        Xi = X[cols].fillna(0).values
        for mname, mdl in (('GBM', HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05, l2_regularization=1.0, min_samples_leaf=200)),
                           ('LOJ', None)):
            if mname == 'LOJ':
                sc = StandardScaler().fit(Xi[isk]); lr = LogisticRegression(C=0.1, max_iter=500).fit(sc.transform(Xi[isk]), y[isk])
                p = lr.predict_proba(sc.transform(Xi))[:, 1]
            else:
                p = mdl.fit(Xi[isk], y[isk]).predict_proba(Xi)[:, 1]
            out = []
            for q in (0.7, 0.8, 0.9):
                thr = np.quantile(p[isk], q); sel = p >= thr
                so = sel & ~isk; si = sel & isk
                pn = t.pnl.values[so]; pf = pn[pn > 0].sum() / -pn[pn < 0].sum() if (pn < 0).any() else np.nan
                out.append(f'üst%{int((1-q)*100)}: IS {t.pnl.values[si].sum():6.0f} | OOS n {so.sum():5d} net {pn.sum():6.0f} PF {pf:4.2f}')
            print(f'  {lab:12s} {mname}: ' + ' || '.join(out))
