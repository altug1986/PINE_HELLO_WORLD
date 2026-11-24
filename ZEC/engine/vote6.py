import sys, json; sys.path.insert(0, '/home/claude/pine_hello_world/ZEC/engine')
import numpy as np, pandas as pd
from zec_data import load; import z01, z02
B = load(2, src='/home/claude/master/'); F = z01.prep(B)
M = np.load('/home/claude/zecwork/votes.npy'); names = json.load(open('/home/claude/zecwork/vote_names.json')); ix = {k: j for j, k in enumerate(names)}
mid = B.index[0] + (B.index[-1] - B.index[0]) / 2
w = np.zeros(len(names)); w[[ix[k] for k in ['P07_rsi', 'P08_zema', 'P09_spike', 'P10_bb', 'P12_oiLiq', 'X17_basis', 'P04_htf4h']]] = 1
S = M.astype(float) @ w
for lab, SS in (('ÇİFT YÖN', S), ('LONG OYLAMA', np.maximum(S, -2))):
    eq, T = z02.sim3(F['o'], F['h'], F['l'], F['c'], F['atr'], SS, 3.0, -1e9, 12.0, 480)
    t = pd.DataFrame(T, columns=['ei', 'xi', 'side', 'ep', 'xp', 'pnl', 'mae']); t['et'] = B.index[t.ei.astype(int)]
    e = pd.Series(eq, index=B.index); f = z01.metrics(e, t); a = z01.metrics(e, t, None, mid); b = z01.metrics(e, t, mid, None); s = z01.sc4(e)
    print(lab, f"net {f['net']:.0f} (%{f['pct']:.0f}) n {f['n']} PF {f['PF']:.2f} win {f['win']:.0f}% | MaxDD {f['MaxDD']:.0f} P95/98/99 {f['P95']:.0f}/{f['P98']:.0f}/{f['P99']:.0f} | Calmar {f['Calmar']:.2f} Sharpe {f['Sharpe']:.2f} Sortino {f['Sortino']:.2f} | SC4 {s['SC4']:.3f} | σ10dk {f['s10']:.2f} σ1s {f['s1h']:.2f} | MAE P90 {f['maeP90']:.1f}% max {f['maeMax']:.1f}% | IS {a['net']:.0f} OOS {b['net']:.0f} | pozitif ay %{f['posM']:.0f}")
