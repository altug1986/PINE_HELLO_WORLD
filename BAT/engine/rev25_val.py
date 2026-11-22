import numpy as np, pandas as pd
from btcx import D, c, t, run, report, trades, ent, cap, b24, b1
from rev25 import feats, apply25
from tvcmp63 import tvx
F_tv = feats(D, True); F_full = feats(D, False)
core66 = run(D, whale='tv'); ex = apply25(D, core66, F_tv)
R, _ = trades(D, ex); P = pd.DataFrame(dict(et=t[R['eb']], xt=t[R['xb']], dr=R['dir'], pnl=R['pnl']))
T = tvx('tv/v67b_tv.xlsx'); T = T[T.xt <= t[-1]]; kt = set(zip(T.et, T.dr)); kp = set(zip(P.et, P.dr)); both = kt & kp
print(f'DOĞRULAMA port V6.6+REV-25 (TV benzeri): {len(P)} işlem {P.pnl.sum():.0f}$ | TV V6.7 {len(T)} işlem {T.pnl.sum():.0f}$ | giriş eşleşmesi TV %{len(both)/len(kt)*100:.1f} port %{len(both)/len(kp)*100:.1f}')
s = ((b24 <= -10) & (b1 >= 0)).astype(np.int64); side = np.where(cap != 0, cap, ent(s, c, 288, 3.0))
rows = [report(D, run(D, whale='tv'), 'V6.6 · TV benzeri'), report(D, ex, 'V6.6 + REV-25 · TV benzeri')]
for w, F, tag in [('tam', F_full, 'tam balina'), ('tv', F_tv, 'TV benzeri')]:
    core = run(D, whale=w, ovr=side); rows += [report(D, core, f'V6.9 · {tag}'), report(D, apply25(D, core, F), f'V6.9 + REV-25 · {tag}')]
pd.set_option('display.width', 250); print(pd.DataFrame(rows).to_string(index=False))
