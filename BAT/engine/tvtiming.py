import numpy as np, pandas as pd, v55
from tvcmp63 import tvx, Mdf, Z
from astre_test import astre, TS, R_
from v55 import t, n
from mtm30 import m30
T = tvx('tv/v63_tv.xlsx'); T = T[T.xt <= t[-1]]
s0 = (pd.Timestamp('2026-05-01') - pd.Timestamp('1970-01-01')) // pd.Timedelta('1ms'); Tl = T[T.et >= s0]; kt = set(zip(Tl.et, Tl.dr))
for lab, sh in [('B (H−1:55 kaydı)', -300000), ('A (H:00 kaydı)', 0)]:
    GL, GS = astre(sh); _, R = R_(GL, GS, TS); M = Mdf(R); Ml = M[M.et >= s0]; km = set(zip(Ml.et, Ml.dr))
    print(f'{lab}: 2026-05→ TV {len(Tl)} işlem {Tl.pnl.sum():.0f}$ | motor {len(Ml)} işlem {Ml.pnl.sum():.0f}$ | TV kapsama %{len(kt & km)/len(kt)*100:.1f} motor kapsama %{len(kt & km)/len(km)*100:.1f}')
# TV benzeri: kapı yalnız 2026-04-01 sonrası (B zamanlaması) → TV ile kalibrasyon; ayrıca kapısız / tam kapılı
GL, GS = astre(-300000); s1 = t >= (pd.Timestamp('2026-04-01') - pd.Timestamp('1970-01-01')) // pd.Timedelta('1ms')
rows = []
for lab, gl, gs in [('motor kapısız', Z, Z), ('motor TV benzeri (kapı 2026-04 sonrası)', GL & s1, GS & s1), ('motor tam kapılı (hibrit hedefi)', GL, GS)]:
    _, R = R_(gl, gs, TS); m = m30(R); rows.append(dict(t=lab, işlem=len(R['pnl']), **{k: m[k] for k in ['net', 'MaxDD 30dk mum içi', 'Calmar 30dk', 'DD P95/P98/P99 30dk', 'Sharpe/Sortino 30dk', 'Ulcer/UPI 30dk']}))
pd.set_option('display.width', 250); print(pd.DataFrame(rows).to_string(index=False))
