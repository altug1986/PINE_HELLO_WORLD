import numpy as np, pandas as pd
from tvcmp63 import tvx, Mdf, Z
from astre_test import astre, TS, R_
from v55 import t
a, b = tvx('tv/v63_tv.xlsx'), tvx('tv/v63b_tv.xlsx')
print('önceki', len(a), round(a.pnl.sum(), 2), '| yeni', len(b), round(b.pnl.sum(), 2), '| aynı giriş', len(set(zip(a.et, a.dr)) & set(zip(b.et, b.dr))))
# kapı istatistiği (motor, V6.3): kaç giriş engellendi, engellenenlerin kapısız sonucu
GL, GS = astre(-300000); _, Rg = R_(GL, GS, TS); _, R0 = R_(Z, Z, TS); Mg, M0 = Mdf(Rg), Mdf(R0)
print('long engelli bar %', round(GL.mean()*100, 1), '| short engelli bar %', round(GS.mean()*100, 1), '| ikisi %', round((GL & GS).mean()*100, 1))
s = (pd.Timestamp('2026-04-16') - pd.Timestamp('1970-01-01')) // pd.Timedelta('1ms')
for lab, x in [('tüm dönem', t >= 0), ('2026-04-16 →', t >= s)]:
    m0 = M0[x[np.searchsorted(t, M0.et)]]; mg = Mg[x[np.searchsorted(t, Mg.et)]]
    k = set(zip(m0.et, m0.dr)) - set(zip(mg.et, mg.dr)); bl = m0[[(e, d) in k for e, d in zip(m0.et, m0.dr)]]
    print(f'{lab}: kapısız {len(m0)} işlem {m0.pnl.sum():.0f}$ | kapılı {len(mg)} işlem {mg.pnl.sum():.0f}$ | kapının kestiği {len(bl)} giriş (long {int((bl.dr>0).sum())} short {int((bl.dr<0).sum())}), kapısızda bu girişlerin sonucu {bl.pnl.sum():.0f}$, ortalama {bl.pnl.mean():.2f}$')
