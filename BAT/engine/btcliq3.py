import itertools, numpy as np, pandas as pd
import btcliq as B
from btcliq2 import D, c, run, Fr, apply25, ent, deep, capB, crash, v, report
rows = [report(D, v(np.zeros(len(c), np.int64)), 'V7.1')]
g = lambda s: np.where(deep == -1, 0, np.nan_to_num(s).astype(np.int64)).astype(np.int64)
S = {'BTC likidasyon 30 dk (−1.5/−1.5)': g((B.bo6 <= -1.5) & (B.bp6 <= -1.5)), 'BTC likidasyon 30 dk (−2/−2)': g((B.bo6 <= -2) & (B.bp6 <= -2)),
     'BTC OI boşalması 24 s ≤ −10': g(B.bo288 <= -10), 'BTC kapitülasyon + BAT 4 s ≤ −4': g((B.bo48 <= -3) & (B.bp48 <= -3) & (B.ap48 <= -4))}
for (lab, s), H, sl in itertools.product(S.items(), (144, 288), (3.0, 5.0)): rows.append(report(D, v(ent(s, c, H, sl)), f'{lab} · {H // 12} s · stop %{sl:g}'))
s44 = ent(g((B.bo48 <= -4) & (B.bp48 <= -4)), c, 288, 5.0)
ov = np.where(capB != 0, capB, s44); rows.append(report(D, apply25(D, run(D, ovr=ov), Fr), 'BTC kapitülasyon 4/4 BTC çöküşünün YERİNE'))
r = pd.DataFrame(rows); b0 = r.iloc[0]; pd.set_option('display.width', 250)
print('V7.1 üstü', int((r.net > b0.net).sum()), '/', len(r) - 1, '| %6 üstü', int((r.net > b0.net * 1.06).sum())); print(r.sort_values('net', ascending=False).head(10).to_string(index=False))
