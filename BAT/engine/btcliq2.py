import itertools, numpy as np, pandas as pd
import btcliq as B
from fundhunt import D, c, t, run, Fr, apply25
from btcx import ent, b24, b1
from edge2 import sig
from p1m import report
deep = Fr['deep']
capB = ent(np.where(deep == -1, 0, sig('Kapitülasyon', (5, 4))).astype(np.int64), c, 144, 5.0)
crash = ent(np.where(deep == -1, 0, ((b24 <= -10) & (b1 >= 0))).astype(np.int64), c, 288, 3.0)
def v(new): ov = np.where(capB != 0, capB, np.where(new != 0, new, crash)); return apply25(D, run(D, ovr=ov), Fr)
rows = [report(D, v(np.zeros(len(c), np.int64)), 'V7.1')]
for (a, b), H, sl, g in itertools.product(((3, 3), (4, 4), (5, 5)), (144, 288), (3.0, 5.0), (True, False)):
    s = ((np.nan_to_num(B.bo48) <= -a) & (np.nan_to_num(B.bp48) <= -b)).astype(np.int64)
    if g: s = np.where(deep == -1, 0, s)
    r = report(D, v(ent(s, c, H, sl)), f'BTC kapitülasyon {a}/{b} · {H // 12} s · stop %{sl:g}' + (' · derin düşüşte kapalı' if g else '')); rows.append(r)
r = pd.DataFrame(rows); r.to_pickle('btcliq2.pkl'); pd.set_option('display.width', 250); b0 = r.iloc[0]
print(r.iloc[:1].to_string(index=False)); print('V7.1 üstü', int((r.net > b0.net).sum()), '/', len(r) - 1, '| %6 üstü', int((r.net > b0.net * 1.06).sum()))
print(r.iloc[1:].sort_values('IS', ascending=False).head(10).to_string(index=False))
