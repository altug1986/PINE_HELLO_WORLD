import numpy as np, pandas as pd, itertools, time, sys
from v55 import *
import v55
from v54 import X0, CFG, comp15
from vote import vote_pos
from ent import g_ent
X15 = comp15(1.5, 4.0)
def tg16(k, m, bk, hold, sl, w16, mode):
    mo = MOM[k]; sg = np.sign(mo)
    if mode == 0: s = np.where((BV >= bk) & (np.abs(mo) >= m), sg, 0)           # BTC hacmi destekli kendi momentumu: takip
    else: s = np.where((BV >= bk) & (np.abs(mo) >= m), -sg, 0)                  # aynı koşulda karşı (sönümlenme)
    X = np.vstack([X0, X15[None, :], g_ent(c, s.astype(np.int64), hold, sl)[None, :]])
    return vote_pos(X, np.array(CFG['w'] + [1.0, w16], float), CFG['on'], CFG['off'], 1)[0]
if __name__ == '__main__':
    rows = []; t0 = time.time(); md = int(sys.argv[1])
    for k, m, bk, hold, w16 in itertools.product([12, 48, 144], [2.0, 3.0, 5.0, 8.0], [1.5, 2.0, 3.0], [48, 144, 288], [1.0, 2.0, 4.0]):
        v55.TG = tg16(k, m, bk, hold, 10.0, w16, md); rows.append(dict(k=k, m=m, bk=bk, hold=hold, w16=w16, **evalp({})))
    d = pd.DataFrame(rows); d.to_pickle(f's55_M{md}.pkl'); pd.set_option('display.width', 250)
    print(md, len(d), f'{time.time()-t0:.0f}s | sc>2239 & OOS>1871:', ((d.sc > 2239) & (d.OOS > 1871)).sum(), '| net>7500:', (d.net > 7500).sum())
    print(d.sort_values('sc', ascending=False).head(6)[['k', 'm', 'bk', 'hold', 'w16', 'sc', 'h0', 'h1', 'net', 'n', 'pf', 'calmar', 'p98', 'mddc', 'IS', 'OOS', 'last12', 'bmin', 'bstd', 'bneg']].to_string(index=False))
