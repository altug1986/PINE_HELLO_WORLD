import itertools, numpy as np, pandas as pd
from monthly import Zs
from crossscan import D, c, t, run, Fr, apply25, ent, deep, capB, crash, fast, v71
from p1m import report
base = fast(v71(), 'V7.1'); rows = []
S = {'BTC z−2.5': (Zs['bo48'] <= -2.5) & (Zs['bp48'] <= -2.5), 'BTC z−3': (Zs['bo48'] <= -3) & (Zs['bp48'] <= -3),
     'BAT|BTC z−2.5': ((Zs['bo48'] <= -2.5) & (Zs['bp48'] <= -2.5)) | ((Zs['ao48'] <= -2.5) & (Zs['ap48'] <= -2.5)), 'BAT|BTC z−3': ((Zs['bo48'] <= -3) & (Zs['bp48'] <= -3)) | ((Zs['ao48'] <= -3) & (Zs['ap48'] <= -3))}
for (lab, cond), H, sl, g, mode in itertools.product(S.items(), (144, 288), (3.0, 5.0), (True, False), ('ekle', 'yerine')):
    s = np.nan_to_num(cond).astype(np.int64)
    if g: s = np.where(deep == -1, 0, s).astype(np.int64)
    e = ent(s, c, H, sl)
    if mode == 'ekle': ex = v71(new=e)
    else: ex = apply25(D, run(D, ovr=e), Fr)
    rows.append(fast(ex, f'{lab} · {H // 12}s · stop{sl:g}' + (' · derin kapalı' if g else '') + f' · {mode}', mode=mode))
r = pd.DataFrame(rows)
for k in ('net', 'IS', 'OOS', 'L365', 'L200'): r['Δ' + k] = r[k] - base[k]
r['çapraz'] = (r.ΔIS > 0) & (r.ΔOOS > 0) & (r.ΔL365 > 0) & (r.ΔL200 > 0); r.to_pickle('monthly2.pkl'); pd.set_option('display.width', 260)
print(base); print('4 dönemde iyi:', int(r.çapraz.sum()), '/', len(r), '| %6 üstü:', int((r.net > base['net'] * 1.06).sum()))
print(r.sort_values('net', ascending=False).head(12)[['t', 'net', 'n', 'PF', 'ΔIS', 'ΔOOS', 'ΔL365', 'ΔL200', 'çapraz']].to_string(index=False))
