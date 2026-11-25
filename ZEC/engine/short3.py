import sys; sys.path.insert(0, '/home/claude/pine_hello_world/ZEC/engine')
import numpy as np, pandas as pd
src = open('/home/claude/pine_hello_world/ZEC/engine/short_scan.py').read().split("rows = []; base = {}")[0]; exec(src)
from path_quality import path
ks = np.array([1.0, 2.0, 3.0])
rng = np.random.default_rng(5); rows = []
def q(e, side=-1):
    P, rt, av, mae = path(h, l, c, a, e.astype(np.int64), side, ks, 60)
    return np.nanmean(P, 0) * 100, np.nanmean(rt), np.nanmean(av), np.nanpercentile(mae, 90)
base = {}
for lab, val in (('TÜM', None), ('UP', 1), ('DN', -1), ('YAT', 0)):
    pool = np.arange(2000, n - 800) if val is None else np.where(RG[2000:n - 800] == val)[0] + 2000
    for half, sel in (('IS', pool[pool < mid]), ('OOS', pool[pool >= mid])):
        e = np.sort(rng.choice(sel, 4000, replace=False)); P, rt, av, mae = q(e); base[(lab, half)] = (P[0], P[1], rt, av)
print('RASTGELE SHORT (referans) P(lehe 1/2 ATR) · doğru taraf % · sürüş bps:  ' + ' | '.join(f'{k[0]}-{k[1]} {v[0]:.1f}/{v[1]:.1f} · {v[2]:.1f} · {v[3]:.1f}' for k, v in base.items()))
for name, trig in T.items():
    ev = dedupe(trig)
    for lab, val in (('TÜM', None), ('UP', 1), ('DN', -1), ('YAT', 0)):
        e = ev if val is None else ev[RG[ev] == val]; e = e[(e > 2000) & (e < n - 800)]
        if len(e) < 40 or (e < mid).sum() < 15 or (e >= mid).sum() < 15: continue
        r = dict(tetik=name, rejim=lab, n=len(e))
        for half, ee in (('IS', e[e < mid]), ('OOS', e[e >= mid])):
            P, rt, av, mae = q(ee); b = base[(lab, half)]
            r[f'P1{half}'] = P[0] - b[0]; r[f'P2{half}'] = P[1] - b[1]; r[f'doğru{half}'] = rt - b[2]; r[f'sürüş{half}'] = av - b[3]; r[f'MAE90{half}'] = mae
        rows.append(r)
D = pd.DataFrame(rows); D.to_csv('/home/claude/pine_hello_world/ZEC/research/SHORT_SCAN_SURUS.csv', index=False)
pd.set_option('display.width', 250)
D['skor'] = D[['P2IS', 'P2OOS']].min(axis=1)
ok = D[(D.P2IS > 0) & (D.P2OOS > 0) & (D.doğruIS > 0) & (D.doğruOOS > 0)]
print(f'\nRASTGELE SHORT\'A GÖRE FARK — iki yarıda P(önce lehe 2 ATR) ve doğru tarafta kalma süresi pozitif: {len(ok)} / {len(D)}')
cols = ['tetik', 'rejim', 'n', 'P1IS', 'P1OOS', 'P2IS', 'P2OOS', 'doğruIS', 'doğruOOS', 'sürüşIS', 'sürüşOOS', 'MAE90OOS']
print(ok.sort_values('skor', ascending=False).head(20)[cols].round(1).to_string(index=False))
