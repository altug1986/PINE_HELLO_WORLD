# BAĞIMSIZ ZEC HEDGE MOTORU: yalnız XMR göreli zayıfken ZEC short (XMR long bloğunun karşılığı) → portföy +200 $ ΔSC4
import sys; sys.path.insert(0, '/home/claude/zecwork'); sys.argv = ['x', 'yok']
import pfsearch as P; import numpy as np, pandas as pd, numba, itertools, time
from pfsearch import o, h, l, c, tms, n, TICK, ix, S, X24, B24, XZ, ZR60
from sc4 import sc4
@numba.njit(cache=True)
def hsim(o, h, l, c, cond, conf, H, sl, tp, tick):
    n = len(c); pos = 0; ep = 0.0; q = 0.0; real = 0.0; last = -10 ** 9; eq = np.zeros(n); stp = 0.0; tpp = 0.0
    for i in range(1, n):
        if pos == -1:
            if h[i] >= stp: real += q * (ep - (max(o[i], stp) + tick)) - 0.0004 * q * (ep + stp); pos = 0
            elif tp > 0 and l[i] <= tpp: real += q * (ep - (min(o[i], tpp) + tick)) - 0.0004 * q * (ep + tpp); pos = 0
        if cond[i] and conf[i]: last = i
        if pos == -1 and i - last > H:
            xp = c[i] + tick; real += q * (ep - xp) - 0.0004 * q * (ep + xp); pos = 0
        elif pos == 0 and cond[i] and conf[i]:
            pos = -1; ep = c[i] - tick; q = 500.0 / c[i]; stp = ep * (1 + sl); tpp = ep * (1 - tp)
        eq[i] = real + (q * (ep - c[i]) - 0.0004 * q * ep if pos == -1 else 0.0)
    return eq
rows = []; T1 = np.ones(n, np.bool_); t0 = time.time()
for th, k, m in itertools.product((0.0, 2.0, 4.0), (60, 240), (1.5, 2.0, 2.5)):
    cond = np.nan_to_num((X24 - B24 <= -th) & (XZ[k] <= -m)).astype(np.bool_)
    for cf, H, sl, tp in itertools.product(('yok', 'ZEC 1s düşüyor'), (60, 240, 720), (0.02, 0.04), (0.0, 0.03)):
        e = hsim(o, h, l, c, cond, T1 if cf == 'yok' else (ZR60 < 0), H, sl, tp, TICK)
        r = P.pmetrics(e, f'HEDGE XMR−BTC ≤ −%{th:g} & XMR {k}dk z ≤ −{m} | teyit:{cf} tut{H} sl%{sl * 100:g} tp%{tp * 100:g}'); rows.append(r)
R = pd.DataFrame(rows); R.to_pickle('/home/claude/zecwork/hedge1.pkl'); pd.set_option('display.width', 330); pd.set_option('display.max_colwidth', 80)
c_ = ['model', 'SEÇİM ΔSC4', 'SEÇİM ΔMaxDD%', 'SEÇİM çukur$', 'DOĞRULAMA ΔSC4', 'DOĞRULAMA ΔCalmar', 'DOĞRULAMA ΔMaxDD%', 'DOĞRULAMA çukur$', 'DOĞRULAMA çukur%+', 'DOĞRULAMA zayıfAy+', 'DOĞRULAMA korAy', 'net', 'MaxDD', 'MaxDDnom']
print('varyant', len(R), f'{time.time() - t0:.0f} sn · SEÇİM ΔSC4 > 0:', (R['SEÇİM ΔSC4'] > 0).sum(), '· DOĞRULAMA ΔSC4 > 0:', (R['DOĞRULAMA ΔSC4'] > 0).sum(), '· ikisi:', ((R['SEÇİM ΔSC4'] > 0) & (R['DOĞRULAMA ΔSC4'] > 0)).sum())
print(R.sort_values('SEÇİM ΔSC4', ascending=False)[c_].head(12).round(2).to_string(index=False))
# V2.9 nominal duyarlılığı
e9 = P.ov5(o, h, l, c, P.posA, P.bEp, P.bXp, P.bXd, P.R10, P.F0, 0, 0, 0.0, ZR60 < 0, P.F0, 0, TICK); z = S(e9, ix).reindex(P.Pp.index, method='ffill').fillna(0.0)
print('\nV2.9 nominal duyarlılığı (ΔSC4 SEÇİM / DOĞRULAMA):')
for nom in (25, 50, 100, 150, 200):
    out = []
    for w, (a0, a1) in P.WIN.items():
        m = (P.Pp.index >= a0) & (P.Pp.index < a1); p = P.Pp[m] - P.Pp[m].iloc[0]; zz = z[m] * nom / 500; zz = zz - zz.iloc[0]
        out.append(sc4(p + zz, 4600 + nom) - P.BASE[w]['SC4'])
    print(f'  {nom:>3} $: {out[0]:+.2f} / {out[1]:+.2f}')
