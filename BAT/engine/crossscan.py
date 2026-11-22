import itertools, time, numpy as np, pandas as pd
import btcliq as B
from btcliq2 import D, c, run, Fr, apply25, ent, deep, capB, crash
from fundhunt import F, Fz, p24, o24, oi, L, ch
from p1m import trades, report, SPL, PP0
t = D['t']; n = len(c); END = t[-1]
z = lambda x, N: ((pd.Series(x) - pd.Series(x).rolling(N, min_periods=N // 2).mean()) / pd.Series(x).rolling(N, min_periods=N // 2).std()).to_numpy()
m = pd.read_pickle('top5.pkl'); f5 = t // 300000 * 300000; jm = np.searchsorted(m.ms.to_numpy(), f5 - 300000, side='right') - 1
mv = lambda col: np.where(jm >= 0, m[col].where(m[col] > 0).to_numpy()[np.clip(jm, 0, None)], np.nan)
tp = np.log(mv('sum_toptrader_long_short_ratio')); gl = np.log(mv('count_long_short_ratio')); tpz = z(tp - L(tp, 288), 8640); glz = z(gl, 8640)
hi24 = c >= pd.Series(c).rolling(288).max().to_numpy(); o1 = ch(oi, 12); N = np.nan_to_num
def v71(new=None, xGL=None, xGS=None, pp=None, stf=2.5, tsB=None):
    ov = np.where(capB != 0, capB, crash) if new is None else np.where(capB != 0, capB, np.where(new != 0, new, crash))
    return apply25(D, run(D, ovr=ov, xGL=xGL, xGS=xGS, pp=pp, stf=stf, tsB=tsB), Fr)
def fast(ex, lab, **kw):
    R, _ = trades(D, ex); p = R['pnl']; et = t[R['eb']]; xt = t[R['xb']]
    return dict(t=lab, net=round(p.sum()), n=len(p), PF=round(p[p > 0].sum() / -p[p < 0].sum(), 3), IS=round(p[et < SPL].sum()), OOS=round(p[et >= SPL].sum()),
                L365=round(p[xt >= END - 365 * 86400000].sum()), L200=round(p[xt >= END - 200 * 86400000].sum()), **kw)
if __name__ == '__main__':
    t0 = time.time(); base = fast(v71(), 'V7.1'); rows = []
    LONG = {'BTC kapitülasyon 3/3': (N(B.bo48) <= -3) & (N(B.bp48) <= -3), 'BTC kapitülasyon 4/4': (N(B.bo48) <= -4) & (N(B.bp48) <= -4), 'BTC 30dk likidasyon': (N(B.bo6) <= -1.5) & (N(B.bp6) <= -1.5),
            'BTC OI boşalması 24s': N(B.bo288) <= -10, 'BAT OI boşalması 24s': N(o24) <= -10, 'Top trader poz 24s z≥2': N(tpz) >= 2, 'Funding z≤−2': N(Fz) <= -2}
    SHORT = {'24s tepe + OI≥+5': hi24 & (N(o24) >= 5), 'Funding ≥ %0.03': N(F) >= 0.03, 'Kalabalık short z≤−2 (momentum)': N(glz) <= -2, 'Hızlı kaldıraç OI 1s≥4': N(o1) >= 4,
             'BTC kaldıraç + düşüş': (N(B.bo288) >= 8) & (N(B.bp288) <= -4)}
    for (lab, cond, d), H, sl in itertools.product([(k, v_, 1) for k, v_ in LONG.items()] + [(k, v_, -1) for k, v_ in SHORT.items()], (48, 144, 288), (3.0, 5.0)):
        s = (cond.astype(np.int64) * d); s = np.where(deep == -d, 0, s).astype(np.int64)
        rows.append(fast(v71(new=ent(s, c, H, sl)), f'YAN MOTOR {"L" if d > 0 else "S"}: {lab} · {H // 12}s · stop{sl:g}', aile='yan motor'))
    hold = lambda x: pd.Series(x.astype(float)).rolling(288, min_periods=1).max().to_numpy() > 0
    Z = np.zeros(n, bool)
    G = [('Funding≥0.03 → long yok', N(F) >= 0.03, Z), ('Funding z≤−2 → short yok', Z, N(Fz) <= -2), ('24s tepe+OI artıyor → long yok', hi24 & (N(o24) >= 5), Z),
         ('BTC kapitülasyon sonrası 24s → short yok', Z, hold((N(B.bo48) <= -3) & (N(B.bp48) <= -3))), ('BTC OI boşalması sonrası 24s → short yok', Z, hold(N(B.bo288) <= -10)),
         ('Kalabalık long z≥2 → long yok', N(glz) >= 2, Z), ('Kalabalık short z≤−2 → short yok', Z, N(glz) <= -2)]
    for lab, gL, gS in G: rows.append(fast(v71(xGL=gL, xGS=gS), f'KAPI: {lab}', aile='kapı'))
    NM = ['onThr', 'onTrend', 'stCnt', 'stHold', 'stop%', 'reArm%', 'tsLoss', 'tsAdx', 'stAdxH', 'stAdxT', 'trAdx']
    PV = {0: (0.02, 0.06), 1: (0.10, 0.20), 2: (0.08, 0.16), 3: (-0.10, -0.20), 4: (5.0, 6.0, 8.0, 10.0), 5: (3.0, 5.0), 6: (0.5, 2.0), 7: (20.0, 30.0), 8: (20.0, 30.0), 9: (25.0, 35.0), 10: (15.0, 25.0)}
    for k, vals in PV.items():
        for val in vals: pp = PP0.copy(); pp[k] = val; rows.append(fast(v71(pp=pp), f'AYAR: {NM[k]} {PP0[k]:g}→{val:g}', aile='ayar'))
    for val in (2.0, 3.0): rows.append(fast(v71(stf=val), f'AYAR: Supertrend çarpanı 2.5→{val:g}', aile='ayar'))
    for val in (216, 360): rows.append(fast(v71(tsB=float(val)), f'AYAR: pozda kalış 288→{val}', aile='ayar'))
    r = pd.DataFrame(rows)
    for k in ('net', 'IS', 'OOS', 'L365', 'L200'): r['Δ' + k] = r[k] - base[k]
    r['çapraz'] = (r.ΔIS > 0) & (r.ΔOOS > 0) & (r.ΔL365 > 0) & (r.ΔL200 > 0); r.to_pickle('crossscan.pkl')
    pd.set_option('display.width', 260); print(base, f'| {len(r)} test, {time.time()-t0:.0f}s')
    print('4 dönemin hepsinde iyi:', int(r.çapraz.sum()), '| aile:', r[r.çapraz].aile.value_counts().to_dict())
    print(r[r.çapraz].sort_values('ΔL365', ascending=False)[['t', 'net', 'n', 'PF', 'ΔIS', 'ΔOOS', 'ΔL365', 'ΔL200']].head(15).to_string(index=False))
    print('--- son 12 ayda en çok katkı (çapraz olmasa da):'); print(r.sort_values('ΔL365', ascending=False)[['t', 'net', 'ΔIS', 'ΔOOS', 'ΔL365', 'ΔL200']].head(8).to_string(index=False))
