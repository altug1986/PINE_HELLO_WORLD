# DENGELİ'nin TradingView'e taşınabilir sürümü: yalnız TV'de hesaplanabilen oylar (OHLC + BTC) ile aynı parametreler; kural tablosu döküm
exec(open('/home/claude/zecwork/mega5.py').read().split("rows = []; t0 = time.time()")[0])
PORT = ['P09_spike', 'P07_rsi', 'P08_zema', 'P10_bb', 'P15_resid', 'P04_htf4h']
cf = (25, 480, 2, 2, 1, 0, 10, 0.4, 0.0, 28.0, 3840, 4.0, 2)
def runo(ordr, lg=0):
    global order; order = ordr; sel.cache_clear(); return run(lg, *cf)
FULL = list(order); res = {}
for nm, od in (('DENGELİ (tam, 15 oy)', FULL), ('DENGELİ-TV (6 oy)', PORT)):
    eq, T = runo(od); x30 = eq[::30]; dd = np.maximum.accumulate(x30) - x30; d = np.diff(x30[::48]); mp = np.diff(np.r_[0, eq[mb]])[:-1]
    gp = T[T[:, 5] > 0, 5].sum(); gl = -T[T[:, 5] < 0, 5].sum()
    res[nm] = dict(n=len(T), net=eq[-1], IS=eq[mid] - eq[0], OOS=eq[-1] - eq[mid], PF=gp / gl, MaxDD=dd.max(), DDp95=np.percentile(dd, 95), DDp99=np.percentile(dd, 99),
                   Sharpe=d.mean() / d.std() * np.sqrt(365), Calmar=eq[-1] / yrs / dd.max(), ayMin=mp.min(), ayPoz=(mp > 0).mean() * 100, son2ay=eq[-1] - eq[s2], Kasım=eq[-1] - eq[p('2026-11-01')],
                   short=T[T[:, 2] < 0, 5].sum(), maeMax=T[:, 6].max() * 100)
    if od is PORT:
        np.save('/home/claude/zecwork/eq_TV.npy', x30); np.save('/home/claude/zecwork/T_TV.npy', T)
        R = RL[0]; L, S = sel(0, 25, 480); f = F[480]; fi = f[:mid]; Ri = R[:mid]; tab = {}
        for k in PORT:
            vi = V[k][:mid]
            for side in (1, -1):
                m_ = ['0'] * 96
                for val in range(96):
                    mm = (vi == side) & (Ri == val)
                    if mm.sum() >= 400 and np.nanmean(fi[mm] * side) >= 25: m_[val] = '1'
                tab[f'{k}_{"L" if side > 0 else "S"}'] = ''.join(m_)
        import json; json.dump(tab, open('/home/claude/zecwork/tv_tablo.json', 'w')); print({k: v.count('1') for k, v in tab.items()})
        print('ilk/son işlem:', idx[int(T[0, 0])], idx[int(T[-1, 1])])
pd.set_option('display.width', 200); print(pd.DataFrame(res).round(2).to_string())
