exec(open('/home/claude/zecwork/mega5.py').read().split("rows = []; t0 = time.time()")[0])
import json, glob
order = ['P09_spike', 'P07_rsi', 'P08_zema', 'P10_bb', 'P15_resid', 'P04_htf4h', 'P12_oiLiq', 'P11_oiCont']; sel.cache_clear()
eq, T = run(0, 25, 480, 2, 2, 1, 0, 10, 0.4, 0.0, 40.0, 1920, 4.0, 2)
R = RL[0]; f = F[480]; fi = f[:mid]; Ri = R[:mid]; tab = {}
for k in order:
    vi = V[k][:mid]
    for side in (1, -1):
        m_ = ['0'] * 96
        for val in range(96):
            mm = (vi == side) & (Ri == val)
            if mm.sum() >= 400 and np.nanmean(fi[mm] * side) >= 25: m_[val] = '1'
        tab[f'{k}_{"L" if side > 0 else "S"}'] = ''.join(m_)
old = json.load(open('/home/claude/zecwork/tv_tablo.json')); print('eski 5 oy tablosu aynı mı:', all(tab[k] == old[k] for k in old), {k: v.count('1') for k, v in tab.items() if 'oi' in k})
json.dump(tab, open('/home/claude/zecwork/tv_tablo2.json', 'w'))
x30 = eq[::30]; dd = np.maximum.accumulate(x30) - x30; d_ = np.diff(x30[::48]); gp = T[T[:, 5] > 0, 5].sum(); gl = -T[T[:, 5] < 0, 5].sum(); yb = np.r_[0, eq[np.searchsorted(idx.values, pd.date_range('2022-01-01', '2026-01-01', freq='YS').values)], eq[-1]]
print(dict(n=len(T), net=round(eq[-1]), PF=round(gp / gl, 2), MaxDD=round(dd.max()), DDp95=round(np.percentile(dd, 95)), DDp99=round(np.percentile(dd, 99)), Sharpe=round(d_.mean() / d_.std() * np.sqrt(365), 2),
           Sortino=round(d_.mean() / d_[d_ < 0].std() * np.sqrt(365), 2), yıllar=np.round(np.diff(yb)).tolist(), short=round(T[T[:, 2] < 0, 5].sum()), maeMax=round(T[:, 6].max() * 100, 1)))
t = pd.concat([pd.read_parquet(f_, columns=['time']).time for f_ in sorted(glob.glob('/home/claude/master/*.parquet'))]).values; ts_ = pd.to_datetime(t, unit='ms')
pd.DataFrame({'giris_utc': ts_[T[:, 0].astype(int)], 'cikis_utc': ts_[T[:, 1].astype(int)], 'yon': T[:, 2].astype(int), 'giris': T[:, 3], 'cikis': T[:, 4], 'net': T[:, 5].round(2)}).to_csv('/home/claude/pine_hello_world/ZEC/research/DENGELI_TV_V1.1_MOTOR_ISLEMLER.csv', index=False)
