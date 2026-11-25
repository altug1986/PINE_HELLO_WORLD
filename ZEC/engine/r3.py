import sys; sys.path.insert(0, '/home/claude/pine_hello_world/ZEC/engine')
import numpy as np, pandas as pd, ztf, z01, z03
D = ztf.build(1, thr=25.0); B = D['B']; c, h, l, o, a = D['c'], D['h'], D['l'], D['o'], D['atr']; n = D['n']; mid = B.index[D['mid']]
sh = lambda x, k: np.r_[np.full(k, np.nan), x[:-k]]; r10 = (c / sh(c, 10) - 1) * 100; z = np.zeros(n, np.bool_)
aL = D['L'] >= 3; aS = (D['Sh'] >= 3) & ~aL
for lab, spkX in (('R3-1dk (kalite seçimi)', None), ('R3-1dk + sıçrama bekçisi %3', 3.0)):
    spk = np.nan_to_num(r10 >= spkX).astype(np.bool_) if spkX else z
    eq, T = z03.sim7(o, h, l, c, a, aL.astype(np.bool_), aS.astype(np.bool_), 11.0, 960, z, 8.0, 12.0, 240, spk, 60)
    t = pd.DataFrame(T, columns=['ei', 'xi', 'side', 'ep', 'xp', 'pnl', 'mae', 'own', 'guard']); t['et'] = B.index[t.ei.astype(int)]; t['xt'] = B.index[t.xi.astype(int)]
    e = pd.Series(eq, index=B.index); f = z01.metrics(e, t); A = z01.metrics(e, t, None, mid); Bm = z01.metrics(e, t, mid, None); sc = z01.sc4(e)
    y = t.groupby(t.et.dt.year).pnl.sum().round(0).astype(int).tolist()
    print(f"{lab}: net {f['net']:.0f} (%{f['pct']:.0f}) n {f['n']} PF {f['PF']:.2f} kazanma %{f['win']:.0f} | MaxDD {f['MaxDD']:.0f} P95/98/99 {f['P95']:.0f}/{f['P98']:.0f}/{f['P99']:.0f} | Calmar {f['Calmar']:.2f} Sharpe {f['Sharpe']:.2f} Sortino {f['Sortino']:.2f} Ulcer {f['Ulcer']:.0f} | SC4 {sc['SC4']:.3f} A95 {sc['A95']:.0f}g | σ10dk {f['s10']:.2f} σ1s {f['s1h']:.2f} | MAE P90 %{f['maeP90']:.1f} maks %{f['maeMax']:.1f} | IS {A['net']:.0f} OOS {Bm['net']:.0f} | +ay %{f['posM']:.0f} | long {t[t.side>0].pnl.sum():.0f} short {t[t.side<0].pnl.sum():.0f} (bekçi {int(t.guard.sum())}) | yıllar {y}")
    w = t[(t.xt >= '2026-11-23 06:00') ]
    print('  23 Kas → son (TR):', '; '.join(f"{(r.et+pd.Timedelta('3h')):%d %H:%M}→{(r.xt+pd.Timedelta('3h')):%d %H:%M} {'L' if r.side>0 else 'S'} {r.ep:.1f}→{r.xp:.1f} {r.pnl:+.1f}$" for r in w.itertuples()) or 'işlem yok')
    if spkX is None:
        t.to_csv('/home/claude/pine_hello_world/ZEC/research/R3_1dk_trades_engine.csv', index=False)
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        m30 = e.resample('30min').last().dropna(); dd = m30.cummax() - m30
        fig, ax = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True, gridspec_kw={'height_ratios': [3, 1.2]})
        ax[0].plot(m30.index, m30.values, '#1f6feb', lw=1); ax[0].set_title('ZEC 1M R3 (canlı sürüş öncelikli seçim) — 30 dk MTM equity (USDT, sabit 500, araştırma motoru)')
        ax[0].axvline(mid, color='#888', ls='--', lw=.8); ax[1].fill_between(dd.index, -dd.values, color='#d1242f', lw=0); ax[1].set_ylabel('MTM DD $')
        for x_ in ax: x_.grid(alpha=.3)
        plt.tight_layout(); plt.savefig('/home/claude/pine_hello_world/ZEC/research/R3_1dk_equity_dd.png', dpi=110)
