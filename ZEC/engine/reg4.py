import sys, json, itertools; sys.path.insert(0, '/home/claude/pine_hello_world/ZEC/engine')
import numpy as np, pandas as pd
exec(open('/home/claude/zecwork/reg3.py').read().split('rows = []')[0])          # aynı veri/kural kurucusu
def build2(Ath, thr, keep=None, drop=()):
    R = np.where(adx >= Ath, di, np.where(adx < 20, 0, 9)); L = np.zeros(n); Sh = np.zeros(n); ch = []
    for k, v in S.items():
        if (keep is not None and k not in keep) or k in drop: continue
        for side in (1, -1):
            for lab, val in (('UP', 1), ('DN', -1), ('YAT', 0)):
                m = (v == side) & (R == val)
                if m[:mid].sum() < 200: continue
                if np.nanmean((f240 * side)[:mid][m[:mid]]) >= thr:
                    ch.append(f'{k}{"L" if side > 0 else "S"}@{lab}'); (L if side > 0 else Sh)[m] += 1
    return L, Sh, ch
def go(Ath=20, thr=25, kL=2, kS=99, ts=8, tH=480, keep=None, drop=(), full=False):
    L, Sh, ch = build2(Ath, thr, keep, drop); tL = L >= kL; tS = (Sh >= kS) & ~tL
    eq, T = z02.sim4(F['o'], F['h'], F['l'], c, F['atr'], z, z, AT4 * 4, tL.astype(np.bool_), tS.astype(np.bool_), float(ts), int(tH), 99.0)
    t = pd.DataFrame(T, columns=['ei', 'xi', 'side', 'ep', 'xp', 'pnl', 'mae', 'own']); t['et'] = B.index[t.ei.astype(int)]
    return pd.Series(eq, index=B.index), t, ch
PINE = ['X17_basis', 'P09_spike', 'P07_rsi', 'P08_zema', 'P10_bb', 'P12_oiLiq', 'P11_oiCont', 'P15_resid', 'P04_htf4h']
def line(lab, e, t):
    f = z01.metrics(e, t); a = z01.metrics(e, t, None, midT); b = z01.metrics(e, t, midT, None); s = z01.sc4(e)
    y = t.groupby(t.et.dt.year).pnl.sum().round(0).astype(int).tolist()
    print(f"{lab}: net {f['net']:.0f} (%{f['pct']:.0f}) n {f['n']} PF {f['PF']:.2f} win {f['win']:.0f}% | MaxDD {f['MaxDD']:.0f} P95/98/99 {f['P95']:.0f}/{f['P98']:.0f}/{f['P99']:.0f} | Calmar {f['Calmar']:.2f} Sharpe {f['Sharpe']:.2f} Sortino {f['Sortino']:.2f} | SC4 {s['SC4']:.3f} A95 {s['A95']:.0f}g | σ10dk {f['s10']:.2f} σ1s {f['s1h']:.2f} | MAE P90 {f['maeP90']:.1f}% max {f['maeMax']:.1f}% | IS {a['net']:.0f} OOS {b['net']:.0f} | +ay %{f['posM']:.0f} | yıllar {y}")
    return f
e, t, ch = go(); f0 = line('TÜM VERİ (balina+OI+top trader)', e, t); e0, t0 = e, t
e, t, chp = go(keep=PINE); line('YALNIZ PINE-TAŞINIR', e, t); print('  Pine kuralları:', ', '.join(chp))
e, t, _ = go(kS=2); line('TÜM VERİ + SHORT (kS2)', e, t); print('  short net', round(t[t.side < 0].pnl.sum()))
print('ABLASYON (aile çıkar → net, MaxDD):')
for fam in [('ttposK4h', 'ttposK1g'), ('whaleT', 'R18_whale'), ('X17_basis',), ('R21_glb',), ('P09_spike',), ('P07_rsi', 'P08_zema', 'P10_bb'), ('P12_oiLiq', 'P11_oiCont'), ('P15_resid',), ('P04_htf4h',)]:
    e, t, _ = go(drop=fam); x = e.resample('30min').last().dropna(); dd = (x.cummax() - x).max()
    print(f'  −{"+".join(fam):22s} net {e.iloc[-1]:7.0f} ({e.iloc[-1] - e0.iloc[-1]:+6.0f})  MaxDD {dd:5.0f}')
nb = []
for A_, kL, ts, tH in itertools.product([20, 22, 25], [2, 3], [6, 8, 10], [360, 480, 720]):
    e, t, _ = go(A_, 25, kL, 99, ts, tH); x = e.values[::15]; nb.append((e.iloc[-1], (np.maximum.accumulate(x) - x).max()))
nb = np.array(nb); print('KOMŞU', len(nb), 'pozitif %', (nb[:, 0] > 0).mean() * 100, 'net medyan', np.median(nb[:, 0]).round(0), 'P10', np.percentile(nb[:, 0], 10).round(0), '| MaxDD medyan', np.median(nb[:, 1]).round(0))
rng = np.random.default_rng(3); L, Sh, _ = build2(20, 25); tL = L >= 2; k = int(tL.sum()); mon = []
for _ in range(100):
    r = np.zeros(n, np.bool_); r[rng.choice(np.arange(1000, n - 500), size=k, replace=False)] = True
    eq, _ = z02.sim4(F['o'], F['h'], F['l'], c, F['atr'], z, z, AT4 * 4, r, z, 8.0, 480, 99.0); mon.append(eq[-1])
print('MONKEY (aynı sayıda aktif bar, rastgele long): P50/P95/P99', np.percentile(mon, [50, 95, 99]).round(0), 'p=', (np.array(mon) >= e0.iloc[-1]).mean())
t0.to_csv('/home/claude/pine_hello_world/ZEC/research/R1_trades_engine.csv', index=False)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
m30 = e0.resample('30min').last().dropna(); dd = m30.cummax() - m30
fig, ax = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True, gridspec_kw={'height_ratios': [3, 1.2]})
ax[0].plot(m30.index, m30.values, '#1f6feb', lw=1); ax[0].set_title('ZEC 2M REJİM OYU R1 — 30 dk MTM equity (USDT, sabit 500, araştırma motoru, balina+OI+top trader)')
ax[0].axvline(midT, color='#888', ls='--', lw=.8); ax[1].fill_between(dd.index, -dd.values, color='#d1242f', lw=0); ax[1].set_ylabel('MTM DD $')
for x_ in ax: x_.grid(alpha=.3)
plt.tight_layout(); plt.savefig('/home/claude/pine_hello_world/ZEC/research/R1_equity_dd.png', dpi=110)
