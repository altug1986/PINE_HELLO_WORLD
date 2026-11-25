exec(open('/home/claude/zecwork/reg6_head.py').read())
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
z = np.zeros(n, np.int64); A4 = np.zeros(n)
K = bit(PF.btcC.values); Vh = bit(PF.relz.values); A_ = bit(PF.atrC.values)
RR = {'BTCkor×sıcak×ATRkor': (RG + 1) * 8 + K * 4 + Vh * 2 + A_, 'baz': RG + 1}
CAND = {'AKRAN-A': ('BTCkor×sıcak×ATRkor', 15, 480, 2, 3, 24.0, 960), 'AKRAN-B': ('BTCkor×sıcak×ATRkor', 15, 480, 2, 3, 24.0, 1920),
        'AKRAN-C': ('BTCkor×sıcak×ATRkor', 10, 480, 3, 3, 24.0, 960), 'AKRAN-D': ('BTCkor×sıcak×ATRkor', 15, 480, 2, 3, 16.0, 1920), 'R5-eski': ('baz', 15, 240, 2, 3, 16.0, 1920)}
t0 = idx[-1].normalize() - pd.Timedelta(days=30); i0 = np.searchsorted(idx.values, t0.to_datetime64()); days = pd.date_range(t0, idx[-1].normalize(), freq='1D')
di = np.searchsorted(idx.values, days.values); di = np.r_[di, n - 1]
S_ = []; DAY = {}; fig, ax = plt.subplots(figsize=(12, 5))
for nm, (rn, thr, hz, kL, kS, ts, tH) in CAND.items():
    L, S = select(RR[rn], thr, hz); aL = L >= kL; aS = (S >= kS) & ~aL
    eq, T = z03.sim10(o, h, l, c, a, aL, aS, ts, tH, spk, 60, z, A4, 0)
    w = eq[i0:] - eq[i0]; adv = eq.copy()
    for e_, x_, p_, ep in T[:, :4]:
        e_, x_ = int(e_), int(x_); s = slice(e_ + 1, x_ + 1); adv[s] = eq[s] + 500 / ep * p_ * ((l[s] if p_ > 0 else h[s]) - c[s])
    m = T[:, 1] >= i0; Tm = T[m]; x30 = w[::30]; d30 = np.maximum.accumulate(x30) - x30; d1 = (np.maximum.accumulate(eq[i0:]) - adv[i0:])
    dp = np.diff(eq[di]); DAY[nm] = dp; h1 = np.diff(eq[i0::60])
    S_.append(dict(sistem=nm, net=w[-1], yüzde=w[-1] / 5, işlem=len(Tm), long=Tm[Tm[:, 2] > 0, 5].sum(), short=Tm[Tm[:, 2] < 0, 5].sum(), kazanç=(Tm[:, 5] > 0).mean() * 100 if len(Tm) else 0,
                   MaxDD_30dk=d30.max(), MaxDD_barİçi=d1.max(), enKötüGün=dp.min(), enİyiGün=dp.max(), pozGün=f'{(dp > 0).sum()}/{len(dp)}', saatStd=h1.std(), maeMaks=Tm[:, 6].max() * 100 if len(Tm) else 0,
                   açıkPoz='LONG' if (T[-1, 1] < n - 1 and False) else ''))
    ax.plot(idx[i0::30], x30, lw=1.2, label=f'{nm}: {w[-1]:+.0f} $')
ax.axhline(0, color='k', lw=0.6); ax.legend(loc='upper left'); ax.grid(alpha=.3); ax.set_ylabel('Net $ (sabit 500 $, 30 dk MTM)')
ax.set_title(f'ZEC 1 dk — son 30 gün ({days[0].date()} → {idx[-1]:%Y-%m-%d %H:%M} UTC), araştırma motoru'); plt.tight_layout(); plt.savefig('/home/claude/zecwork/ZEC_SON30.png', dpi=110)
pd.set_option('display.width', 250); print(pd.DataFrame(S_).drop(columns='açıkPoz').round(1).to_string(index=False))
Dd = pd.DataFrame(DAY, index=[f'{d:%m-%d}' for d in days]); Dd.loc['TOPLAM'] = Dd.sum(); print(Dd.round(1).to_string())
