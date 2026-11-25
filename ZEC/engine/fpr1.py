# FAILED PRESSURE REVERSAL olay çalışması: geçerli sanal sinyal → N bar ulaşamadı + karşı kırılım (+OI artışı) → ters işlem
exec(open('/home/claude/zecwork/vbar1.py').read().split("eq, T = sim(X0); ad('V2.7 baz', eq, T)")[0])
AL4, AS4, EL4, ES4 = X0[0][1:5], X0[1][1:5], X0[2][1:5], X0[3][1:5]
aL = AL4.any(0); aS = AS4.any(0); dL = aL & ~aS; dS = aS & ~aL
p3 = lambda x: x & np.r_[False, x[:-1]] & np.r_[False, False, x[:-2]]
gL = p3(dL) & (AL4 & EL4).any(0); gS = p3(dS) & (AS4 & ES4).any(0)
sig = np.where(gL, 1, np.where(gS, -1, 0)); on = (sig != 0) & (np.r_[0, sig[:-1]] != sig)
ev = np.flatnonzero(on); ev = ev[ev < n - 1000]; dv = sig[ev]
keep = [];  last = -10 ** 9
for e in ev:
    if e - last >= 12: keep.append(e); last = e
ev = np.array(keep); dv = sig[ev]; print('sanal sinyal:', len(ev), '· long', (dv > 0).sum(), '· short', (dv < 0).sum(), flush=True)
oi = M.oi_zec.reindex(idx).ffill().values; oi = np.r_[np.full(5, np.nan), oi[:-5]]
hmax = lambda e, N: np.array([h[x + 1:x + N + 1].max() for x in e]); lmin = lambda e, N: np.array([l[x + 1:x + N + 1].min() for x in e])
cost = 0.0008 * 1e4 + 2 * 0.01 / c[ev] * 1e4   # bps
rows = []; rng = np.random.default_rng(3)
for N in (3, 6, 9, 12):
    HM = hmax(ev, N); LM = lmin(ev, N); P = c[ev]; A = a[ev]; PN = c[ev + N]
    mfe = np.where(dv > 0, HM - P, P - LM); prog = (PN - P) * dv; doi = (oi[ev + N] / oi[ev] - 1) * 100
    for dk, x in [('%', v) for v in (0.15, 0.30, 0.50, 0.75)] + [('ATR', v) for v in (0.25, 0.5, 1.0)]:
        X_ = P * x / 100 if dk == '%' else A * x
        fail = mfe < X_; brk = prog <= -X_
        for cn, cond in (('yalnız ulaşamadı', fail), ('ulaşamadı+karşı kırılım', fail & brk), ('…+OI arttı (tuzak)', fail & brk & (doi > 0)), ('…+OI azaldı', fail & brk & (doi <= 0))):
            for H in (60, 240):
                sel = cond & (ev + N + H < n); e2 = ev[sel] + N; d2 = -dv[sel]
                r = (c[e2 + H] / c[e2] - 1) * d2 * 1e4 - cost[sel]
                allr = (c[ev + N + H] / c[ev + N] - 1) * (-dv) * 1e4 - cost        # kontrol: koşulsuz ters
                isel = ev[sel] < mid
                if len(r) < 30: continue
                mk = np.array([((c[np.minimum(rr + H, n - 1)] / c[rr] - 1) * d2 * 1e4 - cost[sel]).mean() for rr in [rng.integers(1000, n - 1000 - H, len(r)) for _ in range(300)]])
                rows.append(dict(N=N, mesafe=f'{x}{dk}', koşul=cn, H=H, n=len(r), ortBps=r.mean(), kontrolBps=allr.mean(), monkeyP=(mk >= r.mean()).mean(), IS_bps=r[isel].mean(), OOS_bps=r[~isel].mean(),
                                 IS_n=isel.sum(), OOS_n=(~isel).sum(), net500=(r * 0.05).sum(), win=(r > 0).mean() * 100))
R = pd.DataFrame(rows); R.to_csv('/home/claude/pine_hello_world/ZEC/research/FPR_OLAY_1DK.csv', index=False); pd.set_option('display.width', 250)
print('\nkoşul bazında ort. (tüm N/mesafe):'); print(R.groupby(['koşul', 'H'])[['n', 'ortBps', 'IS_bps', 'OOS_bps', 'net500']].mean().round(1).to_string())
g = R[(R.IS_bps > 0) & (R.OOS_bps > 0) & (R.monkeyP < 0.05)]; print('\nIS>0 & OOS>0 & monkey p<0.05:', len(g), '/', len(R))
print(g.sort_values('net500', ascending=False).head(15).round(2).to_string(index=False))
print('\nN=6 ısı (ulaşamadı+karşı kırılım, H=240, ortBps):'); print(R[(R.koşul == 'ulaşamadı+karşı kırılım') & (R.H == 240)].pivot(index='mesafe', columns='N', values='ortBps').round(1).to_string())
