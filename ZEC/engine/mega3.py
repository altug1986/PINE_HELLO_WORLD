# X1 ailesi + portföy farkındalıklı dikkat kapıları (XMR coşku / BAT long / PEPE short) + dar alan fitil girişi; amaç lineerlik + 9.500
exec(open('/home/claude/zecwork/mega1.py').read().split("W = {")[0])
import importlib; importlib.reload(z03)
def peer1m(sym, fs, tcol):
    d = peerfeat.rd(fs, tcol, sym).close; return d.reindex(idx).ffill().values
import glob as _g
BAT = peer1m('BATUSDT', sorted(_g.glob('/home/claude/peer/BATUSDT_1m_FULL*')), 'open_time_ms'); PEP = peer1m('1000PEPEUSDT', ['/home/claude/peer/1000PEPEUSDT_1m_FULL_20230505_THROUGH_20261119.csv.gz'], 'open_time_ms')
def ema(x, n_): return pd.Series(x).ewm(span=n_, adjust=False).mean().values
def r24(x): return np.r_[np.full(1440, np.nan), x[1440:] / x[:-1440] - 1] * 100
xr24 = r24(xr); BATup = np.nan_to_num((BAT > ema(BAT, 1440)) & (r24(BAT) > 0)).astype(bool); PEPdn = np.nan_to_num((PEP < ema(PEP, 1440)) & (r24(PEP) < 0)).astype(bool)
print('kapsama: BATup %', round(BATup.mean() * 100, 1), 'PEPdn %', round(PEPdn.mean() * 100, 1), flush=True)
H5 = pd.Series(h).rolling(5).max().values; L5 = pd.Series(l).rolling(5).min().values; O5 = np.r_[np.full(4, np.nan), o[:-4]]; r5 = np.maximum(H5 - L5, 1e-9)
wkL = (np.minimum(O5, c) - L5) / r5; wkS = (H5 - np.maximum(O5, c)) / r5
a8 = g * 8 + K * 4 + Vh * 2 + A_; REG = {'A×G×XMR': (a8 * 2 + De) * 2 + Xb, 'A×G': a8 * 2 + De}
P = pd.read_pickle('/home/claude/port/portfoy_canli_10dk_utc.pkl').resample('30min').last(); i30 = idx[::30]
ps0 = pd.Timestamp('2026-09-01'); Pw = P[ps0:]; pidx = np.searchsorted(i30.values, Pw.index.values).clip(0, len(i30) - 1)
mb = np.searchsorted(idx.values, pd.date_range('2021-01-01', idx[-1], freq='MS').values); mb = np.r_[mb[mb > 0], n - 1]
s2 = p('2026-09-24'); rows = []; t0 = time.time()
def lin(x): t_ = np.arange(len(x)); return np.corrcoef(t_, x)[0, 1] ** 2 * np.sign(x[-1] - x[0])
for rn, R in REG.items():
    for thr in (15, 25):
        L, S = select(R, thr, 480)
        for kL, kS in itertools.product((2, 3), (2, 3)):
            for thX, cL, useB, cS, wf in itertools.product((3.0, 6.0), (0, 1, 2), (0, 1), (0, 1, 2), (0.0, 0.5)):
                if cL == 0 and (thX == 6.0 or useB == 1): continue
                aL = L >= kL; aS = (S >= kS) & ~aL; condL = (np.nan_to_num(xr24) >= thX) | (BATup if useB else False)
                eL = (L >= kL + cL * condL) & ((wkL >= wf) if wf > 0 else True); eS = (S >= kS + cS * PEPdn) & ((wkS >= wf) if wf > 0 else True)
                eL = np.asarray(eL, np.bool_); eS = np.asarray(eS, np.bool_)
                for ts, tH in itertools.product((24.0, 32.0, 40.0), (960, 1920, 2880)):
                    eq, T = z03.sim11(o, h, l, c, a, aL, aS, eL, eS, ts, tH, spk, 60)
                    if len(T) < 1000 or eq[-1] < 6500: continue
                    x30 = eq[::30]; dd = np.maximum.accumulate(x30) - x30; mp = np.diff(np.r_[0, eq[mb]]); m12 = mp[-13:-1]
                    zc = Pw.values + (x30[pidx] - x30[pidx[0]]) * 200 / 500
                    rows.append(dict(rej=rn, thr=thr, kL=kL, kS=kS, thX=thX, cL=cL, BAT=useB, cS=cS, fitil=wf, ts=ts, tH=tH, n=len(T), net=eq[-1], IS=eq[mid] - eq[0], OOS=eq[-1] - eq[mid],
                                     MaxDD=dd.max(), DDp99=np.percentile(dd, 99), R2=lin(x30), ayMin=mp[:-1].min(), ayStd=mp[:-1].std(), bant=((mp[:-1] >= -100) & (mp[:-1] <= 300)).mean() * 100,
                                     ayPoz=(mp[:-1] > 0).mean() * 100, son12ayMin=m12.min(), son2ay=eq[-1] - eq[s2], son2ayR2=lin(eq[s2::30]), Kasım=eq[-1] - eq[p('2026-11-01')],
                                     portR2=lin(zc), portNet=zc[-1] - zc[0], short=T[T[:, 2] < 0, 5].sum()))
    print(rn, round(time.time() - t0), len(rows), flush=True)
Rr = pd.DataFrame(rows); Rr.to_csv('/home/claude/pine_hello_world/ZEC/research/LINEER_9500_1DK.csv', index=False); pd.set_option('display.width', 300)
f = Rr[(Rr.IS > 0) & (Rr.OOS > 0)]
print('\nnet ≥ 9.500:', (f.net >= 9500).sum(), '· en yüksek net', round(f.net.max()))
print(f.sort_values('net', ascending=False).head(10).round(2).to_string(index=False))
f2 = f[(f.net >= 8000) & (f.MaxDD <= 350)]; f2 = f2.assign(skor=f2.net / f2.DDp99 * f2.R2 * f2.bant / 100 * (f2.son2ayR2.clip(0) + 0.5))
print('\nnet ≥ 8.000 & MaxDD ≤ 350 → lineerlik skoru:', len(f2)); print(f2.sort_values('skor', ascending=False).head(12).round(2).to_string(index=False))
