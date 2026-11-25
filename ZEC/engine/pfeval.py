# Adaylar (pompalı/pompasız × ters katman ayarı) + portföyün zayıf anları (10 dk MTM akışı) ön şartı
exec(open('/home/claude/zecwork/overlay3.py').read().split("F0 = np.zeros(n, np.bool_); RS = ")[0])
B2 = np.load('/home/claude/zecwork/base29.npz'); F0 = np.zeros(n, np.bool_); R10 = np.nan_to_num(rsi(c, 10), nan=50.0)
def base_arrays(T_):
    ep_ = np.zeros(n); xp_ = np.full(n, np.nan); xd_ = np.zeros(n, np.int8); ps_ = np.zeros(n, np.int8)
    for r in T_:
        ei, xi = int(r[0]), int(r[1]); ep_[ei:xi] = r[3]; xp_[xi] = r[4]; xd_[xi] = int(r[2]); ps_[ei:xi] = int(r[2])
    return ps_, ep_, xp_, xd_
ix_ = pd.to_datetime(tms, unit='ms')
Pp = pd.read_pickle('/home/claude/port/portfoy_canli_10dk_utc.pkl'); Pp = Pp[~Pp.index.duplicated()]
pv = Pp.values; legs = []; mode = 0; hi = pv[0]; hi_i = 0; lo = pv[0]; lo_i = 0
for i in range(1, len(pv)):
    if mode >= 0:
        if pv[i] > hi: hi, hi_i = pv[i], i
        if hi - pv[i] >= 40: mode = -1; lo, lo_i = pv[i], i
    if mode == -1:
        if pv[i] < lo: lo, lo_i = pv[i], i
        if pv[i] - lo >= 40: legs.append((Pp.index[hi_i], Pp.index[lo_i], hi - lo)); mode = 1; hi, hi_i = pv[i], i
if mode == -1: legs.append((Pp.index[hi_i], Pp.index[lo_i], hi - lo))
end_t = min(ix_[-1], Pp.index[-1]); W = {'genel': Pp.index[0], 'son300': end_t - pd.Timedelta(days=300), 'son90': end_t - pd.Timedelta(days=90)}
pD = Pp.resample('1D').last().diff().dropna(); pW = Pp.resample('W').last().diff().dropna()
uw = (Pp.cummax() - Pp) > 20                                                    # portföy zirvenin 20 $ altında (sualtı)
def pf(eqZ, lab):
    zs = pd.Series(eqZ, ix_); z10 = zs.reindex(Pp.index, method='ffill'); r = dict(model=lab)
    zD = zs.resample('1D').last().diff().reindex(pD.index); zW = zs.resample('W').last().diff().reindex(pW.index); dz10 = z10.diff()
    for k, t0 in W.items():
        lg = [(a, b, d) for a, b, d in legs if a >= t0]; g = np.array([z10.asof(b) - z10.asof(a) for a, b, d in lg]); loss = sum(d for *_, d in lg)
        dd_ = pD[pD.index >= t0]; ww = pW[pW.index >= t0]; kD = dd_ <= dd_.quantile(0.2); kW = ww <= ww.quantile(0.2)
        r[f'{k} çukur$'] = g.sum(); r[f'{k} çukur%+'] = (g > 0).mean() * 100; r[f'{k} kapama%'] = g.sum() / loss * 100
        r[f'{k} kötüGün$'] = zD.reindex(dd_.index)[kD].sum(); r[f'{k} kötüHafta$'] = zW.reindex(ww.index)[kW].sum(); r[f'{k} korGün'] = zD.reindex(dd_.index).corr(dd_)
        u = uw[uw.index >= t0]; r[f'{k} sualtı$'] = dz10[u.index][u.values].sum()
        comb = (Pp + 0.4 * z10)[Pp.index >= t0]; comb = comb - comb.iloc[0]; base_ = Pp[Pp.index >= t0] - Pp[Pp.index >= t0].iloc[0]
        r[f'{k} portMaxDD'] = (base_.cummax() - base_).max(); r[f'{k} port+ZEC200 MaxDD'] = (comb.cummax() - comb).max()
    return r
def core(eqZ, lab):
    e30 = eqZ[::30]; dd = np.maximum.accumulate(e30) - e30; t23 = np.searchsorted(tms, (pd.Timestamp('2023-02-06') - pd.Timestamp('1970-01-01')) // pd.Timedelta('1ms'))
    y = pd.Series(eqZ, ix_).resample('YE').last(); y = y.diff().fillna(y)
    return dict(model=lab, net=eqZ[-1], IS=eqZ[mid], OOS=eqZ[-1] - eqZ[mid], yakın=eqZ[-1] - eqZ[t23], son300=eqZ[-1] - eqZ[np.searchsorted(tms, tms[-1] - 300 * 86400000)],
                MaxDD=dd.max(), DD99=np.percentile(dd, 99), yıllar=' '.join(f'{v:.0f}' for v in y.values))
rowsC, rowsP = [], []
for bn, (eqb, Tb) in (('pompalı', (B2['eqP'], B2['TP'])), ('pompasız', (B2['eqN'], B2['TN']))):
    ps_, ep_, xp_, xd_ = base_arrays(Tb)
    cands = [('katmansız', None)] + [(f'TERS X94 Y{Y} sl%{sl * 100:g}', (94, Y, sl)) for Y in (65, 70) for sl in (0.02, 0.03)]
    for cn, prm in cands:
        if prm is None: e, P, nt = ov3(o, h, l, c, ps_, ep_, xp_, xd_, F0, F0, F0, F0, 240, 0.03, 99.0, trSh, False, 0.02, TICK)
        else:
            X, Y, sl = prm; e, P, nt = ov3(o, h, l, c, ps_, ep_, xp_, xd_, R10 >= X, R10 <= 100 - X, R10 <= Y, R10 >= 100 - Y, 240, 0.03, 99.0, trSh, True, sl, TICK)
        lab = f'{bn} · {cn}'; rowsC.append(core(e, lab)); rowsP.append(pf(e, lab)); print(lab, round(e[-1]), flush=True)
pd.set_option('display.width', 340); pd.set_option('display.max_columns', 60)
C_ = pd.DataFrame(rowsC); P_ = pd.DataFrame(rowsP); print(C_.round(1).to_string(index=False)); print()
for k in W: print(P_[['model'] + [x for x in P_.columns if x.startswith(k + ' ')]].round(2).to_string(index=False)); print()
C_.merge(P_, on='model').to_csv('/home/claude/pine_hello_world/ZEC/research/PORTFOY_ZAYIF_AN_ADAYLAR.csv', index=False)
