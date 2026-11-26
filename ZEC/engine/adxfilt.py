# İSTEK: yönü takip et — fiyat ADX + balina (top trader POZİSYON oranı) ADX > 20 ile yön; ATR çok yüksekken BTC fiyat hareketinden destek. Taban: V3.0 (V2.9 + tasfiye dönüşü)
src = open('/home/claude/zecwork/pffilt.py').read(); exec(src.split("rows = [ev(")[0])
import glob
M = pd.concat([pd.read_parquet(f, columns=['tt_pos_ls', 'btc_close']) for f in sorted(glob.glob('/home/claude/master/ZECUSDT_MASTER_1m_SYNC_*.parquet'))], ignore_index=True)
def lagd(x, k): return x - np.r_[np.full(k, np.nan), x[:-k]]
def rma(x, N): return S(x).ewm(alpha=1 / N, adjust=False).mean().to_numpy()
ixh = pd.to_datetime(tms, unit='ms')
def ohlc(x, rule):
    s = S(x, ixh); return pd.DataFrame({'o': s.resample(rule).first(), 'h': s.resample(rule).max(), 'l': s.resample(rule).min(), 'c': s.resample(rule).last()}).dropna()
def ohlcZ(rule): return pd.DataFrame({'o': S(o, ixh).resample(rule).first(), 'h': S(h, ixh).resample(rule).max(), 'l': S(l, ixh).resample(rule).min(), 'c': S(c, ixh).resample(rule).last()}).dropna()
def dmi(D, N=14):
    H, L, C = D.h.values, D.l.values, D.c.values; pc = np.r_[C[0], C[:-1]]; tr = np.maximum(H - L, np.maximum(np.abs(H - pc), np.abs(L - pc)))
    up = np.r_[0, np.diff(H)]; dn = np.r_[0, -np.diff(L)]; at = rma(tr, N)
    pdi = 100 * rma(np.where((up > dn) & (up > 0), up, 0.0), N) / np.where(at > 0, at, np.nan); mdi = 100 * rma(np.where((dn > up) & (dn > 0), dn, 0.0), N) / np.where(at > 0, at, np.nan)
    adx = rma(np.nan_to_num(100 * np.abs(pdi - mdi) / np.where(pdi + mdi == 0, 1, pdi + mdi)), N); return adx, pdi, mdi, at / C * 100
def mapH(v, D, td): s_ = S(v, D.index + td); j = np.searchsorted(s_.index.values, ixh.values, side='right') - 1; out = s_.values[np.maximum(j, 0)]; out[j < 0] = np.nan; return out
TF = {'15dk': ('15min', pd.Timedelta('15min')), '1s': ('1h', pd.Timedelta('1h')), '4s': ('4h', pd.Timedelta('4h'))}
PX, WH, BT = {}, {}, {}
wl = S(M.tt_pos_ls.values).ffill().values; bc = S(M.btc_close.values).ffill().values
for k, (rule, td) in TF.items():
    D = ohlcZ(rule); a, p_, m_, ap = dmi(D); PX[k] = [mapH(v, D, td) for v in (a, p_, m_, ap)]
    if k != '15dk':
        Dw = ohlc(wl, rule); a, p_, m_, _ = dmi(Dw); WH[k] = [mapH(v, Dw, td) for v in (a, p_, m_)]
        Db = ohlc(bc, rule); a, p_, m_, _ = dmi(Db); BT[k] = [mapH(v, Db, td) for v in (a, p_, m_)]
D1 = ohlcZ('1h'); _, _, _, ap1 = dmi(D1); ratio = ap1 / S(ap1).rolling(720, min_periods=240).median().values; ATRR = mapH(ratio, D1, pd.Timedelta('1h'))
# V3.0 tasfiye dönüş olayı (Pine birebir)
hc = S(c, ixh).resample('1h').last(); sdh = np.log(hc).diff().rolling(720).std(ddof=0); SDH = mapH(sdh.values, pd.DataFrame(index=sdh.index), pd.Timedelta('1h'))
oiT = S(Z['oi'].astype(float)).ffill().values; EVT = np.nan_to_num((lagd(np.log(c), 60) / SDH <= -3.5) & (np.exp(lagd(np.log(oiT * c), 60)) - 1 <= -0.02), nan=False).astype(np.bool_)
@njit(cache=False)
def f3E(o, h, l, c, tick, vpos, vxp, vnew, F, evt, H, stp, sig, Ht, stpT):
    """V2.9 sanal → giriş vetosu F (yeni girişte) → tasfiye dönüş yan motoru (baz FLAT) → seçmeli TREND takip motoru sig (baz FLAT, sinyal sürdükçe)."""
    n = len(c); gb = 0; real = 0; ep = 0.0; q = 0.0; rz = 0.0; sa = False; sb = 0; ss = 0.0; blk = 0; ta_ = 0; tb_ = 0; ts_ = 0.0; eq = np.zeros(n); nT = 0; nt = 0
    for i in range(1, n):
        if gb != 0 and gb == vpos[i - 1] and vxp[i] > 0: gb = 0
        if real != 0:
            xp = -1.0
            if sa and real == 1 and l[i] <= ss: xp = min(o[i], ss) - tick; sa = False; blk = sb + H
            elif ta_ != 0 and real == ta_ and ((real == 1 and l[i] <= ts_) or (real == -1 and h[i] >= ts_)): xp = (min(o[i], ts_) - tick) if real == 1 else (max(o[i], ts_) + tick); ta_ = 0
            elif (not sa) and ta_ == 0 and real == vpos[i - 1] and vxp[i] > 0: xp = vxp[i]
            if xp > 0: rz += q * (xp - ep) * real - 0.0004 * q * (ep + xp); real = 0; nt += 1
        v = vpos[i]
        if vnew[i]: gb = v if F[i] else 0
        elif gb != v: gb = 0
        if sa and i - sb >= H: sa = False
        if (not sa) and evt[i] and i >= blk and gb == 0 and ta_ == 0 and real == 0: sa = True; sb = i   # Pine: want == 0 ve realPos == 0
        if ta_ != 0 and (sig[i] != ta_ or i - tb_ >= Ht or gb != 0 or sa): ta_ = 0
        if ta_ == 0 and (not sa) and gb == 0 and sig[i] != 0 and Ht > 0: ta_ = sig[i]; tb_ = i; nT += 1
        tg = 1 if sa else (ta_ if ta_ != 0 else gb)
        if tg != real:
            if real != 0: xp = c[i] - tick * real; rz += q * (xp - ep) * real - 0.0004 * q * (ep + xp); real = 0; nt += 1
            if tg != 0:
                real = tg; ep = c[i] + tick * real; q = 500.0 / c[i]
                if sa: ss = ep * (1.0 - stp)
                elif ta_ != 0: ts_ = ep * (1.0 - ta_ * stpT)
        eq[i] = rz + (q * (c[i] - ep) * real - 0.0004 * q * ep if real != 0 else 0.0)
    return eq, nt, nT
ONE = np.ones(n, np.bool_); Z0 = np.zeros(n, np.int64)
e30, nt30, _ = f3E(o, h, l, c, TICK, vpos, vxp, vnew, ONE, EVT, 1440, 0.08, Z0, 0, 0.0); B = ev(e30, nt30, 0, 0, v='V3.0'); print(f"V3.0 taban (Pine sırası: realPos==0 şartı) {e30[-1]:.1f} | TV penceresi {e30[-1] - e30[np.searchsorted(tms, 1675382400000)]:.1f}")
dv = TR[:, 2].astype(int); rows = []
def gate(block):   # block: V2.9 giriş barlarında engellenecek mi
    F = np.ones(n, np.bool_); F[ei[block]] = False; return F
def run(lab, F=ONE, sig=Z0, Ht=0, stpT=0.0, **kw):
    e, nt_, nT = f3E(o, h, l, c, TICK, vpos, vxp, vnew, F, EVT, 1440, 0.08, sig, Ht, stpT); r = ev(e, nt_, 0, 0, v=lab, **kw); r['trendİşlem'] = nT; rows.append(r); return e
def opp(Dm, thr, d):   # trend (ADX ≥ thr) giriş yönüne TERS mi
    a, p_, m_ = Dm[0][ei], Dm[1][ei], Dm[2][ei]; up = p_ > m_; return np.nan_to_num((a >= thr) & (((d < 0) & up) | ((d > 0) & ~up)), nan=False)
for tf in ('15dk', '1s', '4s'):
    for thr in (20, 25, 30): run(f'F1 fiyat ADX {tf} ≥{thr} trende ters girişi engelle', gate(opp(PX[tf], thr, dv)))
for tf in ('1s', '4s'):
    for thr in (20, 25, 30): run(f'F2a balina pozisyon ADX {tf} ≥{thr} ters girişi engelle', gate(opp(WH[tf], thr, dv)))
    for tp, tw in itertools.product((20, 25), (20, 25)): run(f'F2 fiyat ADX {tf} ≥{tp} VE balina ADX ≥{tw} ikisi de ters → engelle', gate(opp(PX[tf], tp, dv) & opp(WH[tf], tw, dv)))
for k_, bt, bthr in itertools.product((1.5, 2.0, 3.0), ('1s', '4s'), (0, 20)):
    hi = np.nan_to_num(ATRR[ei] >= k_, nan=False); run(f'F4 ATR oranı ≥{k_} iken BTC {bt} yönü (ADX≥{bthr}) ters → engelle', gate(hi & opp(BT[bt], bthr, dv)))
for tf, thr, stpT, Ht in itertools.product(('1s', '4s'), (20, 25), (0.03, 0.05), (720, 1440)):
    a, p_, m_ = PX[tf][:3]; aw, pw, mw = WH[tf]; up = (a >= thr) & (p_ > m_) & (aw >= thr) & (pw > mw); dn = (a >= thr) & (m_ > p_) & (aw >= thr) & (mw > pw)
    sig = np.where(np.nan_to_num(up, nan=False), 1, np.where(np.nan_to_num(dn, nan=False), -1, 0)).astype(np.int64)
    run(f'F3 fiyat+balina ADX {tf} ≥{thr} aynı yön → FLAT iken o yöne gir (stop %{stpT * 100:g}, en çok {Ht // 60} s)', sig=sig, Ht=Ht, stpT=stpT)
R = pd.DataFrame(rows)
for k in ('net', 'IS', 'OOS', 'MaxDD', 'DD99', 'Ulcer', 'Calmar'): R['d' + k] = R[k] - B[k]
R['ikiYarı'] = (R.dIS > 0) & (R.dOOS > 0); R.to_pickle('/home/claude/zecwork/adxfilt.pkl'); pd.set_option('display.width', 280); pd.set_option('display.max_colwidth', 78)
print(f"{len(R)} varyant | iki yarı artı {R.ikiYarı.sum()} | kapı (≥{B['net'] * 1.06:.0f} & MaxDD≤{B['MaxDD']:.1f} & Ulcer≤{B['Ulcer']:.2f}): {((R.net >= B['net'] * 1.06) & (R.MaxDD <= B['MaxDD'] + 0.01) & (R.Ulcer <= B['Ulcer'])).sum()}")
R['aile'] = R.v.str[:2]; print(R.groupby('aile').agg(n=('dnet', 'size'), medNet=('dnet', 'median'), maxNet=('dnet', 'max'), ikiYarı=('ikiYarı', 'sum'), medDD=('dMaxDD', 'median')).round(1).to_string())
print(R.sort_values('dnet', ascending=False)[['v', 'işlem', 'trendİşlem', 'net', 'dnet', 'dIS', 'dOOS', 'MaxDD', 'DD99', 'Ulcer', 'Calmar']].head(12).round(1).to_string(index=False))
