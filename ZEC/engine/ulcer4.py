# E0 tutma süresi / stop / kapalı — tam motor (V2.8 gölge) + V2.9 ters katmanı
exec(open('/home/claude/zecwork/trend2.py').read().split("eq, T = sim(X0); ad('V2.7 baz', eq, T); out[-1]")[0])
import numba
stt = state2(200, 55, 20, 0.15, 4); XB = [x.copy() for x in X0]; XB[1][1:5] &= ~(stt > 0)[None, :]; XB[3][1:5] &= ~(stt > 0)[None, :]
def rsi_(x, N):
    d = np.diff(x, prepend=x[0]); u = pd.Series(np.maximum(d, 0)).ewm(alpha=1 / N, adjust=False).mean().to_numpy(); v = pd.Series(np.maximum(-d, 0)).ewm(alpha=1 / N, adjust=False).mean().to_numpy()
    return 100 - 100 / (1 + u / np.where(v == 0, 1e-12, v))
R10 = np.nan_to_num(rsi_(c, 10), nan=50.0); TICK = 0.01; tms_ = np.asarray((idx - pd.Timestamp('1970-01-01')) // pd.Timedelta('1ms'), np.int64)
src = open('/home/claude/zecwork/ulcer2.py').read(); i0 = src.index('@numba.njit(cache=True)\ndef ov8'); i1 = src.index('def ulc(e)')
exec(src[i0:i1].replace('cache=True', 'cache=False'))
F0_ = np.zeros(n, np.bool_); rows = []
def run(lab, TSx, THx, X=XB):
    eqb, Tb = z03.simN9(o, h, l, c, a, *X, TSx, THx, spk, 60, 3, byp, 0.10, False, F0, F0, 0, 0.0, False)
    ps = np.zeros(n, np.int8); bE = np.zeros(n); bX = np.full(n, np.nan); bD = np.zeros(n, np.int8)
    for r in Tb: ei, xi = int(r[0]), int(r[1]); ps[ei:xi] = int(r[2]); bE[ei:xi] = r[3]; bX[xi] = r[4]; bD[xi] = int(r[2])
    e = ov8(o, h, l, c, ps, bE, bX, bD, R10, F0_, TICK); x = e[::30]; d_ = np.maximum.accumulate(x) - x; ue = (pd.Series(e).cummax() - pd.Series(e)).to_numpy() >= 100
    rows.append(dict(v=lab, n=len(Tb), net=e[-1], IS=e[mid], OOS=e[-1] - e[mid], Ulcer=np.sqrt(((d_ / 5) ** 2).mean()), sualtı100=ue.mean() * 100, MaxDD=d_.max(), DD99=np.percentile(d_, 99),
                     son300=e[-1] - e[np.searchsorted(tms_, tms_[-1] - 300 * 86400000)], E0net=Tb[Tb[:, 7] == 1, 5].sum()))
    print(pd.DataFrame([rows[-1]]).round(2).to_string(index=False, header=len(rows) == 1), flush=True)
run('V2.9 (E0 TH3840 TS28)', TS6, TH6)
for th in (960, 1920, 2880):
    t_ = TH6.copy(); t_[1] = th; run(f'E0 tutma {th}', TS6, t_)
for ts in (14.0, 20.0, 40.0):
    s_ = TS6.copy(); s_[1] = ts; run(f'E0 stop {ts:g}×ATR', s_, TH6)
X_ = [x.copy() for x in XB]
for q_ in range(4): X_[q_][1] = False
run('E0 KAPALI', TS6, TH6, X_)
pd.DataFrame(rows).to_pickle('/home/claude/zecwork/ulcer4.pkl')
