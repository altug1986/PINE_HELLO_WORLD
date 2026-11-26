# 25 Kas 18:01 long = POMPA motoru mu? POMPA'yı kaldır / teyitli yap → tüm zincir (orkestra → V2.9 uç TERS → V3.0 tasfiye dönüşü) yeniden koşar
import re, time; T0 = time.time()
exec(open('/home/claude/zecwork/dump28.py').read().split("posA = np.zeros(n, np.int8)")[0])
from numba import njit
for fn, name in (('/home/claude/zecwork/pffilt.py', 'run29'), ('/home/claude/zecwork/adxfilt.py', 'f3E')):
    s = open(fn).read(); a0 = s.index(f'@njit(cache=False)\ndef {name}'); b0 = s.index('\n', s.index('    return', a0)); exec(s[a0:b0 + 1])
cc = np.asarray(c, float); R10 = 100 - 100 / (1 + pd.Series(np.r_[0, np.diff(cc)]).clip(lower=0).ewm(alpha=0.1, adjust=False).mean() / pd.Series(np.r_[0, np.diff(cc)]).clip(upper=0).abs().ewm(alpha=0.1, adjust=False).mean())
Zb0 = np.load("/home/claude/zecwork/base28.npz"); tms = Zb0["tms"]; R10 = R10.values; ixh = pd.to_datetime(tms, unit="ms"); hc = pd.Series(cc, ixh).resample('1h').last(); sd = np.log(hc).diff().rolling(720).std(ddof=0)
s_ = pd.Series(sd.values, sd.index + pd.Timedelta('1h')); jj = np.searchsorted(s_.index.values, ixh.values, side='right') - 1; SDH = s_.values[np.maximum(jj, 0)]; SDH[jj < 0] = np.nan
Zb = np.load('/home/claude/zecwork/base28.npz'); oiT = pd.Series(Zb['oi'].astype(float)).ffill().values; lc = np.log(cc)
def lg(x, k): return x - np.r_[np.full(k, np.nan), x[:-k]]
EVT = np.nan_to_num((lg(lc, 60) / SDH <= -3.5) & (np.exp(lg(np.log(oiT * cc), 60)) - 1 <= -0.02), nan=False).astype(np.bool_)
midI = int(Zb['mid']); yrs = (tms[-1] - tms[0]) / (365.25 * 86400000)
def chain(X, lab):
    eqb, Tb = sim(X); P_ = np.zeros(n, np.int8); bXp = np.full(n, np.nan); bXd = np.zeros(n, np.int8)
    for r in Tb: P_[int(r[0]):int(r[1])] = int(r[2]); bXp[int(r[1])] = r[4]; bXd[int(r[1])] = int(r[2])
    vp = np.zeros(n, np.int8); vx = np.zeros(n); vn = np.zeros(n, np.bool_); TR = np.zeros((20000, 4))
    e29, K = run29(np.asarray(o, float), np.asarray(h, float), np.asarray(l, float), cc, P_, bXp, bXd, R10, 0.01, vp, vx, vn, TR)
    e, nt, _ = f3E(np.asarray(o, float), np.asarray(h, float), np.asarray(l, float), cc, 0.01, vp, vx, vn, np.ones(n, np.bool_), EVT, 1440, 0.08, np.zeros(n, np.int64), 0, 0.0)
    x = e[::30]; d = np.maximum.accumulate(x) - x; pm = Tb[Tb[:, 7] == 5] if Tb.shape[1] > 7 else Tb[:0]
    return dict(v=lab, baz=eqb[-1], V29=e29[-1], net=e[-1], IS=e[midI], OOS=e[-1] - e[midI], MaxDD=d.max(), DD99=np.percentile(d, 99), Ulcer=np.sqrt(((d / 5) ** 2).mean()), Calmar=e[-1] / yrs / d.max(), işlem=nt, pompaİşlem=len(pm), pompaNet=pm[:, 5].sum() if len(pm) else 0.0)
rows = [chain(XB, 'V3.0 (POMPA var)')]; print(rows[-1], f'{time.time() - T0:.0f} sn', flush=True)
X1 = [x.copy() for x in XB]; X1[0][5] = False; X1[2][5] = False; rows.append(chain(X1, 'POMPA yok'))
for k in (3, 5, 10):
    pc = np.zeros(n, np.bool_); pc[k:] = pmp[:-k] & (cc[k:] >= cc[:-k]); X2 = [x.copy() for x in XB]; X2[0][5] = pc; X2[2][5] = pc; rows.append(chain(X2, f'POMPA teyitli: {k} bar sonra fiyat tetik kapanışının üstündeyse'))
R = pd.DataFrame(rows); b = R.iloc[0]
for k in ('net', 'IS', 'OOS', 'MaxDD', 'DD99', 'Ulcer', 'Calmar'): R['d' + k] = R[k] - b[k]
pd.set_option('display.width', 260); print(R.round(2).to_string(index=False)); R.to_csv('/home/claude/zecwork/pompa.csv', index=False); print(f'{time.time() - T0:.0f} sn')
