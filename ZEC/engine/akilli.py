# (1) AKILLI POMPA: 10 dk ≥%3 + hacim 3× sıçraması SADECE BTC de aynı anda yukarı baskıdaysa long (23 Kas'ı al, 25 Kas 18:01'i alma)
# (2) TV teknik derece klonu (1 s, kapanmış bar): ZEC ve BTC ikisi de "Al" iken short ısrarından vazgeç / long aç; ölü kedi bağlamı seçenekli
exec(open('/home/claude/zecwork/pompa.py').read().split("rows = [chain(XB")[0])
import sys; sys.path.insert(0, '/home/claude/zecwork'); from tvrating import rating
s = open('/home/claude/zecwork/kanama.py').read(); a0 = s.index('@njit(cache=False)\ndef rec30'); b0 = s.index('\n', s.index('    return', a0)); exec(s[a0:b0 + 1])
btc = pd.Series(Zb['btc'].astype(float)).ffill().values
def path(X):
    eqb, Tb = sim(X); P_ = np.zeros(n, np.int8); bXp = np.full(n, np.nan); bXd = np.zeros(n, np.int8)
    for r in Tb: P_[int(r[0]):int(r[1])] = int(r[2]); bXp[int(r[1])] = r[4]; bXd[int(r[1])] = int(r[2])
    vp = np.zeros(n, np.int8); vx = np.zeros(n); vn = np.zeros(n, np.bool_); TR = np.zeros((20000, 4))
    run29(np.asarray(o, float), np.asarray(h, float), np.asarray(l, float), cc, P_, bXp, bXd, R10, 0.01, vp, vx, vn, TR)
    e, nt, _ = f3E(np.asarray(o, float), np.asarray(h, float), np.asarray(l, float), cc, 0.01, vp, vx, vn, np.ones(n, np.bool_), EVT, 1440, 0.08, np.zeros(n, np.int64), 0, 0.0)
    return e, Tb, vp, vx, vn
def met(e, lab, **kw):
    x = e[::30]; d = np.maximum.accumulate(x) - x
    return dict(v=lab, net=e[-1], IS=e[midI], OOS=e[-1] - e[midI], MaxDD=d.max(), DD99=np.percentile(d, 99), Ulcer=np.sqrt(((d / 5) ** 2).mean()), Calmar=e[-1] / yrs / d.max(), **kw)
d23 = (pd.Timestamp('2026-11-23') - pd.Timestamp('1970-01-01')) // pd.Timedelta('1ms'); i23 = np.searchsorted(tms, d23)
def pump23(Tb):
    pm = Tb[(Tb[:, 7] == 5) & (Tb[:, 0] >= i23)] if Tb.shape[1] > 7 else Tb[:0]
    return '; '.join(f"{pd.to_datetime(tms[int(r[0])], unit='ms') + pd.Timedelta('3h'):%d.%m %H:%M}→{pd.to_datetime(tms[int(r[1])], unit='ms') + pd.Timedelta('3h'):%H:%M} {r[5]:+.1f}$" for r in pm) or '-'
rows = []; P = {}
X1 = [x.copy() for x in XB]; X1[0][5] = False; X1[2][5] = False; e, Tb, *P['V3.1'] = path(X1); rows.append(met(e, 'V3.1 (POMPA yok)', pompa=0, pompaNet=0.0, kas23='-')); E31 = e
e, Tb, *P['V3.0'] = path(XB); pm = Tb[Tb[:, 7] == 5]; rows.append(met(e, 'V3.0 (POMPA, BTC şartsız)', pompa=len(pm), pompaNet=pm[:, 5].sum(), kas23=pump23(Tb)))
for lab, cond in (('BTC 10 dk ≥ +%0,3', btc / np.r_[np.full(10, np.nan), btc[:-10]] - 1 >= 0.003), ('BTC 10 dk ≥ +%0,5', btc / np.r_[np.full(10, np.nan), btc[:-10]] - 1 >= 0.005),
                  ('BTC 10 dk ≥ +%1', btc / np.r_[np.full(10, np.nan), btc[:-10]] - 1 >= 0.01), ('BTC 60 dk ≥ +%0,5', btc / np.r_[np.full(60, np.nan), btc[:-60]] - 1 >= 0.005), ('BTC 60 dk ≥ +%1', btc / np.r_[np.full(60, np.nan), btc[:-60]] - 1 >= 0.01)):
    pc = pmp & np.nan_to_num(cond, nan=False); X2 = [x.copy() for x in XB]; X2[0][5] = pc; X2[2][5] = pc; e, Tb, *P[lab] = path(X2); pm = Tb[Tb[:, 7] == 5]
    rows.append(met(e, f'AKILLI POMPA: {lab}', pompa=len(pm), pompaNet=pm[:, 5].sum(), kas23=pump23(Tb)))
R1 = pd.DataFrame(rows); b = R1.iloc[0]
for k in ('net', 'IS', 'OOS', 'MaxDD', 'DD99', 'Ulcer', 'Calmar'): R1['d' + k] = R1[k] - b[k]
pd.set_option('display.width', 320); pd.set_option('display.max_colwidth', 60); print(R1.round(2).to_string(index=False), flush=True)
R1.to_csv('/home/claude/zecwork/akilli_pompa.csv', index=False); np.save('/home/claude/zecwork/akilli_paths.npy', np.array([0]))
import pickle; pickle.dump({k: v for k, v in P.items()}, open('/home/claude/zecwork/akilli_paths.pkl', 'wb'))
