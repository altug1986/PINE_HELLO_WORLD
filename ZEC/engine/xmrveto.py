# XMR / BTC / sepet düşerken ZEC long vetosu (+ simetrik short vetosu) → V2.8 gölge → V2.9 ters katman → portföy ön şartı
exec(open('/home/claude/zecwork/trend2.py').read().split("eq, T = sim(X0); ad('V2.7 baz', eq, T); out[-1]")[0])
stt = state2(200, 55, 20, 0.15, 4); XB = [x.copy() for x in X0]; XB[1][1:5] &= ~(stt > 0)[None, :]; XB[3][1:5] &= ~(stt > 0)[None, :]
src3 = open('/home/claude/zecwork/pfeval.py').read().split("rowsC, rowsP = [], []")[0]
src3 = src3.replace("exec(open('/home/claude/zecwork/overlay3.py').read().split(\"F0 = np.zeros(n, np.bool_); RS = \")[0])", "")
ov3src = open('/home/claude/zecwork/overlay3.py').read(); i0 = ov3src.index('@numba.njit'); i1 = ov3src.index('def M(lab')
import numba
exec(ov3src[i0:i1])
import glob as _g
MM = pd.concat([pd.read_parquet(f, columns=['time', 'xmr_close', 'btc_close']) for f in sorted(_g.glob('/home/claude/master/ZECUSDT_MASTER_1m_SYNC_*.parquet'))])
MM.index = pd.to_datetime(MM.time, unit='ms') if MM.time.dtype.kind in 'iu' else pd.to_datetime(MM.time); MM = MM[~MM.index.duplicated()].reindex(idx).ffill()
xm = MM.xmr_close.values; bt = MM.btc_close.values
br_ = np.r_[0, c[1:] / c[:-1] - 1]; trSh = np.nan_to_num((pd.Series(br_).rolling(4320).mean() / pd.Series(br_).rolling(4320).std() * np.sqrt(4320)).to_numpy()); TICK = 0.01
def rsi_(x, N):
    d = np.diff(x, prepend=x[0]); u = pd.Series(np.maximum(d, 0)).ewm(alpha=1 / N, adjust=False).mean().to_numpy(); v = pd.Series(np.maximum(-d, 0)).ewm(alpha=1 / N, adjust=False).mean().to_numpy()
    return 100 - 100 / (1 + u / np.where(v == 0, 1e-12, v))
R10 = np.nan_to_num(rsi_(c, 10), nan=50.0); tms = np.asarray((idx - pd.Timestamp('1970-01-01')) // pd.Timedelta('1ms'), np.int64)
exec(src3.split("B2 = np.load")[0]); exec("def base_arrays" + src3.split("def base_arrays")[1].split("ix_ = pd.to_datetime")[0])
ix_ = idx; exec("Pp = pd.read_pickle" + src3.split("Pp = pd.read_pickle")[1])
def runV(vL, vS, lab):
    X = [x.copy() for x in XB]
    if vL is not None: X[0][1:5] &= ~vL[None, :]; X[2][1:5] &= ~vL[None, :]
    if vS is not None: X[1][1:5] &= ~vS[None, :]; X[3][1:5] &= ~vS[None, :]
    eqb, Tb = sim(X)
    ps_, ep_, xp_, xd_ = base_arrays(Tb)
    e, P, nt = ov3(o, h, l, c, ps_, ep_, xp_, xd_, R10 >= 94, R10 <= 6, R10 <= 70, R10 >= 30, 240, 0.03, 99.0, trSh, True, 0.02, TICK)
    r = core(e, lab); p_ = pf(e, lab); r.update({k: p_[k] for k in ('genel çukur$', 'son300 çukur$', 'son90 çukur$', 'son300 korGün', 'son300 kötüGün$', 'genel port+ZEC200 MaxDD', 'son300 port+ZEC200 MaxDD')})
    rows.append(r); print(pd.DataFrame([r]).round(1).to_string(index=False, header=len(rows) == 1), flush=True)
rows = []; out = []
runV(None, None, 'V2.9 (veto yok)')
def ret(x, k): return np.r_[np.full(k, np.nan), x[k:] / x[:-k] - 1] * 100
SRC = {'XMR': xm, 'BTC': bt}
for sn in ('XMR', 'BTC', 'sepet'):
    for k in (60, 240, 1440):
        r_ = (ret(xm, k) + ret(bt, k)) / 2 if sn == 'sepet' else ret(SRC[sn], k)
        for th in ((1, 2, 3) if k < 1440 else (3, 5, 8)):
            dn = np.nan_to_num(r_ <= -th).astype(np.bool_); upm = np.nan_to_num(r_ >= th).astype(np.bool_)
            runV(dn, None, f'{sn} {k}dk ≤ −%{th}: long yok')
            runV(dn, upm, f'{sn} {k}dk ±%{th}: long/short yok (simetrik)')
pd.DataFrame(rows).to_csv('/home/claude/pine_hello_world/ZEC/research/XMR_VETO_ADAYLAR.csv', index=False)
