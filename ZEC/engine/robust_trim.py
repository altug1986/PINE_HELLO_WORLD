# Seçilen: RSI10 X94 Y70 geri240 devam%3 sh<99 TERS sl%2 (seçim kuralı: IS en yüksek) — gecikme, monkey, MC, DSR, komşu
exec(open('/home/claude/zecwork/overlay3.py').read().split("F0 = np.zeros(n, np.bool_); RS = ")[0])
from scipy.stats import norm, skew, kurtosis
F0 = np.zeros(n, np.bool_); R10 = np.nan_to_num(rsi(c, 10), nan=50.0); rng = np.random.default_rng(11)
def run(trL, trS, riL, riS, fl=True, tb=240, cont=0.03, sh=99.0, fsl=0.02): return ov3(o, h, l, c, posA, bEp, bXp, bXd, trL, trS, riL, riS, tb, cont, sh, trSh, fl, fsl, TICK)
eq0, P0, _ = run(F0, F0, F0, F0, False); C = M('kontrol', eq0, P0, 0)
eq1, P1, nt1 = run(R10 >= 94, R10 <= 6, R10 <= 70, R10 >= 30); S1 = M('SEÇİLEN', eq1, P1, nt1)
out = []
def row(lab, r): out.append(dict(v=lab, net=r['net'], dNet=r['net'] - C['net'], dIS=r['dIS'] - C['dIS'], dOOS=r['dOOS'] - C['dOOS'], dCalm=r['dCalm'] - C['dCalm'], d300=r['d300'] - C['d300'], MaxDD=r['MaxDD'], DD99=r['DD99'], DD95=r['DD95'], Sh=r['Sh'], n=r['nTrim']))
row('kontrol (V2.8 yeniden üretim)', C); row('SEÇİLEN RSI10 94/70 TERS %2', S1)
for k in (1, 2):
    Rk = np.r_[np.full(k, 50.0), R10[:-k]]; e, P, nt = run(Rk >= 94, Rk <= 6, Rk <= 70, Rk >= 30); row(f'gecikme {k} bar', M('', e, P, nt))
e, P, nt = run(R10 >= 94, R10 <= 6, R10 <= 70, R10 >= 30, sh=2.0); row('komşu: sh<2', M('', e, P, nt))
for X, Y in ((93, 70), (95, 70), (94, 65), (94, 75)):
    e, P, nt = run(R10 >= X, R10 <= 100 - X, R10 <= Y, R10 >= 100 - Y); row(f'komşu: X{X} Y{Y}', M('', e, P, nt))
for fsl in (0.015, 0.03):
    e, P, nt = run(R10 >= 94, R10 <= 6, R10 <= 70, R10 >= 30, fsl=fsl); row(f'komşu: ters stop %{fsl * 100:g}', M('', e, P, nt))
pd.set_option('display.width', 300); print(pd.DataFrame(out).round(1).to_string(index=False), flush=True)
# monkey: tetik zamanlarını rastgele (aynı sayı, pozisyondayken), aynı geri giriş/ters kuralları
inpos = np.flatnonzero(posA != 0); k = int((R10[inpos] >= 94).sum() + (R10[inpos] <= 6).sum()); mk = []
for s in range(200):
    m = np.zeros(n, np.bool_); m[rng.choice(inpos, k, replace=False)] = True
    e, P, nt = run(m, m, R10 <= 70, R10 >= 30); mk.append(e[-1] - C['net'])
mk = np.array(mk); print(f'MONKEY: seçilen ΔNet {S1["net"] - C["net"]:.0f} · rastgele tetik ΔNet ort {mk.mean():.0f} P95 {np.percentile(mk, 95):.0f} maks {mk.max():.0f} · p = {(mk >= S1["net"] - C["net"]).mean():.3f}', flush=True)
# MC: günlük fark bootstrap
ix_ = pd.to_datetime(tms, unit='ms'); dD = pd.Series(eq1 - eq0, ix_).resample('1D').last().diff().dropna().to_numpy()
bs = np.array([rng.choice(dD, len(dD)).sum() for _ in range(5000)]); print(f'MC (günlük fark bootstrap, 5000): ΔNet P5 {np.percentile(bs, 5):.0f} · P50 {np.percentile(bs, 50):.0f} · zarar olasılığı {(bs <= 0).mean() * 100:.1f}%')
# blok bootstrap 12 ay ileri DD (birleşik)
dC = pd.Series(eq1, ix_).resample('1D').last().diff().dropna().to_numpy(); d0 = pd.Series(eq0, ix_).resample('1D').last().diff().dropna().to_numpy()
def fdd(d):
    r = []
    for _ in range(3000):
        st = rng.integers(0, len(d) - 30, 13); x = np.concatenate([d[s:s + 30] for s in st])[:365]; e = np.cumsum(x); r.append((np.maximum.accumulate(e) - e).max())
    return np.percentile(r, [90, 95, 99])
print('12 ay ileri DD P90/P95/P99 (günlük, blok bootstrap): kontrol', np.round(fdd(d0)), '→ seçilen', np.round(fdd(dC)))
# DSR: günlük fark Sharpe; deneme sayısı N; Sharpe varyansı TRIM ızgarasından
R2 = pd.read_pickle('/home/claude/zecwork/ov2_all.pkl'); sel = R2.sample(300, random_state=1).v.tolist(); srs = []
RS2 = {N: np.nan_to_num(rsi(c, N), nan=50.0) for N in (7, 10, 14, 21)}
import re
for v in sel:
    m_ = re.match(r'RSI(\d+) X(\d+) Y(\d+) geri(\d+) devam%([\d.]+) sh<([\d.]+)', v)
    if not m_: continue
    N, X, Y, tb, cont, sh = int(m_[1]), float(m_[2]), float(m_[3]), int(m_[4]), float(m_[5]) / 100, float(m_[6]); Rr = RS2[N]
    e, P, nt = ov3(o, h, l, c, posA, bEp, bXp, bXd, Rr >= X, Rr <= 100 - X, Rr <= Y, Rr >= 100 - Y, tb, cont, sh, trSh, False, 0.02, TICK)
    dd_ = pd.Series(e - eq0, ix_).resample('1D').last().diff().dropna().to_numpy(); srs.append(dd_.mean() / dd_.std() if dd_.std() > 0 else 0)
srs = np.array(srs); SR = dD.mean() / dD.std(); Tn = len(dD); g3 = skew(dD); g4 = kurtosis(dD, fisher=False); NT = 20682 + 672 + 7552 + 288
emc = 0.5772156649; SR0 = np.sqrt(srs.var()) * ((1 - emc) * norm.ppf(1 - 1 / NT) + emc * norm.ppf(1 - 1 / (NT * np.e)))
DSR = norm.cdf((SR - SR0) * np.sqrt(Tn - 1) / np.sqrt(1 - g3 * SR + (g4 - 1) / 4 * SR ** 2))
print(f'DSR: günlük fark Sharpe {SR:.3f} (yıllık {SR * np.sqrt(365):.2f}) · deneme {NT} · denemeler arası SR std {srs.std():.3f} · SR0 {SR0:.3f} · çarpıklık {g3:.2f} basıklık {g4:.1f} · DSR = {DSR:.3f}')
np.save('/home/claude/zecwork/eq_TRIM_sel.npy', eq1)
