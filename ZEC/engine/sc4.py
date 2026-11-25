# SUPER-CALMAR V4 (Katman 1) — hafızadaki tanım; 10 dk MTM, sabit nominal %
import numpy as np, pandas as pd, numba
@numba.njit(cache=True)
def roll_mdd(e, w, step):
    out = []
    for s in range(0, len(e) - w, step):
        pk = e[s]; md = 0.0
        for j in range(s, s + w):
            if e[j] > pk: pk = e[j]
            if pk - e[j] > md: md = pk - e[j]
        out.append(md)
    return np.array(out)
def sc4(eq, nominal, detail=False):
    """eq: 10 dk kümülatif $ (pd.Series, 10 dk index). Döner: SC4 ve bileşenler (% nominal)."""
    e = eq.values.astype(float) / nominal * 100; n = len(e); yrs = (eq.index[-1] - eq.index[0]).total_seconds() / (365.25 * 86400)
    ann = (e[-1] - e[0]) / yrs; dd = np.maximum.accumulate(e) - e; mdd = dd.max()
    r30 = roll_mdd(e, 30 * 144, 144); p95, p99 = np.percentile(r30, [95, 99]); es95 = r30[r30 >= p95].mean(); dtail = (p95 + p99 + es95) / 3
    r10 = np.diff(e); r1h = e[6::6] - e[:-6:6] if False else np.diff(e[::6]); r24 = np.diff(e[::144])
    smul = (np.sqrt(144) * r10.std() + np.sqrt(24) * r1h.std() + r24.std()) / 3
    ath = np.maximum.accumulate(e); last = np.zeros(n); k = 0
    for i in range(n):
        if e[i] >= ath[i] - 1e-12: k = i
        last[i] = (i - k) / 144.0
    a95 = np.percentile(last, 95); qz = 1 / (1 + a95 / 7)
    r7 = e[7 * 144::144] - e[:-7 * 144:144]; mu = r7.mean(); qd = 0.0 if mu <= 0 else 1 / (2 - min(1.0, np.percentile(r7, 10) / mu))
    wk = pd.Series(e, eq.index).resample('W').last().diff().dropna().values; pos = wk[wk > 0]; top = np.sort(wk)[::-1][:max(1, int(round(len(wk) * 0.1)))]
    K = top.sum() / pos.sum() if pos.sum() > 0 else 1.0; qk = 1 - K / 2
    s = ann / (mdd + dtail + smul) * np.sqrt(qz) * qd * qk
    if not detail: return s
    return dict(SC4=s, yıllık=ann, MaxDD=mdd, Dtail=dtail, Smulti=smul, A95g=a95, Qz=qz, Qd=qd, Qk=qk, Calmar=ann / mdd, P99=np.percentile(dd, 99))
if __name__ == '__main__':
    Pp = pd.read_pickle('/home/claude/port/portfoy_canli_10dk_utc.pkl'); Pp = Pp[~Pp.index.duplicated()]
    print('PORTFÖY 4.600:', {k: round(v, 3) for k, v in sc4(Pp, 4600, True).items()})
