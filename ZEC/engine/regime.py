"""ADX rejimi (kapanmış HTF barı, Pine: ta.dmi(14,14)[1] + lookahead_on)."""
import numpy as np, pandas as pd
import z02


def dmi(H, n=14):
    h, l, c = H.high, H.low, H.close
    up = h.diff(); dn = -l.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0); mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    c1 = c.shift(1); tr = np.maximum(h - l, np.maximum((h - c1).abs(), (l - c1).abs()))
    r = lambda x: pd.Series(x, index=H.index).ewm(alpha=1 / n, adjust=False).mean()
    atr = r(tr); pdi = 100 * r(pdm) / atr; mdi = 100 * r(mdm) / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi); adx = dx.ewm(alpha=1 / n, adjust=False).mean()
    return adx.values, pdi.values, mdi.values


def regime(B, rule, n=14):
    adx = z02.htf(B, rule, lambda H: dmi(H, n)[0])
    pdi = z02.htf(B, rule, lambda H: dmi(H, n)[1]); mdi = z02.htf(B, rule, lambda H: dmi(H, n)[2])
    return adx, np.sign(pdi - mdi)
