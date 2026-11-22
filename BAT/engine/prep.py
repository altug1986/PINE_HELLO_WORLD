import numpy as np, pandas as pd, sys
# 1m -> TF bars; HTF arrays; BTC; OI. Output npz per TF.
bat = pd.read_csv('BATUSDT_1DK_OHLCV_UTC.csv', usecols=['open_time_ms','open','high','low','close','volume']).to_numpy(float)
btc = pd.read_csv('btc1m.csv.gz', usecols=['time','open','high','low','close']).to_numpy(float)
END = btc[-1,0]  # last BTC 1m open
bat = bat[bat[:,0] <= END]
print('BTC end', pd.to_datetime(END, unit='ms'), 'BAT rows', len(bat))

def rs(a, tfmin):
    k = (a[:,0] // (tfmin*60000)).astype(np.int64)
    st = np.r_[0, np.flatnonzero(np.diff(k)) + 1]
    en = np.r_[st[1:], len(a)]
    o = a[st,1]; c = a[en-1,4]
    h = np.maximum.reduceat(a[:,2], st); l = np.minimum.reduceat(a[:,3], st)
    return k[st], o, h, l, c
def vsum(a, tfmin):
    k = (a[:,0] // (tfmin*60000)).astype(np.int64); st = np.r_[0, np.flatnonzero(np.diff(k)) + 1]
    return np.add.reduceat(a[:,5], st)

oi = pd.read_csv('oi/BATUSDT_OI_KAYNAK_5DK_UTC.csv', usecols=['create_time_utc','sum_open_interest'])
oiT = pd.to_datetime(oi.create_time_utc).to_numpy().astype('datetime64[ms]').astype(np.int64)
oiV = oi.sum_open_interest.to_numpy(float)
m0 = oiV > 0; oiT = oiT[m0]; oiV = oiV[m0]  # OI=0 arşiv hatası: eksik say

for tf in [5]:
    k, o, h, l, c = rs(bat, tf)
    t = k * tf * 60000
    out = dict(t=t, o=o, h=h, l=l, c=c, v=vsum(bat, tf))
    for name, src, htf in [('b30', bat, 30), ('b60', bat, 60), ('x240', btc, 240), ('x60', btc, 60)]:
        hk, ho, hh, hl, hc = rs(src, htf)
        out[name+'_k'] = hk; out[name] = np.stack([ho, hh, hl, hc])
        out[name+'_map'] = (t // (htf*60000)).astype(np.int64)  # HTF bucket of each bar
    # BTC close at chart TF, ffill
    xk, _, _, _, xc = rs(btc, tf)
    idx = np.searchsorted(xk, k, side='right') - 1
    out['xc'] = np.where(idx >= 0, xc[np.clip(idx, 0, None)], np.nan)
    # OI: value with create_time <= bar open, ffill (OI symbol bar close ~ taken at bar open stamp); [1] applied in logic
    j = np.searchsorted(oiT, t, side='right') - 1
    v = np.where(j >= 0, oiV[np.clip(j, 0, None)], np.nan)
    v[t < oiT[0]] = np.nan
    out['oi'] = v
    np.savez(f'd{tf}.npz', **out)
    print(tf, len(t), pd.to_datetime(t[0], unit='ms'), pd.to_datetime(t[-1], unit='ms'))
