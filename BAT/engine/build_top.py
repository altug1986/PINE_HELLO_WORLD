import sys, glob, zipfile, io, numpy as np, pandas as pd
src, out = sys.argv[1], sys.argv[2]; L = []
for f in sorted(glob.glob(src + '/*.zip')):
    with zipfile.ZipFile(f) as z:
        for nm in z.namelist(): L.append(pd.read_csv(io.BytesIO(z.read(nm))))
d = pd.concat(L, ignore_index=True); d['ts'] = pd.to_datetime(d.create_time); d = d.drop_duplicates('ts').sort_values('ts')
C = ['count_toptrader_long_short_ratio', 'sum_toptrader_long_short_ratio', 'count_long_short_ratio', 'sum_taker_long_short_vol_ratio', 'sum_open_interest']
for k in C: d[k] = pd.to_numeric(d[k], errors='coerce')
print('satır', len(d), '|', d.ts.min(), '→', d.ts.max()); print('boş/sıfır oranı:', {k: round(((d[k].isna()) | (d[k] <= 0)).mean() * 100, 2) for k in C})
full = pd.date_range(d.ts.min().floor('D'), d.ts.max().ceil('D') - pd.Timedelta('5min'), freq='5min'); miss = full.difference(d.ts); print('eksik 5dk kayıt', len(miss), f'(%{len(miss)/len(full)*100:.2f})')
print(d[C[:4]].describe().round(3).to_string())
d['ms'] = d.ts.to_numpy().astype('datetime64[ms]').astype(np.int64); d[['ms'] + C].to_pickle(out); print('kaydedildi', out)
