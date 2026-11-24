import numpy as np, pandas as pd
b=pd.read_csv('/home/claude/data/BTCUSDT_PERP_1m_2019_20261115.csv.gz',usecols=['time','close'])
b=b.drop_duplicates('time').set_index(pd.to_datetime(b.time.drop_duplicates(),unit='ms',utc=True)).close
c5=b.resample('5min',label='right',closed='right').last().ffill()           # 5 dk kapanış, etiket = kapanış anı
r5=np.log(c5).diff()
m=pd.read_csv('/home/claude/data/BTCUSDT_OI_5m_UTC_20200910_THROUGH_20261119.csv.gz',usecols=['timestamp_utc','open_interest_qty','sum_taker_long_short_vol_ratio'])
m.index=pd.to_datetime(m.timestamp_utc,utc=True)+pd.Timedelta('5min')  # 1 bar gecikme (güvenli)
m=m[~m.index.duplicated()].drop(columns='timestamp_utc').ffill()
H=pd.DataFrame(index=c5.resample('1h',label='right',closed='right').last().index)
H['px']=c5.reindex(H.index,method='ffill')
H['vol24']=r5.rolling(288).std().reindex(H.index,method='ffill')*np.sqrt(288)
H['mom30']=H.px/H.px.rolling(720).mean()-1
H['ret24']=H.px/H.px.shift(24)-1
H['dd90']=H.px/H.px.rolling(2160).max()-1
H['taker24']=np.log(m.sum_taker_long_short_vol_ratio.where(lambda x:x>0).ffill()).rolling(288).mean().reindex(H.index,method='ffill')
H['oi24']=np.log(m.open_interest_qty).diff(288).reindex(H.index,method='ffill')
W={'vol24':(.25,-1),'mom30':(.25,1),'ret24':(.15,1),'dd90':(.20,1),'taker24':(.15,1)}
S=0
for k,(w,s) in W.items():
    p=H[k].rolling(2160,min_periods=1000).rank(pct=True)*100
    H['s_'+k]=p if s>0 else 100-p; S=S+w*H['s_'+k]
H['fear']=S.round(1); H=H[H.index>='2020-12-10']
H.round(6).to_csv('BTC_KORKU_ENDEKSI_1S_BINANCE.csv.gz',index_label='saat_utc_kapanis')
print(H.fear.describe().round(1).to_string()); print(H.tail(3)[['px','fear']])
