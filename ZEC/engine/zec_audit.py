import pandas as pd, numpy as np, glob, json, sys
D=sys.argv[1]; R={}
m=pd.concat([pd.read_csv(f) for f in sorted(glob.glob(D+'/ZECUSDT_1m_UTC_*.csv.gz'))],ignore_index=True)
t=m.time; R['1m']=dict(rows=len(m),dup=int(t.duplicated().sum()),mono=bool(t.is_monotonic_increasing),
  start=str(pd.to_datetime(t.min(),unit='ms')),end=str(pd.to_datetime(t.max(),unit='ms')),
  bad_ohlc=int(((m.high<m[['open','close']].max(axis=1))|(m.low>m[['open','close']].min(axis=1))|(m[['open','high','low','close']]<=0).any(axis=1)).sum()),
  zero_vol=int((m.volume<=0).sum()))
g=t.diff(); gp=m.loc[g>60000,'time']; gl=g[g>60000]
gaps=pd.DataFrame({'from_utc':pd.to_datetime(gp.values-gl.values+60000,unit='ms'),'to_utc':pd.to_datetime(gp.values,unit='ms'),'missing_min':(gl.values//60000-1)})
gaps.to_csv('/home/claude/zecwork/1m_EKSIK_ARALIKLAR.csv',index=False)
R['1m']['gaps']=len(gaps); R['1m']['missing_min']=int(gaps.missing_min.sum()); R['1m']['big_gaps']=gaps[gaps.missing_min>=60].astype(str).values.tolist()
r=np.log(m.close).diff().abs(); R['1m']['jumps_gt_15pct']=int((r>np.log(1.15)).sum())
# metrics
x=pd.read_csv(D+'/ZECUSDT_METRICS_5m_UTC_20211201_THROUGH_20261123.csv.gz')
ts=x.timestamp_ms; R['metrics']=dict(rows=len(x),dup=int(ts.duplicated().sum()),mono=bool(ts.is_monotonic_increasing),aligned5m=int((ts%300000!=0).sum()),
  start=x.timestamp_utc.iloc[0],end=x.timestamp_utc.iloc[-1],nan={c:int(x[c].isna().sum()) for c in ['open_interest_zec','top_trader_accounts_ls','top_trader_positions_ls','global_accounts_ls','taker_long_short_vol_ratio']},
  zero_oi=int((x.open_interest_zec<=0).sum()),flags={c:x[c].value_counts().to_dict() for c in x.columns if c.startswith('flag_')})
gm=ts.diff(); R['metrics']['gaps_gt5m']=int((gm>300000).sum()); R['metrics']['max_gap_h']=float(gm.max()/3.6e6)
# zaman damgası semantiği: |ΔOI| vs 5 dk hacim, kayma k
v=m.set_index(pd.to_datetime(m.time,unit='ms')).volume.resample('5min',label='left',closed='left').sum()
oi=x.set_index(pd.to_datetime(ts,unit='ms')).open_interest_zec.where(lambda s:s>0)
d=oi.diff().abs(); out={}
for k in range(-3,4):  # ΔOI(t)=OI(t)-OI(t-5) ; hacim penceresi [t-5+5k, t+5k)
    vv=v.shift(-k+1)  # v indexed by window start; window [t-5,t) start=t-5 -> shift(1)
    j=pd.concat([d,vv.reindex(d.index)],axis=1).dropna()
    out[k]=round(float(j.corr(method='spearman').iloc[0,1]),4)
R['metrics']['stamp_test_spearman_|dOI|_vs_vol_lag']=out
# taker 5m vs 1m volume
tk=pd.read_csv(D+'/ZECUSDT_PERP_TAKER_BUY_SELL_5m_UTC_20210101_THROUGH_20261123.csv.gz',usecols=['open_time_ms','total_zec','close_price','data_source'])
R['taker']=dict(rows=len(tk),dup=int(tk.open_time_ms.duplicated().sum()),start=str(pd.to_datetime(tk.open_time_ms.min(),unit='ms')),end=str(pd.to_datetime(tk.open_time_ms.max(),unit='ms')),src=tk.data_source.value_counts().to_dict())
j=tk.set_index(pd.to_datetime(tk.open_time_ms,unit='ms')).total_zec.to_frame().join(v.rename('v1m'),how='inner')
R['taker']['vol_match_rel_err_med']=float(((j.total_zec-j.v1m).abs()/j.v1m.replace(0,np.nan)).median())
c5=m.set_index(pd.to_datetime(m.time,unit='ms')).close.resample('5min',label='left',closed='left').last()
jc=tk.set_index(pd.to_datetime(tk.open_time_ms,unit='ms')).close_price.to_frame().join(c5.rename('c'),how='inner')
R['taker']['close_match_pct']=float((np.isclose(jc.close_price,jc.c)).mean()*100)
for f,c in [('ZECUSDT_OI_5m_UTC_THROUGH_20261123.csv.gz','5m'),('ZECUSDT_OI_10m_UTC_THROUGH_20261123.csv.gz','10m')]:
    o=pd.read_csv(D+'/'+f); R['OI_'+c]=dict(rows=len(o),start=o.create_time_utc.iloc[0],end=o.create_time_utc.iloc[-1],durum=o.durum.value_counts().to_dict(),dup=int(o.create_time_utc.duplicated().sum()))
fu=pd.read_csv(D+'/ZECUSDT_FUNDING_EVENTS_20210101_20261123.csv')
iv=(fu.funding_time_ms.diff()/3.6e6).round().value_counts().head(4).to_dict()
R['funding']=dict(rows=len(fu),start=fu.funding_time_utc.iloc[0],end=fu.funding_time_utc.iloc[-1],interval_h=iv,rate_pct_min=float(fu.funding_rate_percent.min()),rate_pct_max=float(fu.funding_rate_percent.max()))
y=m.assign(y=pd.to_datetime(m.time,unit='ms').dt.year).groupby('y').close.agg(['first','last','min','max'])
R['yearly_price']=y.round(2).to_dict('index')
json.dump(R,open('/home/claude/zecwork/KALITE_RAPORU.json','w'),indent=1,default=str)
print(json.dumps(R,indent=1,default=str))
