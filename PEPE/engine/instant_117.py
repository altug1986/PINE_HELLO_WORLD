import numpy as np, pandas as pd, numba, itertools, time
t0=time.time()
m=pd.read_csv('/mnt/user-data/uploads/1000PEPEUSDT_BINANCE_USDM_1m_2023-05-05_2026-11-19.csv',usecols=['open_time_ms','open','high','low','close'])
for k in ['open','high','low','close']: m[k]=np.round(m[k].values/1e-7)*1e-7
MT=(m.open_time_ms.values//1000).astype(np.int64);O,H,L,C=[m[k].values.astype(np.float64) for k in ['open','high','low','close']]
x=pd.ExcelFile('/mnt/user-data/uploads/b0e4c899-CYBORG___PEPE_5M_SWING_V11.7_BINANCE_1000PEPEUSDT.P_2026-11-21_95259.xlsx')
d=pd.read_excel(x,'İşlemler');d=d[pd.to_datetime(d['Tarih ve saat'],errors='coerce').notna()].copy();d['t']=pd.to_datetime(d['Tarih ve saat'])
e=d[d.Tip.str.contains('giriş')].set_index('İşlem numarası');xx=d[d.Tip.str.contains('çıkış')].set_index('İşlem numarası');j=e.join(xx,lsuffix='_e',rsuffix='_x',how='inner')
TR=pd.DataFrame({'et':j.t_e,'xt':j.t_x,'dir':np.where(j.Tip_e.str.contains('Kısa'),-1,1),'ep':j['Fiyat USDT_e'].astype(float),'xp':j['Fiyat USDT_x'].astype(float),'q':j['Boyut (miktar)_x'].astype(float),'pnl':j['Net K/Z USDT_x'].astype(float)}).reset_index(drop=True)
toS=lambda t:((t-pd.Timedelta(hours=3)+pd.Timedelta(minutes=5)-pd.Timestamp('1970-01-01'))//pd.Timedelta(seconds=1)).values.astype(np.int64)   # TV saati (UTC+3) bar açılışı → dolum = bar kapanışı (UTC)
SI=np.searchsorted(MT,toS(TR.et));EI=np.searchsorted(MT,toS(TR.xt))
print('TV işlem',len(TR),'net',round(TR.pnl.sum(),2),'| boş pencere',int((EI<=SI).sum()),'| ort. pencere dk',round(float((EI-SI).mean()),1))
@numba.njit
def sim(o,h,l,c,si,ei,dr,ep,act,cb,sl,tick):
    n=si.size;xp=np.full(n,np.nan);xi=np.full(n,-1);why=np.zeros(n,np.int64)
    for k in range(n):
        d=dr[k];e=ep[k];armed=False;pk=e
        for i in range(si[k],ei[k]):
            if d>0:                                     # LONG: O-L-H-C
                if armed and l[i]<=pk*(1-cb): xp[k]=min(o[i],pk*(1-cb))-tick;xi[k]=i;why[k]=1;break
                if sl>0 and l[i]<=e*(1-sl): xp[k]=min(o[i],e*(1-sl))-tick;xi[k]=i;why[k]=2;break
                if h[i]>pk: pk=h[i]
                if (not armed) and pk>=e*(1+act): armed=True
                if armed and c[i]<=pk*(1-cb): xp[k]=pk*(1-cb)-tick;xi[k]=i;why[k]=1;break
            else:                                       # SHORT: O-H-L-C
                if armed and h[i]>=pk*(1+cb): xp[k]=max(o[i],pk*(1+cb))+tick;xi[k]=i;why[k]=1;break
                if sl>0 and h[i]>=e*(1+sl): xp[k]=max(o[i],e*(1+sl))+tick;xi[k]=i;why[k]=2;break
                if l[i]<pk: pk=l[i]
                if (not armed) and pk<=e*(1-act): armed=True
                if armed and c[i]>=pk*(1+cb): xp[k]=pk*(1+cb)+tick;xi[k]=i;why[k]=1;break
    return xp,xi,why
@numba.njit
def ibdd(o,h,l,c,si,ei,xi,dr,ep,q,pnl,n1):            # 1m çubuk içi MTM (en kötü fiyat) MaxDD
    eq=0.;pk=0.;mdd=0.;k=0;nt=si.size
    for k in range(nt):
        end=xi[k]+1 if xi[k]>=0 else ei[k]
        for i in range(si[k],end):
            w=l[i] if dr[k]>0 else h[i];u=eq+(w-ep[k])*q[k]*dr[k]
            if pk-u>mdd: mdd=pk-u
            cl=eq+(c[i]-ep[k])*q[k]*dr[k]
            if cl>pk: pk=cl
        eq+=pnl[k]
        if eq>pk: pk=eq
    return mdd
dr=TR.dir.values.astype(np.int64);EP=TR.ep.values;Q=TR.q.values
def run(act,cb,sl):
    xp,xi,why=sim(O,H,L,C,SI,EI,dr,EP,act,cb,sl,1e-7)
    p=TR.pnl.values.copy();ins=why>0;p[ins]=(xp[ins]-EP[ins])*Q[ins]*dr[ins]-(EP[ins]+xp[ins])*Q[ins]*0.0004
    xt=TR.xt.values.copy();xt[ins]=pd.to_datetime(MT[xi[ins]]+60,unit='s')+pd.Timedelta(hours=3)
    return p,why,pd.to_datetime(xt),ibdd(O,H,L,C,SI,EI,xi,dr,EP,Q,p,len(TR))
def rep(p,why,xt,ib,nm):
    s=pd.Series(p,index=xt).sort_index();dd_=s.groupby(s.index.floor('D')).sum();eq=dd_.reindex(pd.date_range(dd_.index.min(),dd_.index.max(),freq='D'),fill_value=0).cumsum();dd=eq.cummax()-eq
    ct=s.cumsum();cdd=(ct.cummax().clip(lower=0)-ct).max();yrs=len(eq)/365.25;ul=np.sqrt((dd**2).mean())
    y=(s.groupby(s.index.year).sum()/5).round(0).astype(int).tolist()
    return {'':nm,'Net $':p.sum(),'%':p.sum()/5,'PF':p[p>0].sum()/-p[p<=0].sum(),'Win %':100*(p>0).mean(),'Anlık TP':int((why==1).sum()),'SL %15':int((why==2).sum()),
            'MaxDD kapanış':cdd,'MaxDD çubuk içi (1m)':ib,'DD P95':np.percentile(dd,95),'DD P98':np.percentile(dd,98),'DD P99':np.percentile(dd,99),'Ulcer':ul,'UPI':eq.iloc[-1]/yrs/ul,
            '2025-05-20 sonrası':s[s.index>='2025-05-20'].sum(),'Son 12 ay':s[s.index>=s.index.max()-pd.Timedelta(days=365)].sum(),'Yıl %':'/'.join(map(str,y))}
b=run(9.,1.,0.);w=run(.065,.0015,.15)
T=pd.DataFrame([rep(*b,'TV V11.6 (WT yok)'),rep(*w,'WT anlık: +%6.5→%0.15 geri, SL %15')]).set_index('');pd.set_option('display.width',250);print(T.T.to_string())
p,why,_,_=w;print(f"\nAnlık TP kapanan {int((why==1).sum())} işlem: TV'de bu işlemler {TR.pnl.values[why==1].sum():.0f}$ → WT ile {p[why==1].sum():.0f}$ | SL kapanan {int((why==2).sum())}: TV {TR.pnl.values[why==2].sum():.0f}$ → WT {p[why==2].sum():.0f}$")
print('\nDUYARLILIK Δnet (SL %15) satır aktivasyon %, sütun geri çekilme %:')
g=[]
for a,c_ in itertools.product([.05,.065,.08,.10],[.0015,.003,.005,.01]): g.append((a*100,c_*100,run(a,c_,.15)[0].sum()-b[0].sum()))
print(pd.DataFrame(g,columns=['akt','geri','dnet']).pivot(index='akt',columns='geri',values='dnet').round(0).to_string())
for s_ in [.10,.15,.20,.25]: r=run(9.,1.,s_);print(f"Sadece SL %{s_*100:.0f}: Δnet {r[0].sum()-b[0].sum():+.0f} | tetik {int((r[1]==2).sum())} | 1m çubuk içi MaxDD {r[3]:.0f}")
print('süre',round(time.time()-t0),'s')
