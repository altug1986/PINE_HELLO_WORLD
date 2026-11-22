import pickle, itertools, numpy as np, pandas as pd, numba
from eng import *
x=pd.ExcelFile('tv101.xlsx');p=pd.read_excel(x,'Performans');print('TV V10.1 net',p.iloc[2,1],'%',p.iloc[2,2],'| MaxDD',p[p.iloc[:,0].astype(str).str.contains('Maks düşüş')].iloc[:,1].tolist())
d=pd.read_excel(x,'İşlemler');xx=d[d.Tip.astype(str).str.contains('çıkış')];pn=xx['Net K/Z USDT'].astype(float);print('TV işlem',len(xx),'PF',round(pn[pn>0].sum()/-pn[pn<=0].sum(),3))
V=pickle.load(open('V101.pkl','rb'));eq,te,tx,td,tp,tc=V[:6]
@numba.njit
def overlay(c,eq,te,tx,td,act,gb,frac,tick,comm,cash):
    n=c.size;dl=np.zeros(n);tpn=np.empty(te.size);cnt=0
    for k in range(te.size):
        e=te[k];x=tx[k];d=td[k];ep=c[e]+tick*d;q=np.floor(cash/ep)*frac;pk=0.;hit=-1;px=0.
        for i in range(e+1,x):
            r=(c[i]/ep-1)*d
            if r>pk: pk=r
            if pk>=act and pk-r>=gb:
                hit=i;px=c[i]-tick*d;break
        xp=c[x]-tick*d
        if hit>=0:
            cnt+=1
            for i in range(hit+1,x): dl[i]-=q*d*(c[i]-px)
            dv=q*d*(px-xp)-q*comm*(px-xp)*d*0
            for i in range(x,n): dl[i]+=dv
            tpn[k]=dv
        else: tpn[k]=0.
    return eq+dl,tpn,cnt
def health(eqn,tpn):
    f=metrics(eqn,tx,td,tpn,tc);o=metrics(eqn,tx,td,tpn,tc,I_SPLIT,N);i=metrics(eqn,tx,td,tpn,tc,0,I_SPLIT)
    e=pd.Series(eqn[DAY_I],index=DAY_T);mo=e.resample('ME').last().diff();y=e.resample('YE').last().diff();y.iloc[0]=e.resample('YE').last().iloc[0]
    l12=e[e.index>=e.index[-1]-pd.Timedelta(days=365)];r2=np.corrcoef(np.arange(len(l12)),l12.values)[0,1]**2
    return dict(net=f['net'],pct=f['net']/5,pf=f['pf'],mdd=f['mdd'],p99=f['p99'],q=f['net']/f['mdd'],sharpe=f['sharpe'],medM=mo.median(),posm=(mo>0).mean(),cv=f['cv'],y26=y.iloc[-1]/5,r2_12=r2,Inet=i['net'],Iq=i['net']/i['mdd'],Onet=o['net'],Ocal=o['calmar'])
rows=[dict(act=0,gb=0,hits=0,**health(eq,tp))]
for act,gb in itertools.product([.02,.03,.04,.06,.08,.10,.15,.20],[.01,.02,.03,.05,.08]):
    if gb>=act: continue
    en,dv,cnt=overlay(C,eq,te,tx,td,act,gb,0.3,TICK,COMM,CASH);rows.append(dict(act=act,gb=gb,hits=cnt,**health(en,tp+dv)))
D=pd.DataFrame(rows);D.to_pickle('ptrail.pkl');pd.set_option('display.width',250)
b=D.iloc[0];print(D.iloc[:1].round(2).to_string(index=False))
print('net (satır: aktivasyon kâr %, sütun: geri çekilme %):');print((D.iloc[1:].pivot_table(index='act',columns='gb',values='net')-b.net).round(0))
print('aylık medyan farkı:');print((D.iloc[1:].pivot_table(index='act',columns='gb',values='medM')-b.medM).round(1))
print('OOS Calmar:');print(D.iloc[1:].pivot_table(index='act',columns='gb',values='Ocal').round(2))
g=D[(D.net>b.net)&(D.q>=b.q)&(D.medM>=b.medM)];print('net↑ kalite↑ medyan↑:',len(g));print(g.sort_values('net',ascending=False).head(6).round(2).to_string(index=False))
