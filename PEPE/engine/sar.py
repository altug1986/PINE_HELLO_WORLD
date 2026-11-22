exec(open("reg2.py").read().split("\nrows=[]\n")[0])
x=pd.ExcelFile('tv81.xlsx');p=pd.read_excel(x,'Performans');print('TV V8.1:',p.iloc[2,1],'$ %',p.iloc[2,2],'|',p[p.iloc[:,0].astype(str).str.contains('Maks düşüş')].iloc[:,1].tolist())
d=pd.read_excel(x,'İşlemler');xx=d[d.Tip.astype(str).str.contains('çıkış')];pf_=xx['Net K/Z USDT'].astype(float);mae_=xx['Tersine hareket %'].astype(float).abs()
print('TV işlem',len(xx),'PF',round(pf_[pf_>0].sum()/-pf_[pf_<=0].sum(),2),'MAE P95/P99/max',np.percentile(mae_,[95,99]).round(2),round(mae_.max(),2))
@numba.njit
def run_sar(c,t,atrp,k,minb,act,tick,comm,cash):
    n=c.size;pos=0;ep=0.;q=0.;ext=0.;ei=0;real=0.;eq=np.empty(n);kk=0;prev=0
    te=np.empty(n,np.int64);tx=np.empty(n,np.int64);td=np.empty(n,np.int64);tp=np.empty(n);tc=np.empty(n);ts=np.zeros(n,np.int64)
    for i in range(n):
        ce=(t[i]!=prev) and t[i]!=0;prev=t[i];want=pos;sar=0
        if ce and t[i]!=pos: want=t[i]
        elif pos==0 and t[i]!=0 and act==1: want=t[i]
        elif pos==0 and act==0 and t[i]!=0 and ce: want=t[i]
        elif pos!=0:
            if pos>0:
                if c[i]>ext: ext=c[i]
                adv=(ext-c[i])/ext
            else:
                if c[i]<ext: ext=c[i]
                adv=(c[i]-ext)/ext
            if k>0 and adv>=k*atrp[i]/100 and i-ei>=minb:
                want=-pos if act==1 else 0;sar=1
        if want!=pos:
            if pos!=0:
                px=c[i]-tick if pos>0 else c[i]+tick;fee=(ep+px)*q*comm;pnl=(px-ep)*q*pos-fee;real+=pnl
                te[kk]=ei;tx[kk]=i;td[kk]=pos;tp[kk]=pnl;tc[kk]=fee;ts[kk]=sar;kk+=1
            if want!=0:
                px=c[i]+tick if want>0 else c[i]-tick;q=np.floor(cash/px);ep=px;ei=i;ext=c[i]
            pos=want
        eq[i]=real+((c[i]-ep)*q*pos-ep*q*comm if pos!=0 else 0.)
    if pos!=0:
        px=c[n-1];fee=(ep+px)*q*comm;te[kk]=ei;tx[kk]=n-1;td[kk]=pos;tp[kk]=(px-ep)*q*pos-fee;tc[kk]=fee;kk+=1
    return eq,te[:kk],tx[:kk],td[:kk],tp[:kk],tc[:kk],ts[:kk]
t8,src=build(['RSIX','MIK','COL','B4H','BFLIP','MID'])
rows=[]
def recs(prm,r):
    f=metrics(r[0],*r[2:6]);i=metrics(r[0],*r[2:6],0,I_SPLIT);o=metrics(r[0],*r[2:6],I_SPLIT,N)
    eqd=pd.Series(r[0][DAY_I],index=DAY_T);b6=eqd.resample('6ME').last().diff().dropna()
    rows.append(dict(**prm,net=f['net'],pct=f['net']/5,pf=f['pf'],n=f['n'],p99=f['p99'],mdd=f['mdd'],q=f['net']/f['mdd'],Ipf=i['pf'],Opf=o['pf'],Onet=o['net'],Ocal=o['calmar'],posm=f['posm'],cv=f['cv'],b6min=b6.min(),sar=int(r[6].sum())))
    return r
recs(dict(k=0,mb=0,act=1),run_sar(C,t8,ATRP,0.,0,1,TICK,COMM,CASH))
for k,mb,act in itertools.product([1.5,2,2.5,3,4,5,6,8],[0,6,12,24],[1,0]):
    recs(dict(k=k,mb=mb,act=act),run_sar(C,t8,ATRP,k,mb,act,TICK,COMM,CASH))
D=pd.DataFrame(rows);D.to_pickle('sar1.pkl');pd.set_option('display.width',250)
print(D.sort_values('q',ascending=False).head(14).round(2).to_string(index=False))
print('n 1900-2100 & net>9250:',((D.n.between(1900,2100))&(D.net>9250)).sum())
print(D[D.act==1].groupby('k')[['net','pf','n','q','Opf','Ocal']].median().round(2).to_string())
