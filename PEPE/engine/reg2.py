exec(open('reg.py').read().split("rows=[];RES={}")[0])
@numba.njit
def run_k(o,h,l,c,t,karr,atrp,tick,comm,cash):
    n=c.size;pos=0;lock=0;ep=0.;q=0.;stp=0.;real=0.;eq=np.empty(n);k=0;ei=0;hs=False
    te=np.empty(n,np.int64);tx=np.empty(n,np.int64);td=np.empty(n,np.int64);tp=np.empty(n);tc=np.empty(n);ts=np.zeros(n,np.int64)
    for i in range(n):
        if pos!=0 and hs:
            if (pos>0 and l[i]<=stp) or (pos<0 and h[i]>=stp):
                px=(min(o[i],stp)-tick) if pos>0 else (max(o[i],stp)+tick)
                fee=(ep+px)*q*comm;pnl=(px-ep)*q*pos-fee;real+=pnl;te[k]=ei;tx[k]=i;td[k]=pos;tp[k]=pnl;tc[k]=fee;ts[k]=1;k+=1;lock=pos;pos=0
        d=t[i]
        if lock!=0 and d!=lock: lock=0
        eff=d if lock==0 else 0
        if eff!=pos:
            if pos!=0:
                px=c[i]-tick if pos>0 else c[i]+tick;fee=(ep+px)*q*comm;pnl=(px-ep)*q*pos-fee;real+=pnl;te[k]=ei;tx[k]=i;td[k]=pos;tp[k]=pnl;tc[k]=fee;k+=1
            if eff!=0:
                px=c[i]+tick if eff>0 else c[i]-tick;q=np.floor(cash/px);ep=px;ei=i;hs=karr[i]>0
                dist=karr[i]*atrp[i]/100;stp=ep*(1-dist) if eff>0 else ep*(1+dist)
            pos=eff
        eq[i]=real+((c[i]-ep)*q*pos-ep*q*comm if pos!=0 else 0.)
    if pos!=0:
        px=c[n-1];fee=(ep+px)*q*comm;te[k]=ei;tx[k]=n-1;td[k]=pos;tp[k]=(px-ep)*q*pos-fee;tc[k]=fee;k+=1
    return eq,te[:k],tx[:k],td[:k],tp[:k],tc[:k],ts[:k]
def build(order):
    t=np.zeros(N,np.int64);src=np.full(N,-1)
    for j,k in list(enumerate(order))[::-1]:
        p=LIB[k].astype(np.int64);t=np.where(p!=0,p,t);src=np.where(p!=0,j,src)
    return t,src
rows=[]
for order in [['RSIX','MIK','COL','B4H','BFLIP','MID'],['RSIX','COL','B4H','BFLIP','MID'],['RSIX','COL','BFLIP','B4H','MID']]:
    t,src=build(order);ks=[0,3,4,6]
    for combo in itertools.product(ks,repeat=len(order)):
        if combo[0]!=0: continue                      # RSIX stopsuz
        if order[order.index('BFLIP')]=='BFLIP' and combo[order.index('BFLIP')] not in (0,): continue
        karr=np.zeros(N)
        for j,kv in enumerate(combo): karr[src==j]=kv
        r=run_k(O,H,L_,C,t,karr,ATRP,TICK,COMM,CASH);f=metrics(r[0],*r[2:6]);i=metrics(r[0],*r[2:6],0,I_SPLIT);o=metrics(r[0],*r[2:6],I_SPLIT,N)
        rows.append(dict(order='>'.join(order),k=''.join(map(str,combo)),net=f['net'],pf=f['pf'],n=f['n'],mdd=f['mdd'],p99=f['p99'],q=f['net']/f['mdd'],Inet=i['net'],Ipf=i['pf'],Iq=i['net']/max(i['mdd'],1),Onet=o['net'],Opf=o['pf'],Ocal=o['calmar']))
D=pd.DataFrame(rows);D.to_pickle('reg2.pkl');pd.set_option('display.width',260)
print('IS PF + IS kalite ile seçilen ilk 10 (OOS sadece rapor):')
D['Isel']=D.Ipf*np.sqrt(D.Iq)
print(D.sort_values('Isel',ascending=False).head(10).round(2).to_string(index=False))
print('PF>=1.8 & net>=8250:',((D.pf>=1.8)&(D.net>=8250)).sum(),'| PF>=1.75 & net>=7900:',((D.pf>=1.75)&(D.net>=7900)).sum(),'| max PF',round(D.pf.max(),2),'max net',round(D.net.max()))
