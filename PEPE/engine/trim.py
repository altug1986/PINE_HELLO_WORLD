exec(open("sar.py").read().split("t8,src=build")[0].split("x=pd.ExcelFile")[0])
exec("@numba.njit\ndef run_sar"+open("sar.py").read().split("@numba.njit\ndef run_sar")[1].split("t8,src=build")[0])
t8,src=build(['RSIX','MIK','COL','B4H','BFLIP','MID'])
d_=np.diff(C,prepend=np.nan)
def rsi(n): u=rma(np.where(d_>0,d_,0.),n);w=rma(np.where(d_<0,-d_,0.),n);return 100-100/(1+u/w)
RS={7:np.nan_to_num(rsi(7),nan=50.),14:np.nan_to_num(rsi(14),nan=50.)}
@numba.njit
def run_trim(c,t,R,X,Yre,M,z,flip,tick,comm,cash):
    n=c.size;pos=0;ep=0.;q=0.;ei=0;real=0.;eq=np.empty(n);kk=0;prev=0;mode=0;xp=0.;xb=0;md=0
    te=np.empty(n,np.int64);tx=np.empty(n,np.int64);td=np.empty(n,np.int64);tp=np.empty(n);tc=np.empty(n);ts=np.zeros(n,np.int64)
    for i in range(n):
        ce=(t[i]!=prev);prev=t[i];want=pos;tag=0
        if ce: mode=0;want=t[i]
        elif mode==0:
            want=t[i]
            if t[i]>0 and pos>0 and R[i]>=X: want=-1 if flip else 0;mode=1;xp=c[i];xb=i;md=1;tag=1
            elif t[i]<0 and pos<0 and R[i]<=100-X: want=1 if flip else 0;mode=1;xp=c[i];xb=i;md=-1;tag=1
        else:
            back=False
            if md>0: back=(R[i]<=Yre) or (i-xb>=M) or (c[i]>=xp*(1+z))
            else: back=(R[i]>=100-Yre) or (i-xb>=M) or (c[i]<=xp*(1-z))
            if back: mode=0;want=t[i];tag=2
        if want!=pos:
            if pos!=0:
                px=c[i]-tick if pos>0 else c[i]+tick;fee=(ep+px)*q*comm;pnl=(px-ep)*q*pos-fee;real+=pnl
                te[kk]=ei;tx[kk]=i;td[kk]=pos;tp[kk]=pnl;tc[kk]=fee;ts[kk]=tag;kk+=1
            if want!=0:
                px=c[i]+tick if want>0 else c[i]-tick;q=np.floor(cash/px);ep=px;ei=i
            pos=want
        eq[i]=real+((c[i]-ep)*q*pos-ep*q*comm if pos!=0 else 0.)
    if pos!=0:
        px=c[n-1];fee=(ep+px)*q*comm;te[kk]=ei;tx[kk]=n-1;td[kk]=pos;tp[kk]=(px-ep)*q*pos-fee;tc[kk]=fee;kk+=1
    return eq,te[:kk],tx[:kk],td[:kk],tp[:kk],tc[:kk],ts[:kk]
rows=[]
for L,X,Y,M,z,fl in itertools.product([7,14],[80,85,90,95],[40,50,60],[12,24,48],[.01,.02,.03],[0,1]):
    r=run_trim(C,t8,RS[L],X,Y,M,z,fl,TICK,COMM,CASH);f=metrics(r[0],*r[2:6]);i=metrics(r[0],*r[2:6],0,I_SPLIT);o=metrics(r[0],*r[2:6],I_SPLIT,N)
    eqd=pd.Series(r[0][DAY_I],index=DAY_T);b6=eqd.resample('6ME').last().diff().dropna()
    rows.append(dict(L=L,X=X,Y=Y,M=M,z=z,fl=fl,net=f['net'],pct=f['net']/5,pf=f['pf'],n=f['n'],p99=f['p99'],mdd=f['mdd'],q=f['net']/f['mdd'],Inet=i['net'],Ipf=i['pf'],Iq=i['net']/max(i['mdd'],1),Onet=o['net'],Opf=o['pf'],Ocal=o['calmar'],posm=f['posm'],cv=f['cv'],b6min=b6.min()))
D=pd.DataFrame(rows);D.to_pickle('trim.pkl');pd.set_option('display.width',260)
print('n 1900-2100 & net>9250:',((D.n.between(1900,2100))&(D.net>9250)).sum(),'| net>7905 (V8 üstü):',(D.net>7905).sum(),'| max net',round(D.net.max()),'| n aralığı',D.n.min(),D.n.max())
D['Isel']=D.Inet/1000*np.sqrt(D.Iq)
print('IS seçimine göre ilk 12 (OOS rapor):');print(D.sort_values('Isel',ascending=False).head(12).round(2).to_string(index=False))
print('n>=1800 olanlarda IS en iyi 6:');print(D[D.n>=1800].sort_values('Isel',ascending=False).head(6).round(2).to_string(index=False))
