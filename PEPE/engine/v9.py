exec(open("trim.py").read().split("rows=[]")[0])
@numba.njit
def run_v9(o,h,l,c,t,R,X,Y,M,z,karr,atrp,tick,comm,cash):
    n=c.size;pos=0;ep=0.;q=0.;ei=0;real=0.;eq=np.empty(n);kk=0;prev=0;mode=0;xp=0.;xb=0;md=0;stp=0.;hs=False;lock=0
    te=np.empty(n,np.int64);tx=np.empty(n,np.int64);td=np.empty(n,np.int64);tp=np.empty(n);tc=np.empty(n);ts=np.zeros(n,np.int64)
    for i in range(n):
        if pos!=0 and hs and ((pos>0 and l[i]<=stp) or (pos<0 and h[i]>=stp)):
            px=(min(o[i],stp)-tick) if pos>0 else (max(o[i],stp)+tick)
            fee=(ep+px)*q*comm;pnl=(px-ep)*q*pos-fee;real+=pnl;te[kk]=ei;tx[kk]=i;td[kk]=pos;tp[kk]=pnl;tc[kk]=fee;ts[kk]=3;kk+=1;lock=pos;pos=0;mode=0
        ce=(t[i]!=prev);prev=t[i];tag=0
        if lock!=0 and t[i]!=lock: lock=0
        want=pos
        if lock!=0: want=0
        elif ce: mode=0;want=t[i]
        elif mode==0:
            want=t[i]
            if t[i]>0 and pos>0 and R[i]>=X: want=0;mode=1;xp=c[i];xb=i;md=1;tag=1
            elif t[i]<0 and pos<0 and R[i]<=100-X: want=0;mode=1;xp=c[i];xb=i;md=-1;tag=1
        else:
            if md>0: back=(R[i]<=Y) or (i-xb>=M) or (c[i]>=xp*(1+z))
            else: back=(R[i]>=100-Y) or (i-xb>=M) or (c[i]<=xp*(1-z))
            if back: mode=0;want=t[i];tag=2
            else: want=0
        if want!=pos:
            if pos!=0:
                px=c[i]-tick if pos>0 else c[i]+tick;fee=(ep+px)*q*comm;pnl=(px-ep)*q*pos-fee;real+=pnl
                te[kk]=ei;tx[kk]=i;td[kk]=pos;tp[kk]=pnl;tc[kk]=fee;ts[kk]=tag;kk+=1
            if want!=0:
                px=c[i]+tick if want>0 else c[i]-tick;q=np.floor(cash/px);ep=px;ei=i
                hs=karr[i]>0;d=karr[i]*atrp[i]/100;stp=ep*(1-d) if want>0 else ep*(1+d)
            pos=want
        eq[i]=real+((c[i]-ep)*q*pos-ep*q*comm if pos!=0 else 0.)
    if pos!=0:
        px=c[n-1];fee=(ep+px)*q*comm;te[kk]=ei;tx[kk]=n-1;td[kk]=pos;tp[kk]=(px-ep)*q*pos-fee;tc[kk]=fee;kk+=1
    return eq,te[:kk],tx[:kk],td[:kk],tp[:kk],tc[:kk],ts[:kk]
karr=np.zeros(N);karr[(src==3)|(src==5)]=3.0;Z0=np.zeros(N)
rows=[];RES={}
def rr(nm,r):
    f=metrics(r[0],*r[2:6]);i=metrics(r[0],*r[2:6],0,I_SPLIT);o=metrics(r[0],*r[2:6],I_SPLIT,N);m=mae(r)
    eqd=pd.Series(r[0][DAY_I],index=DAY_T);b6=eqd.resample('6ME').last().diff().dropna();mo=eqd.resample('ME').last().diff()
    rows.append(dict(sys=nm,net=f['net'],pct=f['net']/5,pf=f['pf'],n=f['n'],mae99=np.percentile(m,99),maemax=m.max(),p95=f['p95'],p99=f['p99'],mdd=f['mdd'],q=f['net']/f['mdd'],sharpe=f['sharpe'],Ipf=i['pf'],Opf=o['pf'],Onet=o['net'],Ocal=o['calmar'],posm=f['posm'],medM=mo.median(),b6min=b6.min()));RES[nm]=r
for X,Y,z in [(90,60,.03),(85,60,.03),(95,60,.03),(90,50,.03),(90,60,.02),(90,60,.04)]:
    rr(f'TRIM X{X} Y{Y} z{z}',run_v9(O,H,L_,C,t8,RS[7],X,Y,12,z,Z0,ATRP,TICK,COMM,CASH))
    rr(f'TRIM+STOP X{X} Y{Y} z{z}',run_v9(O,H,L_,C,t8,RS[7],X,Y,12,z,karr,ATRP,TICK,COMM,CASH))
rr('V8.1',pickle.load(open('V81.pkl','rb')))
D=pd.DataFrame(rows);pd.set_option('display.width',270);print(D.round(2).to_string(index=False))
pickle.dump(RES['TRIM+STOP X90 Y60 z0.03'],open('V9.pkl','wb'))
