import pickle, itertools, numpy as np, pandas as pd, numba
from eng import *
LIB=pickle.load(open('/home/claude/data/LIB.pkl','rb'));W50=pickle.load(open('/home/claude/data/W50.pkl','rb'))
O=B5.open.values
WI=[(int(np.searchsorted(B5.index,a.tz_localize('UTC'))),int(np.searchsorted(B5.index,b.tz_localize('UTC')))) for a,b in W50]
ORDER=['RSIX','MIK','COL','B4H','BFLIP','MID']
# 1h ATR% (son tamamlanmış 1h)
h1,l1,c1,th1,_=htf(60);pc=np.r_[np.nan,c1[:-1]];tr1=np.fmax(np.fmax(h1-l1,np.abs(h1-pc)),np.abs(l1-pc));tr1[0]=h1[0]-l1[0]
ATRP=np.nan_to_num(htf_map(rma(tr1,14)/c1*100,th1,3600),nan=2.0)
@numba.njit
def shadow_ok(p,c,K,th,cost):
    # motorun kendi sanal işlemleri (kapanış→kapanış, maliyet düşülmüş) → son K işlem PF >= th ise izinli
    n=p.size;ok=np.ones(n,np.bool_);buf=np.zeros(K);cnt=0;pos=0;ep=0.
    for i in range(n):
        gp=0.;gl=0.;m=min(cnt,K)
        for j in range(m):
            if buf[j]>0: gp+=buf[j]
            else: gl-=buf[j]
        if cnt>=K: ok[i]=(gl==0) or (gp/gl>=th)
        if p[i]!=pos:
            if pos!=0:
                r=(c[i]/ep-1)*pos-cost;buf[cnt%K]=r;cnt+=1
            if p[i]!=0: ep=c[i]
            pos=p[i]
    return ok
def target(order=ORDER,K=0,th=1.0):
    t=np.zeros(N,np.int64)
    for k in order[::-1]:
        p=LIB[k].astype(np.int64)
        if K>0 and k!='MID': p=np.where(shadow_ok(p,C,K,th,0.001),p,0)
        t=np.where(p!=0,p,t)
    return t
@numba.njit
def run_s(o,h,l,c,t,X,atrp,k_atr,tick,comm,cash):
    n=c.size;pos=0;lock=0;ep=0.;q=0.;stp=0.;real=0.;eq=np.empty(n);k=0;ei=0
    te=np.empty(n,np.int64);tx=np.empty(n,np.int64);td=np.empty(n,np.int64);tp=np.empty(n);tc=np.empty(n);ts=np.zeros(n,np.int64)
    for i in range(n):
        if pos!=0 and (X>0 or k_atr>0):
            hit=(pos>0 and l[i]<=stp) or (pos<0 and h[i]>=stp)
            if hit:
                px=(min(o[i],stp)-tick) if pos>0 else (max(o[i],stp)+tick)
                fee=(ep+px)*q*comm;pnl=(px-ep)*q*pos-fee;real+=pnl;te[k]=ei;tx[k]=i;td[k]=pos;tp[k]=pnl;tc[k]=fee;ts[k]=1;k+=1;lock=pos;pos=0
        d=t[i]
        if lock!=0 and d!=lock: lock=0
        eff=d if lock==0 else 0
        if eff!=pos:
            if pos!=0:
                px=c[i]-tick if pos>0 else c[i]+tick;fee=(ep+px)*q*comm;pnl=(px-ep)*q*pos-fee;real+=pnl;te[k]=ei;tx[k]=i;td[k]=pos;tp[k]=pnl;tc[k]=fee;k+=1
            if eff!=0:
                px=c[i]+tick if eff>0 else c[i]-tick;q=np.floor(cash/px);ep=px;ei=i
                dist=X if X>0 else k_atr*atrp[i]/100
                stp=ep*(1-dist) if eff>0 else ep*(1+dist)
            pos=eff
        eq[i]=real+((c[i]-ep)*q*pos-ep*q*comm if pos!=0 else 0.)
    if pos!=0:
        px=c[n-1];fee=(ep+px)*q*comm;te[k]=ei;tx[k]=n-1;td[k]=pos;tp[k]=(px-ep)*q*pos-fee;tc[k]=fee;k+=1
    return eq,te[:k],tx[:k],td[:k],tp[:k],tc[:k],ts[:k]
def mae(r):
    te,tx,td=r[1],r[2],r[3];ep=np.where(td>0,C[te]+TICK,C[te]-TICK)
    return np.array([100*((ep[j]-L_[e+1:x+1].min())/ep[j] if d>0 else (H[e+1:x+1].max()-ep[j])/ep[j]) if x>e else 0. for j,(e,x,d) in enumerate(zip(te,tx,td))])
rows=[];RES={}
def rec(nm,prm,r):
    f=metrics(r[0],*r[2:6]);i=metrics(r[0],*r[2:6],0,I_SPLIT);o=metrics(r[0],*r[2:6],I_SPLIT,N);m=mae(r)
    wk=[metrics(r[0],*r[2:6],a,b)['net'] for a,b in WI]
    rows.append(dict(sys=nm,**prm,net=f['net'],pct=f['net']/5,pf=f['pf'],n=f['n'],mae95=np.percentile(m,95),mae99=np.percentile(m,99),maemax=m.max(),p95=f['p95'],p99=f['p99'],mdd=f['mdd'],q=f['net']/f['mdd'],Ipf=i['pf'],Opf=o['pf'],Onet=o['net'],Ocal=o['calmar'],weak=sum(wk),stops=int(r[6].sum()) if len(r)>6 else 0));RES[(nm,)+tuple(prm.values())]=r
R=lambda t,X=0.,k=0.:run_s(O,H,L_,C,t,X,ATRP,k,TICK,COMM,CASH)
rec('V8',{},R(target()))
rec('V8-B4H-MIK',{},R(target(['RSIX','COL','BFLIP','MID'])))
for K,th in itertools.product([8,12,20,30],[1.0,1.2,1.5]): rec('GÖLGE',dict(K=K,th=th),R(target(K=K,th=th)))
for X in [.03,.04,.05,.06,.08]: rec('STOP%',dict(X=X),R(target(),X=X))
for k in [2,3,4,5,6]: rec('STOPATR',dict(k=k),R(target(),k=k))
D=pd.DataFrame(rows);pd.set_option('display.width',290);print(D.round(2).to_string(index=False))
pickle.dump(D,open('reg1.pkl','wb'))
