exec(open("ahf.py").read().split("R7=RS[7]")[0])
R7=RS[7];WEAK=((src==3)|(src==5))
@numba.njit
def run_ahf2(c,t,R,thr,wm,onlyw,slc,mc,trim,X,Y,M,z,tick,comm,cash):
    n=c.size;pos=0;ep=0.;q=0.;ei=0;real=0.;eq=np.empty(n);kk=0;prev=0;mode=0;cdir=0;cxp=0.;cxb=0;xp=0.;xb=0;md=0;wf=False
    te=np.empty(n,np.int64);tx=np.empty(n,np.int64);td=np.empty(n,np.int64);tp=np.empty(n);tc=np.empty(n);ts=np.zeros(n,np.int64)
    for i in range(n):
        ce=(t[i]!=prev);prev=t[i];want=pos;tag=0;newcore=False
        if ce:
            mode=0;want=t[i];newcore=True
        elif mode==0:
            want=t[i]
            if pos!=0 and pos==t[i]:
                adv=(c[i]/ep-1.0)*(-pos)
                if thr[i]>0 and adv>=thr[i] and (onlyw==0 or wf):
                    want=-pos;mode=1;cdir=pos;cxp=c[i];cxb=i;tag=4
                elif trim==1 and ((pos>0 and R[i]>=X) or (pos<0 and R[i]<=100-X)):
                    want=0;mode=2;xp=c[i];xb=i;md=pos;tag=1
            elif pos==0 and t[i]!=0: newcore=True
        elif mode==1:
            g=(c[i]/cxp-1.0)*(-cdir)
            if g<=-slc or i-cxb>=mc:
                mode=0;want=t[i];tag=5;newcore=True
            else:
                want=-cdir
        else:
            if md>0: back=(R[i]<=Y) or (i-xb>=M) or (c[i]>=xp*(1+z))
            else: back=(R[i]>=100-Y) or (i-xb>=M) or (c[i]<=xp*(1-z))
            if back:
                mode=0;want=t[i];tag=2;newcore=True
            else:
                want=0
        if want!=pos:
            if pos!=0:
                px=c[i]-tick if pos>0 else c[i]+tick;fee=(ep+px)*q*comm;pnl=(px-ep)*q*pos-fee;real+=pnl
                te[kk]=ei;tx[kk]=i;td[kk]=pos;tp[kk]=pnl;tc[kk]=fee;ts[kk]=tag;kk+=1
            if want!=0:
                px=c[i]+tick if want>0 else c[i]-tick;q=np.floor(cash/px);ep=px;ei=i
                if mode==0: wf=wm[i]
            pos=want
        eq[i]=real+((c[i]-ep)*q*pos-ep*q*comm if pos!=0 else 0.)
    if pos!=0:
        px=c[n-1];fee=(ep+px)*q*comm;te[kk]=ei;tx[kk]=n-1;td[kk]=pos;tp[kk]=(px-ep)*q*pos-fee;tc[kk]=fee;kk+=1
    return eq,te[:kk],tx[:kk],td[:kk],tp[:kk],tc[:kk],ts[:kk]
rows=[]
for k,cap,slc,mc,ow in itertools.product([3.5,4,4.5,5,6],[0,.08,.10],[.03,.05,.08],[144,288,576],[0,1]):
    thr=k*ATRP/100.
    if cap>0: thr=np.minimum(thr,cap)
    r=run_ahf2(C,t8,R7,thr,WEAK,ow,slc,mc,1,90.,60.,12,.03,TICK,COMM,CASH)
    f=metrics(r[0],*r[2:6]);i=metrics(r[0],*r[2:6],0,I_SPLIT);o=metrics(r[0],*r[2:6],I_SPLIT,N)
    rows.append(dict(k=k,cap=cap,slc=slc,mc=mc,ow=ow,net=f['net'],pf=f['pf'],n=f['n'],mdd=f['mdd'],p99=f['p99'],q=f['net']/f['mdd'],Inet=i['net'],Ipf=i['pf'],Imdd=i['mdd'],Onet=o['net'],Opf=o['pf'],Ocal=o['calmar'],flips=int((r[6]==4).sum())))
D=pd.DataFrame(rows);D.to_pickle('ahf3.pkl');pd.set_option('display.width',250)
D['Isel']=D.Inet/1000*np.sqrt(D.Inet/D.Imdd)
print(D.sort_values('Isel',ascending=False).head(12).round(2).to_string(index=False))
print('k ekseni (medyan net / Ocal) ow=0:');print(D[D.ow==0].groupby('k')[['net','pf','q','Ocal']].median().round(2).to_string())
print('k ekseni ow=1:');print(D[D.ow==1].groupby('k')[['net','pf','q','Ocal']].median().round(2).to_string())
print('cap ekseni:');print(D.groupby(['ow','cap'])[['net','q','Ocal']].median().round(2).to_string())
