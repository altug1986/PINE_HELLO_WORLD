exec(open('joint2.py').read().split("\nJ4=dict(")[0])
@numba.njit
def vst2(c,L,S,slL,slS,mb,xl,xs):
    n=c.size;o=np.zeros(n,np.int64);s=0;ep=0.;ei=0
    for i in range(n):
        if s!=0:
            r=(c[i]/ep-1.)*s;sl=slL if s>0 else slS
            if r<=-sl or i-ei>=mb or (s>0 and xl[i]) or (s<0 and xs[i]): s=0
            elif s>0 and S[i]: s=-1;ep=c[i];ei=i
            elif s<0 and L[i]: s=1;ep=c[i];ei=i
        else:
            if L[i]: s=1;ep=c[i];ei=i
            elif S[i]: s=-1;ep=c[i];ei=i
        o[i]=s
    return o
@numba.njit
def run16(c,t,R,thrL,thrS,ps,sT,ex,slcL,slcS,mc,X,Y,M,z,hs,tick,comm,cash):
    n=c.size;pos=0;ep=0.;q=0.;ei=0;real=0.;eq=np.empty(n);kk=0;prev=0;mode=0;cdir=0;cxp=0.;cxb=0;xp=0.;xb=0;md=0
    te=np.empty(n,np.int64);tx=np.empty(n,np.int64);td=np.empty(n,np.int64);tp=np.empty(n);tc=np.empty(n);ts=np.zeros(n,np.int64)
    for i in range(n):
        ce=(t[i]!=prev);prev=t[i];want=pos;tag=0
        if ce:
            mode=0;want=t[i]
        elif mode==3:
            want=0
        elif mode==0:
            want=t[i]
            if pos!=0 and pos==t[i]:
                adv=(c[i]/ep-1.0)*(-pos);trs=ps[i]*pos;th=thrL[i] if pos>0 else thrS[i]
                if th>0 and adv>=th:
                    want=-pos;mode=1;cdir=pos;cxp=c[i];cxb=i;tag=4
                elif pos<0 and hs>0 and adv>=hs:
                    want=0;mode=3;tag=7
                elif ex[i]==pos:
                    want=0;mode=2;xp=c[i];xb=i;md=pos;tag=6
                elif ((pos>0 and R[i]>=X) or (pos<0 and R[i]<=100-X)) and (sT<0 or trs<sT):
                    want=0;mode=2;xp=c[i];xb=i;md=pos;tag=1
        elif mode==1:
            g=(c[i]/cxp-1.0)*(-cdir);slc=slcS if cdir>0 else slcL
            if g<=-slc or i-cxb>=mc:
                mode=0;want=t[i];tag=5
            else:
                want=-cdir
        else:
            if md>0: back=(R[i]<=Y) or (i-xb>=M) or (c[i]>=xp*(1+z))
            else: back=(R[i]>=100-Y) or (i-xb>=M) or (c[i]<=xp*(1-z))
            if back:
                mode=0;want=t[i];tag=2
            else:
                want=0
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
Fb=np.zeros(N,bool);dp6_,do6_=dpn(6),pchg(0,6);bz6,_=bfe(0,6);dp12,do12=dpn(12),pchg(0,12);bz12,bp12=bfe(0,12)
LLq,SLq=(dp6_<=-4)&(do6_<=-4),(dp6_>=4)&(do6_<=-4)
assert (vst2(C,LLq,SLq,.08,.08,72,R7>55,R7<45)==plf(0,4,4,.08,72,55)).all()
EXJ=np.where((dp6_>=3)&(do6_<=-3)&(bz6<=-1.5),1,np.where((dp6_<=-3)&(do6_<=-3)&(bz6<=-1.5),-1,0)).astype(np.int64)
J2L=(dp12>=4)&(do12>=4)&(bz12>=1)&(bp12>=.5);J2S=(dp12<=-4)&(do12>=4)&(bz12>=1)&(bp12<=-.5)
M_=lambda r,a=0,b=N:metrics(r[0],*r[2:6],a,b)
def runv(ks=1.,liqS=.08,j2S=.03,slcS=.02,hs=0.):
    E={'RSIX':LP['RSIX'],'MIK':Z0,'COL':colf(1.,1.,24),'B4H':LP['B4H'],'BFLIP':Z0,'MID':midf(4.,12,3)}
    pl=vst2(C,LLq,SLq,.08,liqS,72,R7>55,R7<45);j2=vst2(C,J2L,J2S,.03,j2S,48,Fb,Fb)
    th=4.5*ATRP/100.;sh_=run16(C,chainY(E,pl,j2,'COL'),R7,th,th*ks,ps,2.,EXJ,.02,slcS,144,90.,60.,12,.03,hs,TICK,COMM,CASH)
    p=np.zeros(N,np.int64)
    for e_,x_,d_ in zip(sh_[1],sh_[2],sh_[3]): p[e_:x_]=d_
    r=cbrk(C,p,2,100.,0,288,0,TICK,COMM,CASH);f=M_(r);td,tp=r[3],r[4];s=td<0;_,up=U(r)
    return {'Net $':f['net'],'PF':f['pf'],'İşlem':f['n'],'Short net $':tp[s].sum(),'Short brüt zarar $':-tp[s&(tp<0)].sum(),'Short PF':tp[s&(tp>0)].sum()/-tp[s&(tp<=0)].sum(),
            'DD P95/P98/P99':f"{f['p95']:.0f} / {f['p98']:.0f} / {f['p99']:.0f}",'MaxDD':f['mdd'],'UPI':up,'İlk / ikinci yarı':f"{r[0][I_SPLIT-1]:.0f} / {r[0][-1]-r[0][I_SPLIT-1]:.0f}"}
T={'V11.7 (baz)':runv()}
print('regresyon baz net',round(T['V11.7 (baz)']['Net $'],1))
T['karşı poz eşiği short ×0.8']=runv(ks=.8)
T['motor stopları short ×0.8']=runv(liqS=.064,j2S=.024,slcS=.016)
T['ikisi birden']=runv(ks=.8,liqS=.064,j2S=.024,slcS=.016)
for h in [.06,.08,.10,.12]: T[f'yeni sabit short stop %{h*100:g}']=runv(hs=h)
for k in [.7,.9]: T[f'karşı poz short ×{k}']=runv(ks=k)
D=pd.DataFrame(T);pd.set_option('display.width',400);print(D.round(2).to_string())
