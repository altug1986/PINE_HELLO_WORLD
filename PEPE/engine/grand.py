exec(open('btc2h.py').read().split("\nC1=[]")[0])
import random
P_=lambda L,S_,h=24:pos(run(C,L,S_,h,TICK,COMM,CASH))
LP={k:LIB[k].astype(np.int64) for k in ['RSIX','MIK','B4H','BFLIP']}
@lru_cache(None)
def colf(fA,fB,h):
    LA,SA=gA&(mA>=3*fA),gA&(mA<=-3*fA);LB,SB=gB&(mB>=2.5*fB),gB&(mB<=-2.5*fB)
    return np.sign(P_(LA,SA,h)+P_(LA|LB,SA|SB,h)).astype(np.int64)
@lru_cache(None)
def midf(m,h,d):
    x=vstate(C,gM&(mM>=m),gM&(mM<=-m),0.,0.,0,F,F,h).astype(np.int64);return deb(x,d) if d>1 else x
@lru_cache(None)
def mikf(d): return deb(LP['MIK'],d) if d>1 else LP['MIK']
DO={0:comps(0)[0]}
@lru_cache(None)
def plf(s,pp,po,sl,pb,re):
    do=DO[s];return vstate(C,(dp6<=-pp)&(do<=-po),(dp6>=pp)&(do<=-po),0.,sl,pb,R7>re,R7<100-re,0).astype(np.int64)
@lru_cache(None)
def exf(s,ep,eo):
    do=DO[s];return np.where((dp6>=ep)&(do<=-eo),1,np.where((dp6<=-ep)&(do<=-eo),-1,0)).astype(np.int64)
Z0=np.zeros(N,np.int64)
SP=dict(fA=[.8,1.,1.2],fB=[.8,1.,1.2],hc=[12,24,36],mm=[2.,2.5,3.,3.5,4.],mh=[6,12,24],md=[1,2,3],mik=[0,1,2,3],rsix=[0,1],b4h=[0,1],bflip=[0,1],mid=[0,1],
        pl=[0,1],pp=[3,4,5],po=[3,4,5],psl=[.03,.05,.08],pb=[24,48,72],pre=[55,60],exo=[0,1],ep=[2,3,4],eo=[1,2,3],
        k=[0.,3.5,4.5,6.],slc=[.02,.03,.05],mc=[72,144,288],X=[85.,90.,95.,101.],Y=[50.,60.,70.],M=[6,12,24],z=[.02,.03,.05],sT=[-1.,1.5,2.,3.])
BC=dict(fA=1.,fB=1.,hc=24,mm=3.,mh=12,md=2,mik=2,rsix=1,b4h=1,bflip=1,mid=1,pl=1,pp=4,po=4,psl=.05,pb=48,pre=55,exo=1,ep=3,eo=2,k=4.5,slc=.03,mc=144,X=90.,Y=60.,M=12,z=.03,sT=2.)
yrs=N/(288*365.25)
def runc(c,s=0):
    E={'RSIX':LP['RSIX'] if c['rsix'] else Z0,'MIK':mikf(c['mik']) if c['mik'] else Z0,'COL':colf(c['fA'],c['fB'],c['hc']),'B4H':LP['B4H'] if c['b4h'] else Z0,
       'BFLIP':LP['BFLIP'] if c['bflip'] else Z0,'MID':midf(c['mm'],c['mh'],c['md']) if c['mid'] else Z0}
    pl=plf(s,c['pp'],c['po'],c['psl'],c['pb'],c['pre']) if c['pl'] else Z0;ex=exf(s,c['ep'],c['eo']) if c['exo'] else Z0
    return run_v14(C,chainY(E,pl),R7,c['k']*ATRP/100.,ZF,ZI,ps,c['sT'],ex,c['slc'],c['mc'],c['X'],c['Y'],c['M'],c['z'],TICK,COMM,CASH,0,0)
def LM(r):
    eq,tx,tp=r[0],r[2],r[4];im=tx<I_SPLIT;pf=lambda m:tp[m&(tp>0)].sum()/max(-tp[m&(tp<=0)].sum(),1e-9)
    e=eq[DAY_I];dd=np.maximum.accumulate(np.maximum(e,0))-e;ul=np.sqrt((dd**2).mean())
    return dict(net=tp.sum(),pf=pf(np.ones(len(tp),bool)),n=len(tp),Inet=tp[im].sum(),Ipf=pf(im),Onet=tp[~im].sum(),Opf=pf(~im),mdd=(np.maximum.accumulate(np.r_[0.,eq])[1:]-eq).max(),upi=(e[-1]/yrs)/ul)
key=lambda c:tuple(c[k] for k in SP);CACHE={}
def ev(c):
    kk=key(c)
    if kk not in CACHE: CACHE[kk]=LM(runc(c))
    return CACHE[kk]
b=ev(BC);print('regresyon baz',{k:round(v,3) for k,v in b.items()})
rnd=random.Random(7)
for i in range(1500): ev({k:rnd.choice(v) for k,v in SP.items()})
for i in range(3000):
    c=dict(BC)
    for k,v in SP.items():
        if rnd.random()<.2: c[k]=rnd.choice(v)
    ev(c)
print('rastgele+yerel',len(CACHE),'süre',round(time.time()-t0),'s')
tIS=1.65+(b['Ipf']-b['pf'])
def J(m): return m['Ipf']-max(0,1.03*b['Inet']-m['Inet'])/400.    # IS: PF büyüt, IS net baz×1.03 altına düşerse ceza
starts=sorted(CACHE.items(),key=lambda kv:J(kv[1]),reverse=True)[:8]
for kk,_ in starts:
    c=dict(zip(SP,kk))
    for _ in range(2):
        for k,v in SP.items():
            best=(J(ev(c)),c[k])
            for val in v:
                c2=dict(c);c2[k]=val;j=J(ev(c2))
                if j>best[0]: best=(j,val)
            c[k]=best[1]
print('tırmanma sonrası',len(CACHE),'süre',round(time.time()-t0),'s')
R=pd.DataFrame([dict(zip(SP,k),**m) for k,m in CACHE.items()]);R.to_pickle('grand.pkl')
hit=R[(R.net>=9200)&(R.pf>=1.65)];print(f"\nTÜM DÖNEM hedef (net≥9200 & PF≥1.65): {len(hit)} / {len(R)} | en yüksek PF (net≥9200): {R[R.net>=9200].pf.max():.3f} | en yüksek net (PF≥1.65): {R[R.pf>=1.65].net.max() if (R.pf>=1.65).any() else 0:.0f}")
isel=R[(R.Inet>=1.03*b['Inet'])&(R.Ipf>=tIS)];print(f"İLK YARI hedefi (IS net≥baz×1.03 & IS PF≥{tIS:.3f}): {len(isel)}")
pd.set_option('display.width',400);cols=list(SP)+['net','pf','n','Inet','Ipf','Onet','Opf','mdd','upi']
if len(hit): print('\nTüm dönem hedefini tutanlar (UPI sırası):');print(hit.sort_values('upi',ascending=False).head(8)[cols].round(3).to_string(index=False))
if len(isel): print('\nİlk yarıda seçilenler → ikinci yarı (IS PF sırası):');print(isel.sort_values('Ipf',ascending=False).head(8)[cols].round(3).to_string(index=False))
F_=R[R.net>=8000].copy();F_['nb']=(F_.net//200)*200;print('\nPareto (net bandı → max PF):');print(F_.groupby('nb').pf.max().round(3).to_string())
