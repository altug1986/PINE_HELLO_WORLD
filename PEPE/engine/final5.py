exec(open('arge6.py').read().split("K_=['net'")[0])
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
L,S_=SIG['BRK48'];bt=BASES['V4+MF']
print('komşu (K,gap): net/PF/MDD/Calmar')
for Kk,gp in [(2,48),(3,24),(3,48),(3,96),(4,48)]:
    r=run_pyr(C,bt,L,S_,Kk,gp,0,TICK,COMM,CASH);f=metrics(r[0],*r[2:6]);print(Kk,gp,round(f['net']),round(f['pf'],2),round(f['mdd']),round(f['calmar'],2))
for n2 in [36,48,72,96]:
    h2,l2=donch(n2);r=run_pyr(C,bt,C>h2,C<l2,3,48,0,TICK,COMM,CASH);f=metrics(r[0],*r[2:6]);print('brk',n2,round(f['net']),round(f['pf'],2),round(f['mdd']),round(f['calmar'],2))
V5=run_pyr(C,bt,L,S_,3,48,0,TICK,COMM,CASH);V4=pickle.load(open('V4.pkl','rb'))
V4x3=(V4[0]*3,V4[1],V4[2],V4[3],V4[4]*3,V4[5]*3)
MF=run_t(C,bt,TICK,COMM,CASH)
K=['net','pf','n','win','long','short','comm','sharpe','sortino','calmar','p95','p98','p99','mdd','r2','posm','cv']
M=lambda r,a=0,b=N:metrics(r[0],*r[2:6],a,b)
T=pd.DataFrame({'V4 (500)':M(V4),'V4 x3 (1500)':M(V4x3),'V4+DAR':M(MF),'V5':M(V5),'V4 OOS':M(V4,I_SPLIT),'V5 OOS':M(V5,I_SPLIT)}).T[K]
print(T.round(2).to_string())
uh=V5[6];print('V5 birim dağılımı % (0..3):',(100*uh/N).round(1),'ort birim',round((uh*np.arange(4)).sum()/N,2))
idx=B5.index;cuts=[pd.Timestamp(x,tz='UTC') for x in ['2023-05-01','2024-01-01','2025-01-01','2026-01-01','2026-11-20']];Y={}
for a,b in zip(cuts[:-1],cuts[1:]):
    i0,i1=max(np.searchsorted(idx,a),1),np.searchsorted(idx,b);Y[str(a.year)]={nm:f"{M(r,i0,i1)['net']:.0f}/{M(r,i0,i1)['pf']:.2f}/{M(r,i0,i1)['n']}" for nm,r in [('V4',V4),('V5',V5)]}
i0=np.searchsorted(idx,pd.Timestamp('2026-05-19',tz='UTC'));Y['son6ay']={nm:f"{M(r,i0)['net']:.0f}/{M(r,i0)['pf']:.2f}/{M(r,i0)['n']}" for nm,r in [('V4',V4),('V5',V5)]}
print(pd.DataFrame(Y).T.to_string())
d=pd.Series(V5[0][DAY_I],index=DAY_T);w=d.resample('W').last().diff().dropna();mo=d.resample('ME').last().diff().dropna()
print('V5 poz hafta',round((w>0).mean(),2),'poz ay',round((mo>0).mean(),2),'en kötü ay',round(mo.min()),'en kötü hafta',round(w.min()))
tp=V5[4];rng=np.random.default_rng(1);q=rng.choice(tp,(3000,len(tp)));q=np.where(rng.random(q.shape)<.05,0,q-0.05)
e=np.cumsum(q,1);dd=(np.maximum.accumulate(np.maximum(e,0),1)-e).max(1)
print('MC net P5/P50',np.percentile(e[:,-1],[5,50]).round(0),'DD P95/P99',np.percentile(dd,[95,99]).round(0))
rs=run_pyr(C,bt,L,S_,3,48,0,3e-7,0.0008,CASH);ms=M(rs);print('STRESS',{k:round(float(ms[k]),2) for k in ['net','pf','calmar','mdd','comm']})
f,ax=plt.subplots(2,1,figsize=(12,7),sharex=True,gridspec_kw={'height_ratios':[3,1]})
for r,cl,lb,lw in [(V4,'#bbb','V4 (500$)',.8),(V4x3,'#f0a35e','V4 ×3 (1500$)',.8),(V5,'#1f6feb','V5 (500$ birim, maks 3)',1.3)]:
    ax[0].plot(idx,r[0],color=cl,lw=lw,label=lb);dd_=np.maximum.accumulate(np.maximum(r[0],0))-r[0];ax[1].plot(idx,-dd_,color=cl,lw=lw*.7)
ax[0].axvline(SPLIT,ls='--',color='k',lw=.8);ax[0].legend(loc='upper left');ax[0].grid(alpha=.3);ax[1].grid(alpha=.3);ax[1].set_ylabel('DD $')
ax[0].set_title('PEPE 5M V5 — V4 + DAR HAREKET + KIRILIM EKLEME | basit getiri, MTM (araştırma motoru)');f.tight_layout();f.savefig('pepe_swing_v5.png',dpi=110)
