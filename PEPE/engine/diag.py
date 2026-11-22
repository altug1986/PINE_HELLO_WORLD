exec(open('q8.py').read().split("K=['net','pf','n','calmar'")[0])
import numba
@numba.njit
def ddtv(c,h,l,te,tx,td,tp,ep_,q_):
    n=c.size;closed=0.;peak=0.;mdd=0.;k=0;nt=te.size;j=0
    cur=-1
    for i in range(n):
        while j<nt and tx[j]==i: closed+=tp[j];j+=1
        if closed>peak: peak=closed
        # open trade at bar i (entered before i, exits at/after i)
        o=0.
        while k<nt and tx[k]<i: k+=1
        if k<nt and te[k]<i and tx[k]>=i:
            adv=l[i] if td[k]>0 else h[i]
            o=(adv-ep_[k])*q_[k]*td[k]
        dd=peak-(closed+o)
        if dd>mdd: mdd=dd
    return mdd
def tvdd(r):
    te,tx,td,tp=r[1],r[2],r[3],r[4];ep_=np.where(td>0,C[te]+TICK,C[te]-TICK);q_=np.floor(CASH/ep_)
    return ddtv(C,H,L_,te,tx,td,tp,ep_,q_)
a30,_=adx_htf(30)
V61=pickle.load(open('V61.pkl','rb'))
print('TV-benzeri DD  V6',round(tvdd(V6),1),'(TV 326.6) | V6.1',round(tvdd(V61),1),'(TV 335.5)')
eq=pd.Series(V6[0][DAY_I],index=DAY_T);r30=(eq-eq.shift(30)).dropna()
w=[];s=r30.copy()
while len(w)<6:
    t=s.idxmin();w.append((t,round(s[t])));s=s[(s.index<t-pd.Timedelta(days=45))|(s.index>t+pd.Timedelta(days=45))]
print('V6 en kötü 30g pencereleri (bitiş, net):',[(str(a.date()),b) for a,b in w])
print('son 30g net',round(eq.iloc[-1]-eq[eq.index<=eq.index[-1]-pd.Timedelta(days=30)].iloc[-1]))
src=np.select([pR!=0,t4!=0],[1,2],0)
def win(a,b):
    i0,i1=np.searchsorted(B5.index,pd.Timestamp(a,tz='UTC')),np.searchsorted(B5.index,pd.Timestamp(b,tz='UTC'))
    te,tx,td,tp=V6[1],V6[2],V6[3],V6[4];s_=(tx>=i0)&(tx<i1)
    eng=np.where(src[te[s_]]==1,'RSI','V4');hold=(tx[s_]-te[s_])*5/60
    ret=100*(C[i1-1]/C[i0]-1);rng=100*(C[i0:i1].max()/C[i0:i1].min()-1)
    print(f'{a}→{b}: net {tp[s_].sum():.0f} | işlem {s_.sum()} (RSI {(eng=="RSI").sum()}) | win% {100*(tp[s_]>0).mean():.0f} | ort kazanç {tp[s_][tp[s_]>0].mean():.1f} kayıp {tp[s_][tp[s_]<=0].mean():.1f} | hold med {np.median(hold):.1f}h | fiyat {ret:+.1f}% aralık {rng:.0f}% | flat% {100*(t4[i0:i1]==0).mean():.0f} | CHOP4h med {np.nanmedian(CH240[i0:i1]):.0f} ADX4h {np.nanmedian(A240[i0:i1]):.0f} ADX30 {np.nanmedian(a30[i0:i1]):.0f}')
win('2026-10-20','2026-11-20')
for t,_ in w[:4]: win(str((t-pd.Timedelta(days=30)).date()),str(t.date()))
win('2024-02-01','2024-03-15')
