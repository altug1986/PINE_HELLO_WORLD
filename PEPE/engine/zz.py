import pickle, numpy as np, pandas as pd, numba
from eng import *
@numba.njit
def zigzag(c,th):
    n=c.size;piv=np.empty(n,np.int64);k=0;d=0;ext=c[0];ei=0;piv[k]=0;k+=1
    for i in range(1,n):
        if d>=0:
            if c[i]>ext: ext=c[i];ei=i
            elif c[i]<=ext*(1-th):
                if d==1 or k==1: piv[k]=ei;k+=1
                d=-1;ext=c[i];ei=i
                continue
        if d<=0:
            if c[i]<ext: ext=c[i];ei=i
            elif c[i]>=ext*(1+th):
                piv[k]=ei;k+=1;d=1;ext=c[i];ei=i
    piv[k]=ei;k+=1
    return piv[:k]
V7=pickle.load(open('V7.pkl','rb'));yrs=N/(288*365.25);cost=0.0010
print('PEPE 5m kapanış: başlangıç',C[0],'son',C[-1],'| al-tut',round(100*(C[-1]/C[0]-1)),'%')
rows=[]
for th in [0.005,0.01,0.02,0.03,0.05,0.10,0.20,0.50]:
    p=zigzag(C,th);s=np.abs(C[p[1:]]/C[p[:-1]]-1);dur=(p[1:]-p[:-1])*5/60
    simp=s.sum();net=np.maximum(s-cost,0).sum();comp=np.exp(np.log1p(np.maximum(s-cost,-.99)).sum())
    rows.append(dict(eşik=f'%{th*100:g}',dalga=len(s),yıl_başı=round(len(s)/yrs),medyan_boy=f'%{100*np.median(s):.1f}',medyan_süre_saat=round(np.median(dur),1),
                     basit_toplam=f'%{100*simp:,.0f}',maliyet_sonrası=f'%{100*net:,.0f}',dolar_500=f'{500*net:,.0f}',bileşik=f'%{100*(comp-1):.2e}',V7_yakalama=f'%{100*V7[0][-1]/(500*net):.2f}'))
pd.set_option('display.width',250);print(pd.DataFrame(rows).to_string(index=False))
print('V7 net',round(V7[0][-1]),'$ = %',round(100*V7[0][-1]/500))
