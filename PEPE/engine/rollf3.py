exec(open('rollf.py').read().split("\nRES=[];BASE=rec(SH)")[0])
from scipy.stats import spearmanr
p=tpS
for K in [20,40,80]:
    pf,_=roll_tr(p,K);x=pf[K-1:-1];y=p[K:]
    q=pd.qcut(x,5,labels=['Q1 en kötü','Q2','Q3','Q4','Q5 en iyi'])
    t=pd.DataFrame({'q':q,'y':y}).groupby('q',observed=True).y.agg(n='count',sonraki_ort='mean',sonraki_PF=lambda v:v[v>0].sum()/-v[v<=0].sum())
    t['geçmiş_PF_aralığı']=pd.Series(x).groupby(q.codes).agg(lambda v:f'{v.min():.2f}-{v.max():.2f}').values
    b=len(p)//K;bp=[p[i*K:(i+1)*K] for i in range(b)];bpf=np.array([v[v>0].sum()/-v[v<=0].sum() for v in bp])
    print(f'K={K}: ardışık {K}-işlem bloklarının PF korelasyonu (Spearman) {spearmanr(bpf[:-1],bpf[1:])[0]:+.2f}');print(t.round(2).to_string())
e=pd.Series(SH[0][DAY_I],index=DAY_T);m=e.resample('ME').last().diff().dropna();w=e.resample('W').last().diff().dropna()
print(f'aylık net otokorelasyon {m.autocorr():+.2f} | haftalık {w.autocorr():+.2f}')
