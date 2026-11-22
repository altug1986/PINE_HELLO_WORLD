exec(open('rollf.py').read().split("\nRES=[];BASE=rec(SH)")[0])
T=B5.index;rows=[]
for K in [20,40,80,160]:
    pf,cal=roll_tr(tpS,K);hpf,hc=tobar(txS,pf),tobar(txS,cal)
    for nm,a,b in [('PF≥1.0',1.,None),('Cal≥0',None,0.),('PF≥0.8',.8,None)]:
        g=gate(hpf,hc,a,b)[teS];s=tpS[~g];k=tpS[g]
        rows.append(dict(K=K,kural=nm,atlanan=len(s),atl_net=s.sum(),atl_ort=s.mean() if len(s) else np.nan,atl_PF=s[s>0].sum()/-s[s<=0].sum() if len(s) else np.nan,kalan_ort=k.mean(),ilk=(T[teS[~g]].min().strftime('%Y-%m') if len(s) else '-')))
print(pd.DataFrame(rows).round(2).to_string(index=False))
print('genel ort işlem',round(tpS.mean(),2))
