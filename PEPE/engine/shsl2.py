exec(open('shsl.py').read().split("\nT={'V11.7 (baz)'")[0])
b=runv();rows=[]
for nm,kw in [('likidasyon short stop %8→6.4',dict(liqS=.064)),('ortak takip short stop %3→2.4',dict(j2S=.024)),('karşı short zarar sınırı %2→1.6',dict(slcS=.016)),
              ('motor stopları ×0.9',dict(liqS=.072,j2S=.027,slcS=.018)),('motor stopları ×0.8',dict(liqS=.064,j2S=.024,slcS=.016)),('motor stopları ×0.7',dict(liqS=.056,j2S=.021,slcS=.014)),('motor stopları ×0.6',dict(liqS=.048,j2S=.018,slcS=.012))]:
    m=runv(**kw);rows.append(dict(v=nm,dnet=m['Net $']-b['Net $'],pf=m['PF'],dshloss=m['Short brüt zarar $']-b['Short brüt zarar $'],dd=m['DD P95/P98/P99'],mdd=m['MaxDD'],upi=m['UPI'],yari=m['İlk / ikinci yarı']))
print('baz:',b['İlk / ikinci yarı'],b['DD P95/P98/P99']);print(pd.DataFrame(rows).round(3).to_string(index=False))
