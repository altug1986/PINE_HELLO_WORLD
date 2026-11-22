exec(open('micro.py').read().split("\nrows=[dict(h='V11.1'")[0])
r=R13(t=base);eq,te,tx,td,tp,tc,ts=r;T=B5.index
a=T.searchsorted(pd.Timestamp('2026-11-08 21:35',tz='UTC'));b=T.searchsorted(pd.Timestamp('2026-11-08 22:10',tz='UTC'))
print('bar(TV+3)   close     tgt PL EX R7 | RSIX MIK COL B4H BFLIP MID | dp6 do6')
for i in range(a,b): print((T[i]+pd.Timedelta(hours=3)).strftime('%m-%d %H:%M'),f'{C[i]:.7f}',base[i],PL[i],EX[i],round(R7[i]),'|',*[ENG[k][i] for k in ORDER],'|',round(dp6[i],2),round(do6[i],2))
m=(tx>=a-50)&(te<b+5)
print(pd.DataFrame({'giris':(T[te[m]]+pd.Timedelta(hours=3)).strftime('%m-%d %H:%M'),'cikis':(T[tx[m]]+pd.Timedelta(hours=3)).strftime('%m-%d %H:%M'),'yon':td[m],'pnl':tp[m].round(2),'tag':ts[m]}).to_string(index=False))
dur=tx-te;print('toplam',len(tp),'| ≤2 mum süren:',(dur<=2).sum(),'net',tp[dur<=2].sum().round(0),'| ≤1:',(dur<=1).sum(),'net',tp[dur<=1].sum().round(0))
# ardışık: çıkış -> 1 mum içinde ters/yeni giriş -> 1-2 mumda tekrar dönüş
k=0
for j in range(1,len(te)-1):
    if dur[j]<=2 and te[j]-tx[j-1]<=1: k+=1
print('çıkış/giriş+≤2 mumda tekrar dönüş zinciri:',k)
