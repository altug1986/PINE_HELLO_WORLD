exec(open('q5.py').read().split("rec('V4',{},t4)")[0])
rec('V4',{},t4)
def rsi(n): u=rma(np.where(d>0,d,0.),n);w=rma(np.where(d<0,-d,0.),n);return 100-100/(1+u/w)
RS={n:rsi(n) for n in [7,14,21]};PR={}
for n,lo,ex,sl,mb in itertools.product([7,14,21],[5,8,10,12,15],[40,50,60],[.05,.08,.12],[48,144,288]):
    R_=RS[n];p=vstate(C,R_<lo,R_>100-lo,0.,sl,mb,R_>ex,R_<100-ex,0);PR[(n,lo,ex,sl,mb)]=p;rec('RSI',dict(rl=n,lo=lo,ex=ex,sl=sl,mb=mb),p)
D=pd.DataFrame(rows);g=D[(D.fam=='RSI')&(D.Ical>0)&(D.Ocal>0)&(D.n>=60)].copy();pd.set_option('display.width',250)
print('RSI IS+&OOS+',round(len(g)/(len(D)-1),2));print(g.sort_values('calmar',ascending=False).head(8).dropna(axis=1,how='all').round(2).to_string(index=False))
# combine top RSI sleeves with V4: (a) 2-bot portfolio sum, (b) single-bot override (RSI wins)
out=[]
for _,r in g.sort_values('calmar',ascending=False).head(15).iterrows():
    k=(int(r.rl),int(r.lo),int(r.ex),r.sl,int(r.mb));p=PR[k]
    rr=run_t(C,p.astype(np.int64),TICK,COMM,CASH);eq=V4[0]+rr[0]
    cat=lambda j:np.concatenate([V4[j],rr[j]]);o=np.argsort(cat(2))
    tr=[cat(j)[o] for j in range(2,6)];f=metrics(eq,*tr);fo=metrics(eq,*tr,I_SPLIT,N);fi=metrics(eq,*tr,0,I_SPLIT)
    ov=np.where(p!=0,p,t4);ro=run_t(C,ov.astype(np.int64),TICK,COMM,CASH);m2=metrics(ro[0],*ro[2:]);m2o=metrics(ro[0],*ro[2:],I_SPLIT,N);m2i=metrics(ro[0],*ro[2:],0,I_SPLIT)
    out.append(dict(k=k,sum_net=f['net'],sum_cal=f['calmar'],sum_mdd=f['mdd'],sum_p98=f['p98'],sum_Ical=fi['calmar'],sum_Ocal=fo['calmar'],ov_net=m2['net'],ov_pf=m2['pf'],ov_cal=m2['calmar'],ov_mdd=m2['mdd'],ov_p98=m2['p98'],ov_Ical=m2i['calmar'],ov_Ocal=m2o['calmar'],ov_n=m2['n']))
print(pd.DataFrame(out).round(2).to_string(index=False))
pickle.dump(PR,open('/home/claude/data/PR.pkl','wb'))
