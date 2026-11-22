exec(open('fix.py').read().split("rows=[];RES={}")[0])
rows=[];RES={}
exec("def rec"+open('fix.py').read().split("def rec")[1].split("Z=np.zeros")[0])
Z=np.zeros(N,bool);prot=pR!=0
def h3(n=24,m=3,lo=18,h=12):
    g=(a30>lo)&(a5>lag(a5,3));mo=mom(n);return vstate(C,g&(mo>=m),g&(mo<=-m),0.,0.,0,F,F,h)
mid=A240<20;t4h=np.where(mid&(t4>0)&(DI240<0),0,np.where(mid&(t4<0)&(DI240>0),0,t4))
pm=h3()
rec('BAZ',{},run_x(C,t6,prot,Z,0.,0.,0,0,TICK,COMM,CASH))
rec('H3',{},run_x(C,np.where(t6!=0,t6,pm).astype(np.int64),prot,Z,0.,0.,0,0,TICK,COMM,CASH))
rec('H4',{},run_x(C,np.where(pR!=0,pR,t4h).astype(np.int64),prot,Z,0.,0.,0,0,TICK,COMM,CASH))
for X in [20,25]:
    midx=A240<X;tx_=np.where(midx&(t4>0)&(DI240<0),0,np.where(midx&(t4<0)&(DI240>0),0,t4))
    for n,m,lo in [(24,3,18),(24,3,15),(24,3,20),(12,3,18),(48,3,18),(24,2,18)]:
        p=h3(n,m,lo);rec('H3+H4',dict(X=X,n_=n,m=m,lo=lo),run_x(C,np.where(pR!=0,pR,np.where(tx_!=0,tx_,p)).astype(np.int64),prot,Z,0.,0.,0,0,TICK,COMM,CASH))
D=pd.DataFrame(rows);pd.set_option('display.width',250);print(D.drop(columns=[c for c in D if c in('kl','cool','act','gb','hd')],errors='ignore').round(2).to_string(index=False))
pickle.dump(RES,open('/home/claude/data/fix2RES.pkl','wb'))
