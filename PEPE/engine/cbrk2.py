exec(open('cbrk.py').read().split("\nB=row(SH5)")[0])
for s in [0,1,2]:
    sh_=runc(A4,s);p=np.zeros(N,np.int64)
    for e_,x_,d_ in zip(sh_[1],sh_[2],sh_[3]): p[e_:x_]=d_
    r=cbrk(C,p,2,100.,0,288,0,TICK,COMM,CASH);f=M_(r);b=M_(sh_)
    print(f"s={s}: V11.5 {b['net']:.0f}/PF {b['pf']:.3f}/P99 {b['p99']:.0f}/MaxDD {b['mdd']:.0f} → kesici {f['net']:.0f}/PF {f['pf']:.3f}/P99 {f['p99']:.0f}/MaxDD {f['mdd']:.0f} | tetik {r[6]}")
