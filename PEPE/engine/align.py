exec(open('micro.py').read().split("\nrows=[dict(h='V11.1'")[0])
# 1) hizalama teşhisi: |OI değişimi| hangi bardaki |BTC fiyat hareketi| ile en çok örtüşüyor?
a=np.abs(np.nan_to_num(d1));bp=np.abs(bp1)
for k in [-2,-1,0,1,2]:
    s=np.roll(bp,k) if k else bp
    ok=slice(500,N-500);print(f'|OI d1[t]| ~ |BTC fiyat Δ[t{"+" if k<0 else "-"}{abs(k)}]| korelasyon: {np.corrcoef(a[ok],s[ok])[0,1]:.3f}')
# PEPE OI için aynı test
po=np.log(OI);pd1=np.abs(np.nan_to_num(np.r_[np.nan,np.diff(po)]));pp=np.abs(chg(C,1))
for k in [-1,0,1]:
    s=np.roll(pp,k) if k else pp;ok=slice(500,N-500);print(f'PEPE |OI d1[t]| ~ |PEPE Δ[t{"+" if k<0 else "-"}{abs(k)}]|: {np.corrcoef(pd1[ok],s[ok])[0,1]:.3f}')
# 2) ateşleme motoru BTC OI kaydırma duyarlılığı (pozitif = daha fazla gecikme)
def shift(x,k):
    if k==0: return x
    return np.r_[np.full(k,0.),x[:-k]] if k>0 else np.r_[x[-k:],np.full(-k,0.)]
for k in [-1,0,1,2]:
    z1=shift(oz1,k);ig=vstate(C,(z1>=2)&(bp1>=.6),(z1>=2)&(bp1<=-.6),0.,.03,24,F,F,0).astype(np.int64)
    r=R13(t=chain2(PL,ig,'colust'));h=HH(r);print(f'BTC OI kaydırma {k:+d} bar: net {h["net"]:.0f} | UPI {h["upi"]:.1f} | OOS UPI {h["OOS_upi"]:.1f}')
r=R13(t=base);print('V11.1 (ateşlemesiz):',round(HH(r)['net']))
