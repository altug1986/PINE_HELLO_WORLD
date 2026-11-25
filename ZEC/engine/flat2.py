exec(open('/home/claude/zecwork/flat1.py').read().split("eq, T = z03.simN3(")[0])
eq, T = z03.simN3(o, h, l, c, a, AL, AS, EL, ES, TSa, THa, spk, 60); out.append(met('V2.2', eq, T))
for m in (1, 3, 5, 10):
    eq, T = z03.simN6(o, h, l, c, a, AL, AS, EL, ES, TSa, THa, spk, 60, m); out.append(met(f'oybirliği giriş, {m} bar', eq, T)); np.save(f'/home/claude/zecwork/T_OB{m}.npy', T); np.save(f'/home/claude/zecwork/eq_OB{m}.npy', eq[::30])
pd.set_option('display.width', 260); print(pd.DataFrame(out).round(2).to_string(index=False))
