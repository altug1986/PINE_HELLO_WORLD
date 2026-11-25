exec(open('/home/claude/zecwork/mega5.py').read().split("rows = []; t0 = time.time()")[0])
import json, glob
tv = np.load('/home/claude/zecwork/tv_V.npz')
for k in tv.files:
    if k not in V: V[k] = tv[k].astype(np.float32)
R = RL[0]; Ri = R[:mid]; out = {}
NEED = {(25, 480): ['P04_htf4h', 'P07_rsi', 'P08_zema', 'P09_spike', 'P10_bb', 'P15_resid', 'P11_oiCont', 'P12_oiLiq', 'X17_basis'],
        (10, 120): ['P04_htf4h', 'P07_rsi', 'P08_zema', 'P09_spike', 'P10_bb', 'P15_resid', 'P11_oiCont', 'P12_oiLiq'],
        (25, 120): ['P04_htf4h', 'P07_rsi', 'P08_zema', 'P09_spike', 'P10_bb', 'P15_resid', 'P11_oiCont', 'P12_oiLiq', 'X17_basis'],
        (25, 240): ['P04_htf4h', 'P07_rsi', 'P08_zema', 'P09_spike', 'P10_bb', 'P15_resid', 'P11_oiCont', 'P12_oiLiq', 'P01_ema', 'P02_st', 'P03_htf30', 'P05_don', 'P06_roc', 'P13_btc', 'P14_xmr', 'X16_flow']}
for (thr, hz), comps in NEED.items():
    fi = F[hz][:mid]
    for k in comps:
        vi = np.asarray(V[k])[:mid]
        for side in (1, -1):
            m_ = ['0'] * 96
            for val in range(96):
                mm = (vi == side) & (Ri == val)
                if mm.sum() >= 400 and np.nanmean(fi[mm] * side) >= thr: m_[val] = '1'
            out[f'{thr}_{hz}|{k}|{"L" if side > 0 else "S"}'] = ''.join(m_)
json.dump(out, open('/home/claude/zecwork/ork_masks.json', 'w')); print(len(out), 'maske ·', sum(v.count('1') > 0 for v in out.values()), 'dolu')
T = np.load('/home/claude/zecwork/T_ORK4.npy'); t = pd.concat([pd.read_parquet(f_, columns=['time']).time for f_ in sorted(glob.glob('/home/claude/master/*.parquet'))]).values; ts_ = pd.to_datetime(t, unit='ms')
pd.DataFrame({'giris_utc': ts_[T[:, 0].astype(int)], 'cikis_utc': ts_[T[:, 1].astype(int)], 'yon': T[:, 2].astype(int), 'giris': T[:, 3], 'cikis': T[:, 4], 'net': T[:, 5].round(2), 'motor': T[:, 7].astype(int)}).to_csv('/home/claude/pine_hello_world/ZEC/research/ORKESTRA4_MOTOR_ISLEMLER.csv', index=False)
w = T[ts_[T[:, 0].astype(int)] >= pd.Timestamp('2023-02-06')]; print('TV penceresi işlem', len(w), 'net', round(w[:, 5].sum()), 'long', round(w[w[:, 2] > 0, 5].sum()), 'short', round(w[w[:, 2] < 0, 5].sum()))
