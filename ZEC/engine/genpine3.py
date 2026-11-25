import json, re
M = json.load(open('/home/claude/zecwork/ork_masks.json'))
src = open('/home/claude/pine_hello_world/ZEC/pine/CYBORG_ZEC_1M_DENGELI_TV_V1.1.pine').read()
VN = {'P04_htf4h': 'vP04', 'P07_rsi': 'vP07', 'P08_zema': 'vP08', 'P09_spike': 'vP09', 'P10_bb': 'vP10', 'P15_resid': 'vP15', 'P11_oiCont': 'vP11', 'P12_oiLiq': 'vP12', 'X17_basis': 'vX17',
      'P01_ema': 'vP01', 'P02_st': 'vP02', 'P03_htf30': 'vP03', 'P05_don': 'vP05', 'P06_roc': 'vP06', 'P13_btc': 'vP13', 'P14_xmr': 'vP14', 'X16_flow': 'vX16'}
ENG = [('E0', '25_480'), ('E1', '10_120'), ('E2', '25_120'), ('E3', '25_240')]
SET = {'E0': ['P04_htf4h', 'P07_rsi', 'P08_zema', 'P09_spike', 'P10_bb', 'P15_resid', 'P11_oiCont', 'P12_oiLiq', 'X17_basis'],
       'E1': ['P04_htf4h', 'P07_rsi', 'P08_zema', 'P09_spike', 'P10_bb', 'P15_resid', 'P11_oiCont', 'P12_oiLiq'],
       'E2': ['P04_htf4h', 'P07_rsi', 'P08_zema', 'P09_spike', 'P10_bb', 'P15_resid', 'P11_oiCont', 'P12_oiLiq', 'X17_basis'],
       'E3': ['P04_htf4h', 'P07_rsi', 'P08_zema', 'P09_spike', 'P10_bb', 'P15_resid', 'P11_oiCont', 'P12_oiLiq', 'P01_ema', 'P02_st', 'P03_htf30', 'P05_don', 'P06_roc', 'P13_btc', 'P14_xmr', 'X16_flow']}
decl, sums = [], {}
for e, tb in ENG:
    Ls, Ss = [], []
    for k in SET[e]:
        for sd, lst, sg in (('L', Ls, '1'), ('S', Ss, '-1')):
            m = M[f'{tb}|{k}|{sd}']
            if '1' not in m: continue
            nm = f'M_{e}_{VN[k][1:]}_{sd}'; decl.append(f'var array<bool> {nm} = toArr("{m}")'); lst.append(f'kural({nm}, {VN[k]}, {sg})')
    sums[e] = (' + '.join(Ls) or '0', ' + '.join(Ss) or '0')
SOK = 'BINANCE_ZECUSDT_ZEC-1M-CYBORG_1M_BOTKODU'
head = '''//@version=6
// ═══════════════════════════════════════════════════════════════════════════════════════════════
// CYBORG | ZEC 1M ORKESTRA V2.0  —  BINANCE:ZECUSDT.P, 1 dakika
// REV-03 (2026-11-25): ORKESTRA — tek 500 $ birimde öncelik sıralı 4 motor (V1.1 çatısı üzerine).
//   E0 ana (fiyat+OI+basis, 8 s ufuk, dikkat+fitil) > E1 hızlı (fiyat+OI, 2 s) > E2 (fiyat+OI+basis, 2 s) > E3 (geniş fiyat seti, 4 s)
//   Boşta en öncelikli izinli motor girer. Daha öncelikli motor aynı yönü isterse pozisyonu devralır, tersini isterse kapatır.
//   Yeni oylar: X17 basis (perp / BINANCE:ZECUSDT spot, 1.440 dk z) · P01 EMA40/120 · P02 Supertrend(60, 3√2) · P03 30 dk EMA20 eğimi
//               P05 Donchian 240 orta · P06 120 dk ROC/ATR · P13 BTC 60 dk · P14 XMR 60 dk · X16 mum yönlü hacim akışı 20 dk
//   Kural tabloları ZEC 2021-01→2023-12 (ilk yarı) verisinden sabitlendi (motor başına ayrı ufuk/eşik); TV'de yeniden seçim YOK.
// ARAŞTIRMA MOTORU (1 dk, %0,04×2 + 1 tick, sabit 500 $, basis = index fiyatı): net 8.740 $ (%1.748) · 2.557 işlem · PF 1,52
//   MaxDD 391 · DD P95/P99 220/314 (30 dk MTM) · Sharpe 2,42 · Calmar 3,79 · ilk/ikinci yarı 4.421/4.318 · monkey yön+zaman ✓
//   yıllar 2021 +2.197 · 2022 +2.258 · 2023 −67 · 2024 +1.015 · 2025 +2.532 · 2026 +804
//   TV penceresi (2023-02-06→): beklenen 4.207 $ / 1.682 işlem (long +3.853, short +354). TV'de DOĞRULANMADI.
//   Basis TV'de spot ile hesaplanır (motor index kullanır) → ilk TV export'unda X17 farkı kontrol edilecek.
// ═══════════════════════════════════════════════════════════════════════════════════════════════
strategy("CYBORG | ZEC 1M ORKESTRA V2.0", shorttitle="ZEC1M-ORK", overlay=true, initial_capital=500,
     default_qty_type=strategy.cash, default_qty_value=500, commission_type=strategy.commission.percent, commission_value=0.04,
     slippage=1, process_orders_on_close=true, pyramiding=0, margin_long=0, margin_short=0, max_bars_back=2000)
'''
# V1.1 gövdesinden ortak bölümleri al: sabitler … oylar … rejim … saatlik özellikler (kural tablosu ve motor bölümü yeniden yazılacak)
body = src[src.index('// ── Sabitler'):src.index('int g  = RG == 9')]
body = re.sub(r'// Kural tablosu:.*?\n\n', '', body, flags=re.S)
body = body.replace('string ZOI = "BINANCE:ZECUSDT.P_OI"   // [REV-02]', 'string ZOI = "BINANCE:ZECUSDT.P_OI"\nstring ZSP = "BINANCE:ZECUSDT"   // [REV-03] spot, basis için')
body = body.replace('float TS = 40.0              // [REV-02] 28 → 40\nint   TH = 1920              // [REV-02] 3840 → 1920', 'float TS = 28.0              // [REV-03] bütün motorlar 28×ATR')
body = body.replace('int K_L = 2\nint K_S = 2\nint K_DIKKAT = 2\n', '')
body = re.sub(r'string M_P\w+ = "[01]+".*\n', '', body)
body = body.replace('float bat1 = request.security(BAT, "1", close)', 'float bat1 = request.security(BAT, "1", close)\nfloat spot1 = request.security(ZSP, "1", close)   // [REV-03]')
ext = '''// [REV-03] Yeni oylar
float e40 = ta.ema(close, 40)
float e120 = ta.ema(close, 120)
int vP01 = e40 > e120 ? 1 : e40 < e120 ? -1 : 0
var float fu = na
var float fd = na
var int vP02 = 1
float mST = 3 * R2
if not na(atr)
    float up = hl2 - mST * atr
    float dn = hl2 + mST * atr
    if na(fu)
        fu := up
        fd := dn
    else
        float fuP = fu
        float fdP = fd
        fu := close[1] > fuP ? math.max(up, fuP) : up
        fd := close[1] < fdP ? math.min(dn, fdP) : dn
        vP02 := close > fdP ? 1 : close < fuP ? -1 : vP02
float s30 = request.security(syminfo.tickerid, "30", (ta.ema(close, 20) - ta.ema(close, 20)[1])[1], lookahead = barmerge.lookahead_on)
int vP03 = na(s30) ? 0 : s30 > 0 ? 1 : s30 < 0 ? -1 : 0
float donM = (ta.highest(high, 240)[1] + ta.lowest(low, 240)[1]) / 2
int vP05 = na(donM) ? 0 : close > donM ? 1 : close < donM ? -1 : 0
float roc = nz((close - close[120]) / atr)
int vP06 = roc > R2 ? 1 : roc < -R2 ? -1 : 0
float bt60 = nz((btc1 / btc1[60] - 1) * 100)
int vP13 = bt60 > 0.1 * R2 ? 1 : bt60 < -0.1 * R2 ? -1 : 0
float xm60 = nz((xmr1 / xmr1[60] - 1) * 100)
int vP14 = xm60 > 0.2 * R2 ? 1 : xm60 < -0.2 * R2 ? -1 : 0
float flw = nz(math.sum(math.sign(close - open) * volume, 20) / math.sum(volume, 20))
int vX16 = flw > 0.15 ? 1 : flw < -0.15 ? -1 : 0
float bs = (close / spot1 - 1) * 1e4
float bz = nz((bs - ta.sma(bs, 1440)) / ta.stdev(bs, 1440, false))
int vX17 = bz <= -2 ? 1 : bz >= 2 ? -1 : 0
int vP07 = 0   // RSI: hiçbir tabloda seçilmedi

'''
body = body.replace('// ── Rejim: 4s ADX', ext + '// ── Rejim: 4s ADX')
tail = '''int g  = RG == 9 ? -1 : RG + 1
int rc = g < 0 ? -1 : ((g * 8 + K * 4 + Vh * 2 + A) * 2 + De) * 2 + Xb

// ── Kural tabloları (motor başına; karakter i = rejim hücresi i) ──────────────────────────────────
toArr(string m) =>
    array<bool> a = array.new_bool(96, false)
    for i = 0 to 95
        array.set(a, i, str.substring(m, i, i + 1) == "1")
    a
kural(array<bool> m, int v, int side) => rc >= 0 and rc < 96 ? (v == side and array.get(m, rc) ? 1 : 0) : 0
''' + '\n'.join(decl) + f'''

int L0 = {sums['E0'][0]}
int S0 = {sums['E0'][1]}
int L1 = {sums['E1'][0]}
int S1 = {sums['E1'][1]}
int L2 = {sums['E2'][0]}
int S2 = {sums['E2'][1]}
int L3 = {sums['E3'][0]}
int S3 = {sums['E3'][1]}

// ── Giriş izinleri: fitil (son 10 dk mum) + E0 long dikkati ───────────────────────────────────────
float Hw = ta.highest(high, 10)
float Lw = ta.lowest(low, 10)
float Ow = open[9]
float rw = math.max(Hw - Lw, 1e-9)
bool fL = (math.min(Ow, close) - Lw) / rw >= 0.4
bool fS = (Hw - math.max(Ow, close)) / rw >= 0.4
bool batUp = bat1 > ta.ema(bat1, 1440) and nz(bat1 / bat1[1440]) > 1
bool dikkatL = nz(xmr1 / xmr1[1440] - 1) >= 0.04 or batUp
bool aL0 = L0 >= 2
bool aS0 = S0 >= 2 and not aL0
bool eL0 = L0 >= 2 + (dikkatL ? 2 : 0) and fL
bool eS0 = S0 >= 2 and fS
bool aL1 = L1 >= 2
bool aS1 = S1 >= 1 and not aL1
bool eL1 = L1 >= 2 and fL
bool eS1 = S1 >= 1 and fS
bool aL2 = L2 >= 2
bool aS2 = S2 >= 1 and not aL2
bool eL2 = L2 >= 2
bool eS2 = S2 >= 1
bool aL3 = L3 >= 2
bool aS3 = S3 >= 2 and not aL3
bool eL3 = L3 >= 2 and fL
bool eS3 = S3 >= 2 and fS

// ── ORKESTRA: öncelik, devralma, çıkış, giriş (araştırma motoru simN ile aynı sıra) ─────────────────
var int own = -1
var int ti = 0
var int noS = -1
bool spk = nz(close / close[10] - 1) >= 0.03
if spk
    noS := bar_index + 60
int w0 = aL0 ? 1 : (aS0 and bar_index > noS) ? -1 : 0
int w1 = aL1 ? 1 : (aS1 and bar_index > noS) ? -1 : 0
int w2 = aL2 ? 1 : (aS2 and bar_index > noS) ? -1 : 0
int w3 = aL3 ? 1 : (aS3 and bar_index > noS) ? -1 : 0
array<int> W = array.from(w0, w1, w2, w3)
array<int> THa = array.from(3840, 480, 960, 480)
array<bool> ELa = array.from(eL0, eL1, eL2, eL3)
array<bool> ESa = array.from(eS0, eS1, eS2, eS3)
int pos = strategy.position_size > 0 ? 1 : strategy.position_size < 0 ? -1 : 0
if pos == 0
    own := -1
else if own < 0
    own := 3
bool guard = pos == -1 and spk
if pos != 0 and not guard and own > 0
    for j = 0 to own - 1
        if array.get(W, j) == pos
            own := j
            ti := bar_index
            break
bool ex = false
int blk = 0
if pos != 0
    bool tout = bar_index - ti >= array.get(THa, own)
    bool opp = false
    for j = 0 to own
        int wj = array.get(W, j)
        if wj != 0 and wj != pos
            opp := true
    ex := guard or opp or tout
    blk := tout and array.get(W, own) == pos ? pos : 0
if ex
    strategy.close_all(comment = "EXIT-ALL_{SOK}")
    pos := 0
    own := -1
if pos != 0 and array.get(W, own) == pos
    ti := bar_index
if pos == 0 and not na(atr)
    for k = 0 to 3
        int wk = array.get(W, k)
        bool izin = wk == 1 ? array.get(ELa, k) : wk == -1 ? array.get(ESa, k) : false
        if wk != 0 and wk != blk and izin
            float ep = close + syminfo.mintick * wk
            own := k
            ti := bar_index
            if wk == 1
                strategy.entry("L", strategy.long, comment = "ENTER-LONG_{SOK}")
                strategy.exit("SL-L", from_entry = "L", stop = ep - TS * atr, comment = "EXIT-ALL_{SOK}")
            else
                strategy.entry("S", strategy.short, comment = "ENTER-SHORT_{SOK}")
                strategy.exit("SL-S", from_entry = "S", stop = ep + TS * atr, comment = "EXIT-ALL_{SOK}")
            break

plot(own, "Sahip motor", color.gray, display = display.data_window)
plot(rc, "Rejim hücresi", color.gray, display = display.data_window)
plot(L0 - S0, "E0 oy farkı", color.blue, display = display.data_window)
'''
out = head + '\n' + body + tail
open('/home/claude/pine_hello_world/ZEC/pine/CYBORG_ZEC_1M_ORKESTRA_V2.0.pine', 'w').write(out)
print(len(out.splitlines()), 'satır ·', len(decl), 'tablo dizisi'); print([l for l in out.splitlines() if 'TS' in l or 'TH' in l][:6])
