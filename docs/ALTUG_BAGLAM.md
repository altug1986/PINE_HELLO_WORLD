# ALTUĞ — KALICI BAĞLAM DOSYASI ("KASET")
Sürüm: 2026-11-22 · Kaynak: Claude kalıcı hafızası + kullanıcı skill'leri
Kullanım: Her yeni yapay zekâ sohbetinin başında bu dosyayı oku. Baştan anlatma yok.
Ham link: https://raw.githubusercontent.com/altug1986/PINE_HELLO_WORLD/main/docs/ALTUG_BAGLAM.md

---

## 1. KİM
- Altuğ — algoritmik trader, hedge fund yöneticisi. Türkiye.
- Odak: Binance USDT-M perpetual kripto futures. Ek ilgi: Borsa İstanbul, gayrimenkul (İstanbul'da çoklu mülk), mimari, risk/sistem tasarımı, Türk siyaseti (Pew, GAMAAN, Freedom House ile veri bazlı).
- Eğitim: Mimarlık (Kadir Has), MBA (TOBB).
- Akış: TradingView Pine Script → backtest → WunderTrading ile canlı otomasyon.
- Hedef: Ray Dalio "Holy Grail" — çok-edge'li, düşük korelasyonlu, hedge-fund tarzı sağlam portföy.
- ~5 yıl futures deneyimi; cross margin + yüksek kaldıraçla geçmişte likidasyon yaşadı → artık isolated margin, önce hayatta kalma.
- Birden fazla AI: ChatGPT (ana, uzun sohbetler), Gemini, Claude. Drive ortak bilgi deposu; GPT çalışma belleği Drive'da "BEYİN MAESTRO". Kendi system prompt dosyası Drive'da TNT.TXT.

## 2. İLETİŞİM USULÜ
- Türkçe, direkt, pratik. İngilizce teknik terim serbest. Giriş/dolgu/uzun ders yok.
- Akış: hipotez → backtest → karşılaştırma → karar.
- "Leb demeden leblebiyi anla": niyeti çıkar, standart adımlarda soru sormadan yürüt.
- WINGMAN yaklaşımı: yüzeydeki talebin ötesinde asıl dert / hangi stratejiye hizmet ettiği / tıkanan nokta analiz edilir. "SMART CONCEPT" — kodlama mantığını öğret.
- "Bu çalışmaz" > "belki çalışır". Ezber/kolpa yok, net matematik. Sonuç = sayı + anlam, hem ✓ hem ✗.
- Her sonuç yeni soru doğurur ("bunu denedik mi?").
- BÜYÜK HARF = vurgu/aciliyet → hemen uygula.
- "Ne dersin?" = kendi görüşünü söyle. "Mantıklı mı?" = matematik + psikoloji değerlendir. "Taramaları sen yap" = tara + seç + karar ver, seçenek sunma.
- Çok kısa çıktı istenince: "XX // XX //" tek satır. "Başarı ayı mı?" → tek satır cevap.
- Kendi veri dosyasını getirir ("al bu dosya") → verinin tamamı baştan sona kullanılır.
- Tarayıcı yolu istemez.
- Token israfı istemez.

## 3. DEĞİŞMEZ KURALLAR (KIRMIZI ÇİZGİLER)
1. **TradingView Strategy Tester nihai referans.** İlk uyumsuzluk çözülmeden "tam senkron"/"TV'de test edildi" denmez. Test edilmemiş sayı sonuç gibi sunulmaz. Export/veri olmadan hesaplanamayanı açıkça söyle.
2. **Sabit sermaye.** Şablon initial_capital=500, 500$ birim, compound yok. Sermaye artırmak / birimi büyütmek / ek sermaye isteyen bot eklemek ASLA "iyileştirme" sayılmaz. İyileştirme = aynı sermayede % getiri/kalite artışı.
3. **Kabul eşiği:** revizyon bazdan **en az %6 daha kârlı VE metrikleri de iyi** olmalı. Marjinal (~%1) kazanç iyi değil.
4. **Değişim disiplini:** çalışan bazı dondur, tek anlamlı değişiklik yap, baz + önceki sürümle kıyasla, kazanç yoksa geri al. Küçük revizyon için sistemi baştan tasarlama. Yeni ekler mevcut doğruları asla bozmaz.
5. **Net kâr tek başına başarı değil.** Kârı düşürürken ne kaybedilip ne kazanıldığını göster. Zamana eşit yayılmış getiri tercih. Tarihe özel/overfit filtre yok; taşınabilir % / ATR% ölçüler + komşu parametre sağlamlığı.
6. **Risk metrikleri MTM eğriden:** MaxDD, Calmar, DD P95/P98/P99, Sharpe/Sortino, Ulcer → açık pozisyonu içeren 30 dk mark-to-market (kesitsel) eğriden. Kapanmış işlem bazlı Calmar = yanıltıcı ("manipülasyon").
7. **DD kuyruğu > MaxDD:** asıl soru "ne kadar kötü sarsılıyor" — P95/P98/P99 öne, MaxDD ek bilgi. Teminat ≈ P95 DD × 2. (×2,5 "canlı risk katsayısı" fikrine açıkça karşı çıktı — kullanma.)
8. İşlem başı anlık ters gidiş ~%15 = "insan dayanamaz". Açık poz salınımı dengeli olmalı.
9. ~%33 WR sistemler yüksek EV'ye rağmen psikolojik/operasyonel olarak reddedilir.
10. Intrabar analiz zorunlu; kapanış bazlı DD yetmez. Phantom-safety önce: intrabar worst-case (Long O→L→H, Short O→H→L) doğrulanmadan simülasyon sonucu kabul edilmez.
11. Sharpe, Sortino, slippage, komisyon, net nakit akışı asla atlanmaz. Kümülatif/marjinal hesaplarda matematiksel dizilim eksiksiz.
12. Sistemde hiçbir zaman input.bool aç/kapa toggle yok; kurallar sabit açık; tek dosya.
13. OHLC'yi sormadan önce tüm erişim yerlerinde ara (sohbet yüklemeleri, Drive, GitHub `data/`).
14. Kötü sonuç → bırak. Doğrulanmış bazı geçemeyen konfigürasyon ne kadar karmaşık olursa olsun reddedilir.
15. Ters kontrol önce: OOS, MaxDD, worst-case → sonra iyi metrikler.

## 4. ÇALIŞMA TESLİMATI (her çalışmada)
- Ne değişti · baz vs yeni (vs önceki) tablo · net / PF / risk metrikleri / işlem sayısı / maliyet
- Zaman bloğu dağılımı + OOS · TV-doğrulanmış mı yoksa research-engine mi
- Tam Pine kodu (snippet değil) + REV diff · faydalıysa PNG equity/DD grafikleri
- Sharpe, Sortino, Calmar, CV grafikle gösterilsin.

## 5. PINE STANDARTLARI
**Şablon:** `overlay=true, process_orders_on_close=true, pyramiding=0, initial_capital=500, default_qty_type=strategy.cash, default_qty_value=500, commission_value=0.04, margin_long/short=0, calc_on_order_fills=false, calc_on_every_tick=false, max_bars_back=2000–5000`. 500/0.04 çoğu tekil çalışma için referans; esas olan son Pine kodunun ayarları.

**Repaint (kanonik):** ifade `request.security` dışında hesaplanır, `lookahead_off`, `[1]` dışarıdan, sinyal değişkenlerinde `barstate.isconfirmed`; bypass koşullarında `ta.crossunder` içinde `[1]` yok. Fonksiyon-içi varyant: `lookahead_on` + tüm seri `[1]` ofsetli (son tamamlanmış HTF mum). Gerçeklik referansı = TV Bar Replay ("replay her zaman gerçeği söyler").

**Kod stili:** tam dosya; değişen satır `// [REV-XX]`; NAV etiketleri `// §§REV07_...§§` + dosya başı NAV listesi; REV LOG en üstte; header'da sürüm amacı + son TV export sonucu + "Strategy Tester'da doğrula"; input'lar numaralı grpXxx gruplu. REV-14B tarzı (tek değişken, non-invasive) kanonik revizyon.

**ADX:** son söz `ta.dmi` uygun şekilde kullanılabilir (eski not: manuel TR/DM/DI + fixnan).

**Akış:** durum/rejim bazlı (event/flip değil): `entry = position_size==0 and stBull and adxPass`, `exit = position_size>0 and stBear`.

**WunderTrading:**
- Comment'ler her emir çağrısında `comment=` içine literal yazılır (const/değişken yok) — WT yalnız böyle okur. Dokunulmaz.
- "Wunder soketi" = tam metin: `ENTER-LONG_` / `ENTER-SHORT_` / `EXIT-ALL_` + `BINANCE_<SEMBOL>_<BOT-ADI>_<TF>_<bot kodu>`. Yeni sürümde soketler kullanıcının dosyasındakiyle birebir korunur.
- İlk EXIT-ALL tüm pozisyonu (tüm katmanları) kapatır → çok katmanlı backtest buna göre.
- Çift exit / mükerrer entry / state uyumsuzluğu yok. Manuel müdahale çözümü yok (pause/cooldown/reaktivasyon otomatik).
- WT instant terminal ayarları (ör. +%6.5 sonra %0.15 geri çekilmede kapat, SL %15) mum içi fiyatla WT tarafında çalışır, Pine değişikliği değildir; 1m OHLC üzerinde mum içi sıra tahminiyle, ek komisyonsuz simüle edilir.
- Aynı barda EXIT+ENTER yok (handoff: bu bar EXIT, sonraki bar ENTER).

**Diğer:** DCA karşılaştırmaları sermaye-nötr (100 birim). Saat/gün filtreleri sisteme özel, genellenmez. Overlay/instant exit: mantıksal ve gerçek poz state'i ayrı; erken çıkış yeni girişi serbest bırakmaz, orijinal çıkışta ikinci exit üretmez. Açık kaynak TV indikatörleri "aşçının malzemesi". Yeni açık talimat eski çelişeni ezer.

## 6. MİMARİ REFERANS ÖRNEKLERİ
**LTC 3M TRINITY LINEAR MASTER V10.10-H13S** ("iyi tasarlanmış sistem örneğimiz"):
- A (7 kural) / B (6 kural) / C (4 ağaçlı GBM, 25 normalize feature) sanal motorlar yalnız oy üretir; master tek emir. vote = 3A+2B+2C; ≥2 long, ≤−2 short, aksi flat; motor içi ters sinyal = giriş yok.
- Shadow master: filtresiz akış ayrı izlenir; filtre yalnız giriş pulse'ında kabul/ret; reddedilen işleme ortadan girilmez.
- H11 dust path-lock: bloklanan giriş 0.01 qty, alert kapalı, PnL ×(500/qty) sanal ölçek; state makineleri baz yolda.
- Reservoir/duty pacing: yön bazlı rolling realized net (14g cap 100/75, 60g cap 200) aşılınca her N. işlem; nominal küçültülmez.
- Sabit +500/0/−500; net-USDT TP/SL ve Daily BE stop fiyatı komisyon dahil cebirsel.
- Sidecar: core flatken tek-bar limit, öncelik/fallback zinciri.
- Filtre kabul kuralı: her blok 2020–22 / 2023–24 / 2025–26 parçalarının üçünde de negatif olmalı.

**XMR 6M REV-18 (250+250 DCA):** shadow engine START/EXIT otoritesi; ghost = gerçek emir yok, saf matematik; REFERENCE engine önceki sürümün kabul yolunu ayrı ref equity ile dondurur; linear capital router (kararlar session başında kilit; suppress > boost/promote > legacy DCA); HWM/underwater governor shadow PnL'den (HWM'de son 3 shadow ≤ +7 → ghost; underwater'da son 6 ≥ 0 → blok aç; PF <1.8 kapa / >1.9 aç).

**DOGE V42 kalıbı:** owner state machine (0 flat / 1 CORE / 2 yan sistemler), öncelik CORE > BRIDGE > recovery > hunter > trend > ten; canXEnter bayrakları; `strategy.exit` stop/limit, zaman çıkışı `strategy.close`; ghost/shadow health exit; closedtrades döngüsüyle kapı (lookback 64, eşik −%11), rolling PF proxy (40 trade); tüm eşikler ölçekten bağımsız (% / ATR% / RSI / ER).

**Portföy tasarım ilkesi:** XMR long düşüşte kendi pozisyonunu azaltır (% seviye partial SL); koruyucu short'lar crossover değil durum şartıyla girer (ör. EMA10<EMA20) ki hareketi asla kaçırmasın — kâr ikinci planda. "Roket" sistemler ~%5 ağırlıkla sınırlı.

## 7. METRİK ÇERÇEVESİ
- Sharpe = verimlilik; R² = istikrar (zamana dayalı). **Fractal Linearity** = saatlik+günlük+haftalık R² ortalaması (0.75 altı portföye girmez, 0.80+ sağlam; aylık R² yapay yüksek).
- **Sermaye Ataleti** = (MaxDD × recovery süresi) / getiri (düşük iyi).
- CV (günlük/haftalık), pozitif gün/hafta/ay oranı, Ulcer, underwater episode (sayı, ort/medyan/P95/max), P95 intrabar DD.
- Portföy skor ağırlıkları (eski): Consistency 25, Daily WR 20, Low intrabar DD 20, Sharpe 20, Return 15.
- Felsefe: "NET VADELİ MEVDUAT FONU" — intrabar dahil düzgün TIK-TIK-TIK diyagonal equity. Optimizasyon kalitesi = saat/gün/hafta/ay dağılım eşitliği; sweet-spot kovalamak overfit.
- MaxDD = running equity peak-to-trough.
- Audit çerçevesi: net, ROI, PF, MaxDD (USDT + /nominal %), Sharpe/Sortino saatlik 8760, Calmar, Ulcer, Equity R; dönem tablosu (7g/90g/12ay/YTD); günlük dağılım (CV, P95, medyan, MAD/medyan); underwater; haftalık/aylık dağılım; tam ay floor; sistem katkısı.

## 8. SAĞLAMLIK TESTİ STANDART SETİ (prensip başlıkları)
- MC bootstrap (yerine koymalı) → net P1/P5, DD P95/P99, zararla biten senaryo oranı
- MC shuffle → sıra/yol riski
- Monkey rastgele yön → sistem rastgelelerin en iyisini geçmeli
- Monkey rastgele zaman → aynı
- 6 aylık zaman blokları → her blok pozitif mi, dağılım düzgün mü (eksik son blok işaretli)
- Walk-forward: IS %40 / OOS %60 (proje bazında teyit); OOS'a tekrar tekrar bakma. 60 günlük pencere 2,5 yıllık pozitif kanıtı ezmez.
- Referans: V8.1 tüm seti geçti.

## 9. OPTİMİZASYON YOLU
- İstenen: TV backtester'ın aynısı offline (kitaplıklaşmış motor, PyneCore) + Pine + kendi 1m OHLC → otomatik çalıştır, sync-check, optimize. "Bu kadar basit."
- Tarama sırası interleaved (ilk→son→ikinci→sondan ikinci…); min 50 kombinasyon; seç + karar ver.
- Platform kısıtları (WT DCA min oran, deviation multiplier cap 10, max DCA orders) gridin başından dahil.
- Giriş filtreleri trade-substitution yüzünden toplanmaz; çıkış tarafı değişiklikleri daha güvenli.
- Komisyon %0.04 TV export'ta zaten dahil; ikinci kez düşülmez (overlay çıkışlarında ek %0.04 = muhafazakâr slippage payı).
- Kendi Python motorları: TV_FIXED_TESTER_ENGINE 2.0.0, V2 NORMALIZED ROBUST; eski SİM ENGINE v4 phantom=0 (exit önceliği: Trail → BE → Hard SL → Ext SL).
- TECO: mega kâr kümelerini ayır, FULL/CORE/TAIL katmanlarıyla optimize et.

## 10. VERİ
- 1m OHLC: GitHub `altug1986/PINE_HELLO_WORLD/data/` → `https://raw.githubusercontent.com/altug1986/PINE_HELLO_WORLD/main/data/<dosya>` (raw okuma auth'suz).
- Dosyalar (…_1m_FULL_THROUGH_20261116.csv.gz): XMR (2020-02-03→), ETC (2020-01-16→), ADA (2020-01-31→), DOGE (2020-07-10→), BTCDOM (2021-06-21→), LTC (2020-01-09→); BTCUSDT_PERP_1m_2019_20261115.csv.gz. Kolonlar: time,datetime_utc,datetime_tr,open,high,low,close,volume (BTC: time ms UTC,o,h,l,c,v).
- Yeni parite/güncelleme → `data/`; >100MB bölünür.
- Drive'da: 1dk OHLC tam setleri, PAXG spot, TV export (.xlsx) + .pine sürüm dosyaları, portföy audit PNG serisi.
- BAT: 1m OHLC 2020-02→2026-11 + OI (Binance metrics 5m/10m, 2021-12→).

## 11. CANLI PORTFÖY — "CYBORG 4000" (2026-11-20)
16 bot, 4.000 USDT sabit nominal (adında PINE geçen botlar nominal ×2 sayılır). Baz → aday:
| Sistem | Baz | Aday |
|---|---|---|
| XMR Path-Lock 6dk Ana | 375 | 375 |
| XMR Path-Lock Partial | 95 | 95 |
| XMR Micro 8dk Ana | 400 | 300 |
| XMR Micro Partial | 160 | 160 |
| XMR MANU DCA 6dk Ana | 325 | 225 |
| XMR MANU Partial SL%3,2 | 325 | 225 |
| XMR Short 5dk | 200 | 200 |
| XMR Short 4dk (Bear TOTAL2) | 340 | 400 |
| DOGE Short 5dk | 240 | 250 |
| DOGE Long 10dk | 150 | 150 |
| DOGE 3dk çift yön | 140 | 200 |
| BTCDOM Long | 480 | 500 |
| LTC Trinity | 350 | 350 |
| LTC Polaris | 140 | 200 |
| ETC Smart Swing | 140 | 200 |
| ADA Swing Trend | 140 | 170 |

Regülatör referansı (aday): 208 hafta +21.489,83 USDT, 201/208 pozitif hafta, haftalık medyan %2,025, CV 0,817, 47/47 pozitif ay (baz +20.805,57).

**Tarihçe:** v4/v7 ($5.000, 10 sistem: XMR, ORDI, HBAR, ADA, ETC, BABA; Sharpe ~6.5–6.9, 27/27 pozitif ay) → 6.250 (9 sistem, 2026-10) → 14.000 "SON AYARLAR" (~2 ay kârlı, hesap 250→770; sonra hedge'ler senkron çalışmadı, kâr 500→150; düşük backtest DD'sini "tesadüf" sayıyor) → 3.450/3.600 → CYBORG 4000.
**Kasım 2026 değişiklikleri:** pump'ta ezilen short'lar kesildi; short rejimini kaçırmayan 2 özel short eklendi; XMR long'lara partial SL; roket sistemler ~%5'le sınırlı. Kendi değerlendirmesi: "olgunlaşma var ama tam oturmadı" — kaizen ile evriliyor.
**Canlı Binance (2026-11-17):** 1 yıl net −1.183,50 (hacim 2,47M); son 3 ay +188,29; 4–15 Kasım 12 gün üst üste kayıp −576. 3 ay sembol: LTC +535,55 · XMR +262,17 · ADA +92,76 · ETC +8,59 · BTCDOM −50,34 · DOGE −129,57. Canlı hesap sermayesi ~380 USDT.

**YOL HARİTASI (2026-11-17):** (1) DONDUR 60–90 gün, başarı ölçütünü önceden yaz · (2) MUTABAKAT Binance ↔ TV export, bot bazında fark (maliyet/execution/repaint) · (3) MALİYET ~4,8 bps/hacim tahmini doğrula, brüt edge ~10 bps altı sistemi küçült/kapat, maker/BNB indirimi · (4) RİSK açık poz dahil MTM baz vs Instant · (5) Kasım olay analizi · (6) Short "asla kaçırmaz" iddiasını olay bazında sına · (7) Portföy freni kuralı · (8) Yoğunlaşma (XMR payı %63) gerekçesi · (9) Asıl kanıt bir sonraki gerçek düşüş, o zamana kadar nominal küçük.

## 12. SİSTEM BAZLI NOTLAR
- **XMR:** en güçlü sistem. Edge derin adverse excursion'ı emmeye dayanır → işlem başı %1 SL felaket (net 3.778 → −1.130). Binary MAE yapısı (MAE <~%4 ≈ neredeyse hep kazanç). MFE'nin ~%89'unu yakalıyor. Risk portföy seviyesinde.
- **ETC 3M:** V16.3 (R1 DD20 + E3 hard %5 + SC_LONG off) kabul edilen baz/canlı. Hedef net >%2100 (500$), çeyrekler medyana yakın. V16.4 (OI flush long) reddedildi; RSI-gecikme, PF-ters çevirme, sniper-short reddedildi. Sonraki: OI ile ulcer/underwater bölgelerini kârlı yapmak.
- **BTCDOM 3M V2.8M:** yalnız long, short yok.
- **LTC 3m SYNC FIXED-500 TARGET3300:** martingale yok; 3×166,67 bucket; sabit 13x cross exposure; 10x sanal referans motor (sync ağırlıklı ort. fiyat) sinyal/SL/aylık kilit için; SL %6; açık sorun Max Underwater Days ≤10.
- **PEPE 5M SWING V11.7** (1000PEPEUSDT.P): kalite referansı — TV 1751 işlem, net 9.767$, PF 1.69. Long+short, flat yok, trend-takip canavarı.
- **BAT 5M SWING (CYBORG BAT):** sıfırdan, PEPE V11.7 ilhamlı; önce "her % hareketi takip et, hiç kaçırma" (≤4000 işlem), sonra adım adım (PF filtresi); 14 bileşen sistem konsorsiyumu (olasılıksal giriş/çıkış); 6000$ üstü en tutarlı sürümle devam (2026-11-21).
- **ORDI:** DCA çoğunlukla zararlı; MAIN için −0,70% trigger optimum (PF 2.026→2.111).
- **HBAR:** dip-capture girişi BE mekanizmasını bozar; BE pair-specific.
- Overlay (mental) SL borsada görünmez → stop-hunt'tan korur.

## 13. DİĞER ALANLAR
- Gayrimenkul/vergi/stopaj/matrah: adım sırasına tam sadık, stopaj doğru basamaktan, matrah hatalı küçültülmez. Finansal senaryolar bütüncül (kira, fırsat maliyeti, yaşam gideri + trading sermayesi).
- Pazarlama: soğuk ulaşım değil "pusu/inbound" — nitelikli altyapı, gelen talebi karşıla, ürün niteliği öne.
- Mimari/tasarım: estetik + teknik detay + nitelikli malzeme; klasik, zamansız, mekanik hassasiyet.

---
# EK: KULLANICI SKILL DOSYALARI (Claude)
Diğer skill'ler (Claude hesabında): cyborg-system-playbook, teco-core-optimization, tv-optimizer-mc-oos, kaizen.


## SKILL: cyborg-portfolio-regulator

---
name: "cyborg-portfolio-regulator"
description: "Re-weight the CYBORG multi-bot crypto portfolio: per-system unit PnL, baz vs aday nominal, weekly/monthly maps, DD percentiles, Sharpe/Sortino/Calmar, constrained weight optimization."
---

# CYBORG Portföy Regülatörü (v2 — keskinleştirilmiş)

Canlı referans: **CYBORG 4000** — 16 bot, 4.000 USDT sabit nominal, bileşik yok.
Amaç: toplam kâr değil; negatif hafta↓, kârın haftalara/aylara eşit yayılması, düşük kuyruk DD (P95–P99), sistemlerin birbirini gerçekten dengelemesi. Baz vs aday her zaman yan yana.

## 0. Demir kurallar
1. **İki katman ayrı.** Katman A = işlem simülasyonu (TV export / `tv-offline-backtester`) → sistem başına K/Z serisi. Katman B = bu skill: doğrulanmış serileri nominalle birleştirir. B asla emir/dolum/SL üretmez.
2. **Sentetik seri yok.** Haftalık farkı /7 ile günlere yaymak YASAK. Günlük/aylık metrik yalnızca gerçek günlük (veya daha ince) sistem serisinden hesaplanır. Yalnız haftalık veri varsa: aylık = "TAHMİNİ", DD/Sharpe = "haftalık çözünürlük, hafta içi DD görünmez" etiketi.
3. **Her portföyün % paydası kendi nominal toplamı.** Baz gösterilirken aday toplamı kullanılmaz. Toplam ≠ hedef (4.000) ise ekranda kırmızı uyarı.
4. **Etiket zorunlu:** her sayı `TV-export` / `offline-motor` / `TAHMİNİ`. Doğrulanmamış sayı sonuç gibi sunulmaz.
5. **Regresyon önce.** Yeni özellik eklemeden önce referans değerler birebir üretilir (bkz. §6). Tutmazsa önce fark çözülür.
6. Değişiklik disiplini: baz dondurulur, tek anlamlı değişiklik, baz ve önceki sürümle kıyas, kazanç yoksa geri al.

## 1. Veri sözleşmesi (JSON v2)
```
meta:   {tz:"Europe/Istanbul", source:[16 export dosyası + tarih], commission:"dahil, tekrar düşülmez", funding:"hariç", res:"1d"|"10m"}
names:  [16]
nom_bt: [16]   # backtest'te kullanılan efektif nominal (unit normalizasyonu için)
base:   [16]   # önceki/baz nominal (USDT)
cand:   [16]   # aday nominal
dates:  [D]    # TR günleri, ortak pencere
unit_d: [16][D]   # kapanmış net K/Z / nom_bt  (1 USDT nominal başına, günlük)
unit_mtm_min: [16][D]  # opsiyonel: gün içi en kötü açık K/Z (10 dk MTM) / nom_bt → intrabar DD
first_valid: [16] # sistemin gerçek veri başlangıcı
```
- Kaynak: her TV export `İşlemler` sayfası → çıkış satırları → net K/Z (komisyon dahil) çıkış zamanında → TR saatine çevir → günlük topla → `nom_bt`'ye böl. DCA/partial bacaklar aynı güne toplanır.
- Efektif nominal: Wunder'da adında PINE geçen botlar ×2 sayılır; tablo bu tanımla tutarlı olmalı.
- Ortak pencere: geç başlayan sistemin boşluğu **sessizce 0 doldurulmaz** (volatiliteyi düşük gösterir). Ortak pencere raporlanır; kısa geçmişli sistem işaretlenir.
- Hafta = ISO Pzt–Paz (TR); yalnız tam hafta/ay; kısmi dönem ayrı işaretli.

## 2. Çekirdek hesap (sabit nominal, doğrusal)
```
pnl[t]   = Σ_i unit_d[i][t] × nom[i]
r[t]     = pnl[t] / Σ nom            # kendi toplamı
eq[t]    = Σ_{≤t} pnl                # kümülatif K/Z
DD[t]    = max(eq[≤t]) − eq[t]       # USDT; MTM varsa eq_low[t]=eq[t-1]+Σ unit_mtm_min×nom
DD%nom   = DD / Σ nom × 100
DD%eq    = DD / (C + max(eq[≤t])) × 100   # C = hesap sermayesi verilirse (running-equity tabanı)
```
Haftalık/aylık = günlük toplamı (asla /7 yayma).

## 3. Metrikler (baz | aday | Δ — hepsi)
- **Kâr:** net, ROI/nominal, yıllık basit, PF (dönem bazlı ise etiketle).
- **Dağılım:** pozitif gün/hafta/ay oranı, haftalık ve aylık medyan, CV (n−1, std/|ort|) günlük + haftalık, MAD/medyan, hedef üstü hafta sayısı (%1,5/%2/%2,5).
- **Kümelenme:** en iyi %10 haftanın net içindeki payı; 4×52 hafta blok kâr payları (ideal %25'er); sistem katkı payları; XMR payı, long/short nominal oranı.
- **Risk (öncelik kuyrukta):** underwater gün dağılımı P70/P80/P90/P95/P98/P99/P100(=MaxDD); 30 günlük kayan pencere maxDD dağılımı (aynı persentiller); underwater epizot sayısı, medyan/P95/max süre (gün), MaxDD peak/trough/recovery tarihi. **Teminat önerisi = P95 DD × 2** (USDT).
- **Oranlar:** Sharpe = ort(r)/std(r)×√365; Sortino = ort(r)/√(ort(min(r,0)²))×√365; Calmar = yıllık net / MaxDD; Ulcer = √(ort(DD%nom²)); Equity R² (lineerlik).
- **Dengeleme:** sistemler arası günlük korelasyon matrisi; portföyün en kötü %10 gününde koşullu korelasyon; ortak zarar haftaları (aynı hafta negatif sistem sayısı histogramı); hedge testi: XMR long toplamının negatif olduğu haftalarda XMR short katkısı (USDT ve kapsama %).

## 4. Ağırlık optimizasyonu
- Kısıtlar: Σ nom = hedef (4.000); nominaller 5'in katı, arama adımı varsayılan 25 (kullanıcı söylerse o); **hiçbir sistem çıkarılmaz** (alt sınır > 0, varsayılan baz×0,5 … baz×1,5); kullanıcı "roket" sistemleri belirtirse tavan ~%5; XMR payı ve long/short oranı her adayda raporlanır.
- Skor: `tv-optimizer-mc-oos` BEST-OF-BEST (yıllık getiri, R², P98 DD, pozitif ay oranı, aylık volatilite cezası) — haftalık CV ve negatif hafta sayısı ek bileşen.
- Arama: koordinat inişi + simulated annealing; her aday için komşu (±1 adım) medyan skoru = plato skoru, tepe değil.
- Doğrulama: walk-forward (IS 24 ay → OOS 6 ay, kayan) — ağırlık IS'te bulunur, OOS'ta uygulanır; Monte Carlo blok bootstrap (günlük, blok 5–10 gün, ≥2.000 tur) → DD P95/P99 dağılımı.
- **Kabul:** skor↑ VE OOS bazdan kötü değil VE MC P99 DD bazdan kötü değil VE plato skoru ≥ baz. Biri tutmazsa aday reddedilir; ne kaybedildi/ne kazanıldı tablosu her durumda verilir.

## 5. Çıktı
1. **Tek HTML artifact** (çevrimdışı, gömülü JSON v2, vanilla JS): 16 sayı girişi + slider, anında yeniden hesap, baz/aday/serbest seçim, "başlangıca dön"; baz ve aday **yan yana** 4×52 haftalık harita (hücre tık → tarih aralığı + 16 sistem katkısı zarar→kâr sıralı); aylık USDT/% tablosu; equity + underwater; DD persentil çubukları; Sharpe/Sortino/Calmar/CV/Ulcer kartları; korelasyon ısı haritası; nominal toplam uyarısı. Durum yalnız bellekte; kalıcı ayar istenirse URL hash.
2. Analiz için Python motoru (aynı formüller) + PNG grafikler; HTML ve Python sonuçları regresyonda birbirini tutmalı.
3. Rapor: değişen ne, baz vs aday tablo (Δ ile), DD P95/P98/P99, Sharpe/Sortino/Calmar/CV, blok dağılımı + OOS, kaynak etiketi, kayıp ✗ / kazanç ✓.

## 6. Regresyon referansları (CYBORG 4000 v1 HTML, haftalık matris)
- Aday: 208 hafta toplam **+21.489,83**, pozitif **201/208**, haftalık medyan **≈%2,025**, CV **≈0,817**, pozitif tam ay **47/47** (v1'de aylık /7 ile tahmini — v2'de günlük veriyle yeniden hesaplanınca değişebilir, fark raporlanır).
- Baz: toplam **+20.805,57**.
- Nominal tablosu (baz→aday, toplam 4.000→4.000): XMR Path-Lock 6dk Ana 375→375 · Partial 95→95 · XMR Micro 8dk Ana 400→300 · Partial 160→160 · XMR MANU DCA 6dk Ana 325→225 · Partial SL%3,2 325→225 · XMR Short 5dk 200→200 · XMR Short 4dk 340→400 · DOGE Short 5dk 240→250 · DOGE Long 10dk 150→150 · DOGE 3dk çift yön 140→200 · BTCDOM Long 480→500 · LTC Trinity 350→350 · LTC Polaris 140→200 · ETC Smart Swing 140→200 · ADA Swing Trend 140→170.
- Tolerans: USDT ±0,01; oran ±0,001. Haftalık değerler v1 JSON'dan birebir; günlük v2 serisinin haftalık toplamı v1 `unit×nom` ile tutmalı (tutmazsa kaynak farkı raporlanır, devam edilmez).

## 7. Sınırlar (raporda açıkça yazılır)
- Nominalle doğrusal ölçek varsayımı: min emir, slippage, marjin çakışması doğrusal değil — büyük ağırlık değişiminde sapma.
- Funding hariç; canlı maliyet backtest'ten yüksekse (Binance mutabakatı) sonuç iyimser.
- Pozisyon çakışması / marjin kullanımı yeniden hesaplanmaz (MTM verisi yoksa intrabar DD eksik).
- Instant SL/TP/trailing değişikliği Katman A işidir; burada simüle edilmez.

## SKILL: tv-offline-backtester

---
name: "tv-offline-backtester"
description: "Run a TradingView Pine strategy offline on 1m OHLC with a TV-faithful engine (PyneCore), sync-check vs TV export, then auto-optimize. Use when the user gives Pine + OHLC/TV export and says test/optimize/backtest."
---

# TV-OFFLINE BACKTESTER

Goal: Pine + OHLC in → TV-equivalent backtest → sync check vs TV export → automatic optimization → decision. Do NOT ask the user questions that the files answer. Do NOT stop for clarification on standard steps. Turkish, terse, numbers first.

## 0-HIZ. Speed rules (user complaint: "çok yavaşsın")
- FIRST action: check uploads for a TV export xlsx. If present → report from it within 1–2 tool calls, THEN (only if needed) build the engine.
- If no export and the task is "backtestte sonucunu bul": say in one line that the TV export gives the result instantly, and start the engine in parallel — don't wait.
- Don't read a 2–3k-line Pine fully before starting data prep: launch the 1m→3m/HTF feature build in background FIRST, read Pine while it runs.
- Pure-Python arch. B port of a 2.4k-line V65B-style script runs 1.1M 3m bars in ~20 s; PyneCore not needed for this family. Use numba for rma/ema features.

## 0a. TV export first (fastest result)
- If a TV export xlsx is attached, it IS the backtest result: report from it immediately (no engine needed): net, PF, win%, long/short net, Sharpe/Sortino, DD P95/P98/P99 (daily underwater), maxDD, UI, yearly net, last 30/60/90/180/365-day net + positive weeks, weekly heatmap PNG (ISO week x year, cell = weekly net, row label = year total + positive weeks).
- TV export timestamps = bar OPEN time of the fill bar. Trades sheet `İşlemler`: col0 trade#, col1 type (`giriş`/`çıkış`, `Kısa`=short), col2 time, col4 price, col7 net PnL. Open trade rows have a string in col2.
- Build the offline engine only for what the export cannot answer: variants, optimization, ablation, rule tests.

## 0. Inputs (find them yourself)
- Pine code: attached .txt/.pine.
- OHLC: attached 1m csv/csv.gz/parquet (user archive: `{SYM}_1m.parquet` or `{SYM}USDT_1m_FULL_THROUGH_*.csv.gz`, cols time(ms UTC),open,high,low,close,volume[,ffill]). If not attached: search Drive by title; files >5 MB cannot come through the Drive connector — ask ONCE: "dosyayı sohbete sürükle". Shell network is blocked (Binance 403); never retry it.
- TV export xlsx (Turkish TV): sheets `İşlemler`, `Performans`, `Özellikler`. `Özellikler` gives backtest window start, slippage, commission, magnifier, inputs.

## 1. Engine
`pip install --break-system-packages pynesys-pynecore pandas numpy openpyxl pyarrow`
- PyneCore = TV-validated Pine runtime (ta.*, request.security, strategy broker). PyneComp (Pine→Pyne compiler) is cloud-only → port by hand.
- Two valid architectures:
  A) Generic: port Pine to a Pyne strategy script (`@script.strategy(...)`) and run with `ScriptRunner`, broker does fills.
  B) Hybrid (for huge state-machine strategies, faster): Pyne *indicator* script returns all ta.* features per bar (exact TV math) → pure-Python port of the strategy logic + own TV-semantics broker over numpy arrays. Reference build: ETC 3M V16/V14C (`feat3m.py, feathtf.py, feats.py, engine.py, run.py, compare_tv.py`).
- Resample 1m → chart TF with bar time = open; HTF via own indicator run on 60/240 data; `lookahead_on + [1]` == previous completed HTF bar value.

## 1b. Generic Pine parser (for large scripts)
- `pip install pynescript` parses Pine v5/v6 to an AST (`from pynescript.ast import parse`). Parsing a 4k-line script takes ~2 min → parse once and pickle the AST. Do NOT use `pynescript.ast.walk` (extremely slow); write a recursive walker over `_fields` skipping `ctx`.
- Use the AST to inventory: request.security calls (symbols/TFs → which OHLC files are needed), subscripts (which series need history), strategy.* calls, ta.* functions, var declarations, switch/for/while usage. Answer "which OHLC is needed" from this inventory without reading the whole script.
- Node set seen in real scripts: Assign(mode=Var), ReAssign, AugAssign, Conditional, Switch(subject), Case, ForTo, While, Break, FunctionDef, Tuple, Subscript, Specialize (array<T>), Qualify(const).

## 2. Pine → Pyne porting rules
- File starts with `"""@pyne"""`; logic in `main()`; inputs = main() kwargs with `input.*` defaults.
- Only vars annotated `Series[T]` keep history; any `x[n]` on a user var needs `Series`. `var` → `Persistent[T]`.
- `str.*` → `string.*`; `array.from` → `array.from_items`; arrays are lists; `switch` → if/elif.
- ta.* inside functions keep per-call-site state (same as Pine) — keep ta calls unconditional at top level.
- Strategy decorator: copy every strategy() arg; ALWAYS set `use_bar_magnifier=False` unless TV export says magnifier on (PyneCore default is True!).
- na: comparisons with nan are False like Pine; use `na()`, `nz()`, lib `math.max` (na-propagating).
- Drop tables/labels/plotshapes (no trade effect).

## 3. TV broker semantics (own broker, arch. B)
- `process_orders_on_close=true`: market orders fill at the same bar close ± slippage ticks. Resting stops are checked at the next bar before the script: long stop fill = min(open, stop) − tick; short = max(open, stop) + tick.
- Commission percent on both sides: (entry+exit)*qty*c. qty = floor(cash/fill / mincontract)*mincontract.
- Known gap: if strategy currency=USD and symbol is USDT, TV converts with USDTUSD → qty differs ~0.0–0.1%. Note it, don't chase it.
- Exit orders bound to an entry disappear when that trade closes.

## 4. Sync check (mandatory before any result claim)
1. Run on the SAME window TV loaded (`Özellikler` → backtest start; TV restarts all indicators there, so start the data there, not earlier).
2. Match trades by (entry time, dir); report: matched/total, exit-time matches, Σ|pnl diff|, FIRST mismatch with bar context.
3. Fix first mismatch → rerun. Until synced, label every number "araştırma motoru, TV doğrulanmadı". Never write "tam senkron" before 100% entry+exit match.

## 5. Automatic optimization
- Freeze baseline. Expose the target section's thresholds as params (defaults = current).
- Ablation first (each rule off/on), then one-parameter neighbor grid (e.g. ±1–3 steps). One change at a time.
- Run full history (e.g. 2020→now) AND report time blocks (yearly + 2020-22 / 2023-24 / 2025-26) + last-6-months OOS.
- Robustness: winner must hold in neighbors and in ≥2/3 time blocks; no date filters; prefer %/ATR% measures.
- Accept only if better than baseline on risk AND not worse on net by more than the risk gain justifies; otherwise keep baseline ("değişiklik yok" is a valid result).

## 6. Report (fixed format, Turkish)
- What changed (REV diff), baseline vs new table: net, PF, win%, trades, commission, LONG/SHORT net, Sharpe, Sortino, Calmar, **DD P95/P98/P99 (daily underwater)** first, maxDD as extra, intrabar MAE P95/P99.
- Monthly/yearly distribution, OOS block, CV of monthly returns.
- PNG equity + DD charts when useful.
- Full Pine file with the change (inputs default = winner), sent as a file; state it is NOT TV-verified until the user runs it.

## 6b. Lessons from verified runs
- ETC 3M V16.3: offline engine matched TV 2526/2532 trades over 2020–2026 (net 9954.8 vs TV 9944.3); residual mismatches came from 1m data differences, not logic.
- TV sizes `default_qty_type=cash` with a USD→USDT conversion: qty ±0.1% vs cash/fill. Ignore.
- Run grid variants sequentially in ONE process (each full 1.2M-bar run ≈ 30–70 s); 2 parallel processes on the 565 MB feature pickle got OOM-killed and multiprocessing.Pool deadlocked.
- Keep a frozen copy of the engine before adding knobs; regression-check the default params reproduce the baseline net to the cent.
- User rules: never add on/off (input.bool) toggles to delivered Pine — rules are hard-wired ON; deliver ONE file.

- DOGE 3M V65B DD-REF (2026-11): arch. B port matched TV 2302/2328 entries, net 13122 vs TV 13017; 2021–2026 yearly nets within ~1%; nearly all residual mismatches in 2020 (1m data differences).
- PRICE GRID: TV stores prices as integer ticks × tick. Snap 1m OHLC with `round(p/1e-6)*1e-6` (or the symbol's finest historical tick). Without it, threshold tests like `mfe <= 2.0` flip on float ulps (100*(0.3774/0.37-1)=2.0000000000000018 vs TV 1.99999999999998) and whole trade chains diverge. Do NOT snap to today's mintick (e.g. 1e-5) if older history had finer ticks — that made sync worse.
- 50m (non-divisor) HTF: bucket per UTC day (00:00 anchor); last bar of the day is shorter.
- Broker details that mattered: process_orders_on_close fills market orders at close; newly placed stop/limit already beyond close fills at close; resting exits next bar: gap→open, else OHLC path (open nearer high → O-H-L-C); closedtrades.profit is net of both commissions; openprofit = gross − entry fee; close_all(immediately=true) updates position inside the same bar.

## 7. Speed notes
- 1.2M 3m bars: PyneCore feature pass ≈ 3–5 min; arch. B strategy pass ≈ 30–60 s per variant → cache features to parquet, run grid in a loop or multiprocessing.