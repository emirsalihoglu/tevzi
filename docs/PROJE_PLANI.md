# Adaptif Multimodal Kaynak Zamanlama — Proje Planı

**Proje adı:** Tevzi — https://github.com/emirsalihoglu/tevzi
**İngilizce başlık:** Adaptive Multimodal Resource Scheduling for Resource Efficiency and Job Completion Time
**Belge sürümü:** v1.3 — 8 Kasım 2026
**Durum:** Aşama 2 tamamlandı, Aşama 3'e geçiliyor
**İlgili belgeler:** `docs/KESIF_OZETI.md` (Aşama 1–2 bulguları), `docs/LITERATUR_NOTLARI.md` (ilgili çalışmalar)

**v1.3'teki başlıca değişiklikler:** M2'nin kaynağı ve tasarımı (kanal bazlı, havuz bazlı, 5 dk adım, 2 saat pencere + anlık görüntü), eğitim döneminin 19. günde başlaması, ana görev tanımı, M1 tarihçe özelliklerinde değişiklikler, dışlama kurallarının uygulanmış hali ve dışlama akışı. Gerekçeler Bölüm 11'de.

---

## 1. Projenin özü (tek paragraf)

GPU kümelerinde yapay zekâ işlerini sunuculara dağıtan bir zamanlayıcı için, işin **süresini** ve **gerçek kaynak kullanımını** tahmin eden bir derin öğrenme modeli geliştiriyoruz. Model iki farklı yapıdaki veriyi, farklı encoder'larla işliyor: **statik iş özellikleri (tablo)** ve **küme dinamikleri (zaman serisi)**. Bu iki kaynağın birleştirme ağırlıkları, kümenin o anki durumuna göre **adaptif** olarak ayarlanıyor. Asıl amaç yalnızca iyi tahmin yapmak değil; **tahmin kalitesinin ve adaptifliğin, gerçek zamanlama kararlarına (verimlilik ve tamamlanma süresi) ne kadar katkı sağladığını** dikkatli deneylerle ölçmek.

**Keşiften gelen motivasyon:** Kullanıcılar sistematik olarak fazla kaynak istiyor. Medyan iş, ana görevinde istediği GPU'nun yalnızca %5'ini, istediği belleğin %17'sini kullanıyor; işlerin üçte biri GPU'yu hiç kullanmıyor. Çalışma süreleri 12 saniyeden günlere yayılıyor (medyan ~9 dk). Kümenin genelinde tıkanma yok ama üst seviye GPU havuzlarında (V100, V100M32) işler saatlerce bekliyor. Hem yerleştirmede (kullanım tahmini) hem sıralamada (süre tahmini) kazanılacak alan büyük.

---

## 2. Araştırma soruları

### Ana soru
GPU kümelerinde, statik iş özelliklerini ve küme dinamiklerini (zaman serisi) farklı encoder'larla işleyip birleştirme ağırlıklarını küme durumuna göre ayarlayan adaptif bir multimodal model, zamanlama kararlarında kaynak verimliliği ile iş tamamlanma süresi arasındaki dengeyi iyileştirir mi?

### Alt sorular
| Kod | Soru | Nasıl cevaplanacak |
|---|---|---|
| **AS1 – Modalite katkısı** | Küme dinamiklerini zaman serisi olarak eklemek, yalnızca iş özelliklerine dayalı tahmine göre doğruluğu artırıyor mu? | Yaklaşım 2 ile 3'ün tahmin metrikleri karşılaştırması + ablation; **GPU tipi havuzuna göre ayrıştırılmış** |
| **AS2 – Adaptiflik** | Durum-bağımlı birleştirme, sabit birleştirmeden daha iyi mi? Fark hangi koşullarda ortaya çıkıyor? | Yaklaşım 3 ile 4'ün, yük ve havuz koşullarına göre ayrıştırılmış karşılaştırması + gating ağırlıklarının analizi |
| **AS3 – Tahminden karara** | Tahmin doğruluğundaki iyileşme, zamanlama metriklerine yansıyor mu? Yansımıyorsa neden? | Tüm yaklaşımların simülasyon sonuçları; tahmin hatası ve sıralama kalitesi ↔ JCT/verimlilik ilişkisi |
| **AS4 – Sağlamlık** | Model, zamanlayıcının kendi kararlarıyla oluşan ve eğitimde görülmemiş küme durumlarında tutarlı mı? | Simülasyonda üretilen durumlarda tahmin hatasının trace'teki durumlardaki hatayla karşılaştırılması; eğitim–simülasyon M2 dağılım farkı; gün 59–60 patlamasının ayrık analizi |

### Ön kanıt (Aşama 2 keşfi, eğitim dönemi)
- Küme geneli talep bazlı durum; süreyi, kullanımı ve beklemeyi açıklamıyor (|ρ| ≤ 0,04).
- **Havuz baskısı** (kuyruktaki GPU talebi / havuzdaki boş GPU) beklemeyi açıklıyor (V100 ρ = 0,37, V100M32 0,38, P100 0,16), ama süre artıklarıyla ilişkisi zayıf (V100M32 dışında ~0).
- Dolayısıyla M2'nin süre/kullanım tahminine katkısının **küçük** ve **havuza bağlı** olması bekleniyor. Bu beklenti sonuçlar görülmeden önce kaydedildi (bkz. İlke).

### İlke
Önerilen yöntemin daha iyi olacağını **baştan varsaymıyoruz**. Olumsuz veya karışık sonuçlar da "neden ve hangi koşullarda" analiziyle birlikte raporlanacak. Sonuçları yorumlayacağımız **birincil metrikler, sonuçlar görülmeden önce** belirlendi (Bölüm 7). Karar ölçütleri de mümkün olduğunca **sonuçlar görülmeden önce** yazılıyor (Bölüm 11).

---

## 3. Veri

### Kaynak
- **Alibaba Cluster Trace — cluster-trace-gpu-v2020** (Alibaba PAI; 6.742 GPU, 1.897 makine; 68,9 günlük kayıt).
- Atıf: Weng ve ark., *MLaaS in the Wild*, NSDI 2022.

### Doğrulanmış içerik
Ayrıntılar `docs/KESIF_OZETI.md` içinde.
- **Hiyerarşi:** iş (1.055.501) → görev (1.261.050) → instance denemesi (7.522.002).
- **Talepler** görev seviyesinde ve instance başına: `plan_cpu` (çekirdek %), `plan_mem` (GB), `plan_gpu` (GPU %), istenen GPU tipi.
- **Gerçek kullanım** (sensor): deneme başına özet. **Gün 18'de başlıyor** (öncesinde kapsam %0).
- **Makine ölçümleri** (`machine_metric`): **gün 24'te başlıyor**; yalnız doğrulama referansı olarak kullanılıyor.
- **Zamanlar** göreli saniye. job `start_time` gönderilme anı.
- **Makineler:** 6 tip; GPU havuzları MISC 2.240, P100 1.596, T4 994, V100 832, V100M32 1.080 GPU.
- **Ek etiketler** (`group_tag`, `inst_id` ile bağlanır): kullanıcı, grup, workload.

### Kapsam ve dışlama kuralları (uygulandı: `src/tevzi/data/cleaning.py`)
Kayıtlar silinmiyor; her kural bir bayrak. Bir iş, takıldığı **ilk** kurala yazılıyor.

| # | Kural | Çıkan | Kalan |
|---|---|---|---|
| — | Tüm işler | — | 1.055.501 |
| R1 | Durum Terminated değil | 323.146 | 732.355 |
| R2 | Hiçbir görev GPU istemiyor (yalnız CPU işi) | 17.452 | 714.903 |
| R3 | Eksik talep | 0 | 714.903 |
| R4 | `plan_cpu = 0` | 0 | 714.903 |
| R5 | Talep, istenen tipteki en büyük makineyi aşıyor | 83 | 714.820 |
| R6 | Zaman tutarsızlığı (gönderilmeden önce başlama veya süre ≤ 0) | 7 | 714.813 |
| R7 | Gözlenen kullanım makine kapasitesini aşıyor | 540 | **714.273 (model kümesi)** |

- Kullanım verisi olmayan işler süre eğitiminde kalır, kullanım kaybı maskelenir; simülasyonda **arka plan yükü** olur.
- Kısa işler bilerek **dışlanmıyor** (en kısa iş önce sıralamasının kazancı kısa/uzun karışımından gelir).

### Analiz kümeleri
| Küme | Tanım | İş sayısı | Kullanım |
|---|---|---|---|
| `m2_arrival` | Gönderilme zamanı olan tüm işler | 1.051.838 | M2 geliş kanalları |
| `m2_occupancy` | Talebi ve aralığı geçerli tüm Terminated işler (CPU işleri dahil) | 732.257 | M2 doluluk/kuyruk kanalları, simülasyonda arka plan |
| `model` | R1–R7'yi geçen işler | 714.273 | Tahmin modelleri |

### Veri bölme (zamansal)
Gün sayıları ilk işin gönderildiği andan (t₀ = 494.319 s).

| Bölüm | Günler | İş (purge sonrası) | Gerekçe |
|---|---|---|---|
| Isınma (dışarıda) | 0–19 | 123.831 | Gün 0–7 eksik kayıt; **sensor kaydı gün 18'de başlıyor**. Tarihçe özellikleri ve M2 penceresi için kullanılır |
| Eğitim | 19–45 | 281.064 | Kullanım kapsamı %96,4 |
| Doğrulama | 45–52 | 88.611 | Tam bir hafta |
| Test | 52–69 | 219.435 | Sakin günler ve gün 59–60 patlaması (19.543 iş) |

- **Purging:** eğitim (531 iş) ve doğrulama (801 iş) dönemlerinde dönem bitmeden tamamlanmayan işler çıkarılıyor; bunlar uzun işler (süre medyanı ~10–12 saat). Doğrulamada 1 günden uzun işlerin ~%33'ü purge ediliyor → Aşama 3'te model seçiminde dikkate alınacak.
- **Patlama:** test sonuçları hem tüm dönem için hem patlama hariç raporlanır.

---

## 4. Model tasarımı

### 4.1 Modalite 1 (M1) — Statik iş özellikleri (uygulandı: `src/tevzi/features/m1.py`)
- **Yapı:** Tablo, iş başına bir satır; 46 özellik.
- **Kural:** Yalnızca iş **geldiği anda** bilinen bilgiler. Tarihçe **as-of birleştirme** ile, yalnızca gelişten **kesin önce** bilinen olaylardan (7 birim testiyle kanıtlı: `tests/test_features.py`).
- **Encoder (aday):** MLP; kategorik alanlar için embedding.

| Grup | Özellikler |
|---|---|
| **A. Talepler** (11) | Ana görevin `plan_gpu/cpu/mem`, `inst_num`; iş toplamı GPU/CPU/bellek; istenen GPU tipi; kesirli GPU mu; GPU başına bellek ve CPU |
| **B. İş yapısı** (6) | Görev sayısı; PS var mı; evaluator var mı; PS instance sayısı; toplam instance; ana görev adı |
| **C. Kimlik** (3) | Kullanıcı; workload (+ eksiklik göstergesi) |
| **D. Tarihçe** (22) | Grup ve kullanıcı için **bitmiş** işlerden: sayı, log-süre ortalaması ve std, **son biten işin süresi**, **son iki biten işin ortalaması**; kullanım tarihçesi (sayı, GPU ve bellek oranı ortalaması, **GPU'yu hiç kullanmamış iş payı**); gönderimlerden: grup daha önce görüldü mü, son gönderimden beri geçen süre, son 1 saatteki gönderim sayısı |
| **E. Zaman** (4) | Saat ve haftanın günü (sin/cos) |

- **v1.2'den farklar:** medyan → `last` / `last2` (Tsafrir ve ark., TPDS 2007); **başarısızlık oranı çıkarıldı** (Failed bitişlerinin %84'ü kayıtsız → bilinme anı belirsiz); kullanım tarihçesi ve GPU-sıfır payı eklendi.
- **Kategorik kodlama** Aşama 3'te, yalnız eğitim döneminde fit (Empirical Bayes shrinkage adayı).
- **İlk sinyal:** grup tarihçesi olan eğitim işlerinde `grp_rt_last2` ile log-süre Spearman ρ = 0,932; işlerin ~%28'i grup tarihçesi olmadan geliyor.

### 4.2 Modalite 2 (M2) — Küme dinamikleri (uygulandı: `src/tevzi/features/m2.py`)
- **Yapı:** Çok değişkenli zaman serisi, **5 dk** ızgara (19.858 adım). Her iş için gönderilmeden **önceki son 24 adım (2 saat)** + **gönderilme anının tam durumu** (anlık görüntü) → girdi boyutu (25 × 11).
- **Zaman kuralı:** t anındaki durum yalnız t'den kesin önceki olaylarla (6 birim testi: `tests/test_m2.py`).
- **Kaynak (karar D, "kanal bazlı karma"):**
  - **Dışsal kanallar** (gelişler): gönderilme zamanı olan **tüm** işlerden (Failed dahil). Simülasyonda trace'ten aynen oynatılır.
  - **İçsel kanallar** (doluluk, kuyruk, kullanım): **Terminated** işlerden (aralıkları %100 biliniyor). Simülasyonda simüle edilen işler + arka plan yükü ile aynı küme.
  - İş durumu (Failed vb.) kanal olarak kullanılmıyor (gelişte bilinmez).
- **Kanallar (11):**

| Grup | Kanal | Tanım |
|---|---|---|
| Küme (7) | `arr_count` | Son 5 dk'daki geliş sayısı (log1p) |
| | `arr_gpu` | Son 5 dk'da gelen GPU talebi / toplam GPU |
| | `occ_gpu`, `occ_cpu`, `occ_mem` | Çalışan işlerin talebi / kapasite |
| | `pressure` | Kuyruktaki GPU talebi / boş GPU |
| | `use_gpu` | Çalışan işlerin gerçek GPU kullanımı / toplam GPU |
| İşin kendi havuzu (4) | `pool_occ`, `pool_pressure`, `pool_arr_gpu`, `pool_use_gpu` | Aynıları, işin istediği GPU tipinin havuzu için (MISC → tüm havuzlar) |

- **Havuz tanımı:** doluluk, işin **gerçekte çalıştığı** GPU makine tipine (instance'larının en sık tipi; CPU makineleri yok sayılır); kuyruk ve gelişler, işin **istediği** tipe göre.
- **Standartlaştırma** yalnız eğitim döneminden.
- **Doğrulama:** anlık görüntüdeki havuz baskısı bekleme süresini açıklıyor (V100 ρ = 0,37, V100M32 0,38); kullanım kanalı makine ölçümüyle örtüşüyor (gün ≥ 24, ρ = 0,63).
- **Bilinen tavizler:** (i) Failed işlerin yükü içsel kanallarda yok → duyarlılık analizi: doluluk şekli korunuyor (ρ = 0,93), baskı seviyesi değişiyor (ρ = 0,86); (ii) arka plan işleri talep kadar yer tutuyor → Aşama 4/5'te duyarlılık koşusu; (iii) M2 yalnız gönderilme anına kadar; işin çalışması boyunca değişen yük görülmüyor.
- **Ablation:** N ∈ {0 (yalnız anlık görüntü), 12, 24, 72}; adım ∈ {1, 5, 10 dk}; kanal kümeleri (yalnız küme / yalnız havuz / tüm havuzlar).
- **Encoder (aday):** LSTM, TCN veya küçük bir Transformer. Aşama 3'te karşılaştırılacak.

### 4.3 Birleştirme (fusion)
- **Sabit birleştirme (Yaklaşım 3):** M1 ve M2 temsilleri uç uca → ortak katmanlar.
- **Adaptif birleştirme (Yaklaşım 4):** M2'den çıkan durum temsili, gating veya attention ile modalite ağırlıklarını belirler. Hedef başına ayrı ağırlıklar alternatifi.
- **Yorumlanabilirlik:** Gating ağırlıkları kaydedilir; küme yoğunluğuna ve **havuza** göre analiz edilir (AS2).

### 4.4 Tahmin hedefleri (multi-task) (uygulandı: `src/tevzi/data/targets.py`)
| Hedef | Tanım | Dağılım (model kümesi) |
|---|---|---|
| **Çalışma süresi** | İş bitişi − ilk instance başlangıcı; `log1p` | Medyan 534 s; %99 65.199 s |
| **GPU kullanım oranı** | Ana görevin tüm instance ve denemeleri arasında en yüksek `gpu_wrk_util` / `plan_gpu` | Medyan 0,05; **%34 tam sıfır**; %3,5 > 1 |
| **Tepe bellek oranı** | Aynı şekilde `max_mem` / `plan_mem` | Medyan 0,17; %99 3,08; **%10,6 > 1** |

- **Ana görev:** GPU isteyen görevler arasında **toplam GPU talebi** (`plan_gpu × inst_num`) en yüksek olan; eşitlikte yüksek `plan_gpu` → **rol önceliği** (worker > chief > evaluator > ps) → görev adı. (Rol önceliği olmadan, eşit talepli 8.360 işte hedef evaluator/PS görevine bağlanıyordu.)
- Kullanım verisi: işlerin %80,5'i; kullanımı olmayanlarda kayıp maskelenir. İşlerin %78,4'ünde tüm denemeler ölçülmüş.
- **Bekleme süresi** hedef değil; simülatör doğrulaması için referans (medyan 7 s, %90 110 s).
- **Aşama 3'te karar verilecek:** GPU oranı için iki parçalı (hurdle) yapı; bellek oranının kuyruğu (dönüşüm/kırpma); kayıp ağırlıkları; asimetrik (az tahmini cezalandıran) kayıp.

---

## 5. Zamanlayıcı ve simülatör

### 5.1 Zamanlayıcı (tüm yaklaşımlar için aynı)
Model **tahmin eder**, kararı basit ve şeffaf bir kural verir. Yaklaşımlar arasında **yalnızca tahmin kaynağı değişir.**
- **Birim:** İş (gang scheduling).
- **Sıralama:** Tahmini çalışma süresine göre en kısa önce + yaşlanma (aging).
- **Yerleştirme:** Instance başına. Ana görevin instance'ları tahmini kullanım (oran × talep) + güvenlik payına göre best-fit. GPU bölünebilir (kapasite = 100 × GPU adedi). Diğer görevler talepleriyle. İstenen GPU tipi kısıtı (MISC = herhangi bir tip).
- **Güvenlik payı (α):** %0, %10, %20, %30, %50 → verimlilik–JCT **denge eğrisi**.

### 5.2 Simülatör
- **Tür:** Olay tabanlı (iş gelişi, iş bitişi, zamanlama kararı).
- **Girdi:** Test dönemindeki, kullanım verisi olan model kümesi işleri (gerçek gönderilme zamanları, talepler, süreler, kullanımlar).
- **Arka plan yükü:** Kullanım verisi olmayan Terminated işler (CPU işleri dahil) trace'teki gerçek zamanlarıyla, talep kadar kaynak tutarak kümeye yerleştirilir; zamanlayıcı onlara dokunmaz.
- **M2'nin simülasyonda üretilmesi:** `src/tevzi/features/m2.py` içindeki aynı fonksiyonlarla (`Level`, `Window`, `make_state`) simülatörün kendi durumundan; gelişler trace'ten aynen.
- **Küme modeli:** Makine tipleri ve kapasiteleri; gerçekçi yük için orantılı küçültme.
- **Yanlış tahminin bedeli:** Kapasite aşımında ceza — yavaşlama (GPU; referans: Horus, ortalama %18, en fazla 2,4–3,4×) veya yeniden başlatma (bellek/OOM).
- **Doğrulama:** Tahminsiz yaklaşımla bekleme süresi dağılımı trace'e makul ölçüde yakın olmalı, **havuz bazında** (T4 ~saniyeler, V100/V100M32 saatler).
- **Varsayımlar** raporda açıkça listelenecek.

---

## 6. Karşılaştırılacak yaklaşımlar

| # | Yaklaşım | Tahmin kaynağı | Amaç |
|---|---|---|---|
| 0 | **Tahminsiz** | Kullanıcı talebi; geliş sırası | Alt sınır |
| 1 | **Klasik ML** | Gradient boosting (M1, tarihçe dahil) | "Neden derin öğrenme?" |
| 2 | **Yalnızca M1** (DL) | MLP | Zaman serisi katkısının referansı |
| 3 | **M1 + M2, sabit birleştirme** | MLP + zaman serisi encoder'ı | Multimodalitenin katkısı |
| 4 | **M1 + M2, adaptif birleştirme** | + gating/attention | **Önerilen yöntem** |
| 5 | **Kâhin (oracle)** | Gerçek süre ve kullanım | Üst sınır |

**Aday ek baseline (Aşama 3'te karar):** Kullanıcının/grubun son iki biten işinin ortalaması (Tsafrir ve ark., 2007) — literatürde standart, ucuz.

---

## 7. Değerlendirme metrikleri

### 7.1 Birincil ve ikincil metrikler
| Alan | Birincil | İkincil |
|---|---|---|
| Süre tahmini | log ölçekte MAE; aynı saat içinde gelen iş çiftlerinde Kendall τ | RMSE, R², MAPE (dikkatle), tüm test üzerinde Spearman ρ |
| Kullanım tahmini | MAE; **az tahmin oranı** | Yoğun / sakin dönemlere göre ayrıştırılmış hata |
| Zamanlama | Ortalama JCT; GPU kullanım oranı (denge eğrisi üzerinde) | Medyan ve 95. yüzdelik JCT, bekleme süresi, kaynak israfı, taşma/ceza oranı, makespan |

- Tüm sonuçlar ayrıca **yoğun / sakin** dönemlere, **patlama dahil / hariç** test dönemine ve **GPU tipi havuzuna** göre ayrıştırılır. Süre tahmini ayrıca **grup tarihçesi olan / olmayan** işler ve **süre dilimleri** (uzun işler) için raporlanır.

### 7.2 Zamanlama metriklerinin tanımları
| Metrik | Tanım |
|---|---|
| **JCT** | Gönderilmeden bitişe kadar geçen süre |
| **Bekleme süresi** | Gönderilmeden çalışmaya başlamaya kadar |
| **GPU kullanım oranı** | Kümede gerçekten kullanılan GPU / toplam GPU (zaman ortalaması) |
| **Kaynak israfı** | Ayrılan ama kullanılmayan kaynak |
| **Taşma / ceza oranı** | Yanlış tahmin nedeniyle cezaya uğrayan iş yüzdesi |
| **Makespan** | Tüm işlerin bitmesi için geçen toplam süre |

### 7.3 Ana sonuç grafikleri (hedef)
1. Denge eğrisi (verimlilik vs. ortalama JCT, α değiştikçe).
2. Kâhin farkının kapatılma oranı.
3. Gating ağırlıkları – küme yoğunluğu / havuz ilişkisi.
4. Ablation tablosu (M1 grupları A–E, M2 kanal kümeleri, N, adım).
5. Tahmin kalitesi ↔ zamanlama ilişkisi.

### 7.4 Güvenilirlik ve yorumlanabilirlik
- Her model en az 3–5 tohum; ortalama ± standart sapma; anlamlılık kontrolü.
- GBM için SHAP.
- **Model seçim ölçütü (Aşama 3 kararı):** yalnız tahmin metriği mi, doğrulama dönemindeki simüle JCT mi (NeurIPS 2026, "Better Predictions, Worse Schedules").

---

## 8. Kapsam

### Kesin yapılacaklar
- Trace'in incelenmesi ✅, temizlenmesi ✅ ve özellik çıkarımı ✅
- M2 zaman serisinin üretilmesi ✅
- 6 yaklaşımın (0–5) uygulanması
- Olay tabanlı küme simülatörü (ceza modeli ve arka plan yükü dahil)
- Tahmin ve zamanlama deneyleri, ablation, denge eğrileri
- Rapor ve açık kaynak kod deposu

### Zaman kalırsa
- Graf modalitesi (M3) + GNN encoder
- Ek sezgisel baseline'lar (ör. Tiresias tarzı)
- Farklı yük senaryoları (yapay olarak yoğunlaştırılmış trace)
- Sansürlü (Running) işleri kullanan süre modelleri — **not:** Running kayıtlarının çoğu gerçekte çalışmıyor (durumu güncellenmemiş), bu madde zayıfladı
- `cluster-trace-gpu-v2026` ile genelleme testi

### Kapsam dışı
- Gerçek bir kümede çalıştırma; RL ile zamanlama; sıfırdan yeni algoritma; yalnız CPU işleri; başarısızlık tahmini; makine seviyesi modalite (işin yerleşeceği makine tahmin anında bilinmiyor).

---

## 9. Yol haritası (aşamalar)

### Aşama 0 — Kurulum ✅ (30.10.2026)

### Aşama 1 — Veri edinme ve keşif ✅ (7.11.2026)
- **Çıktı:** `notebooks/02_exploration.ipynb`, `docs/KESIF_OZETI.md`, plan v1.2

### Aşama 2 — Veri hazırlama ✅ (8.11.2026)
- [x] M2'nin kaynağına karar (D: kanal bazlı karma)
- [x] Kalan keşif kontrolleri (Failed/Running aralıkları, `machine_metric`, yalnız CPU görevleri)
- [x] Dışlama kuralları + dışlama akışı tablosu (`cleaning.py`)
- [x] İş seviyesinde birleştirme; hedefler; ana görev (`targets.py`)
- [x] Zamansal bölme + purging; eğitim 19. günde (`split.py`)
- [x] M1 özellikleri + 7 sızıntı testi (`features/m1.py`)
- [x] M2 zaman serisi (5 dk, 2 saat + anlık görüntü, havuz bazlı) + 6 sızıntı testi (`features/m2.py`); duyarlılık analizi; `machine_metric` sağlaması
- **Çıktı:** Drive `data/processed/` (`jobs_flags`, `targets`, `m1_features`, `m2_grid`, `m2_snapshots` parquet; `m2_capacities.json`), `results/tables/` (`exclusion_flow`, `split_summary`, `m2_sensitivity_failed`), `notebooks/03_data_preparation.ipynb`, plan v1.3, KESIF güncellemesi, `docs/LITERATUR_NOTLARI.md`

### Aşama 3 — Tahmin modelleri
- [ ] Kategorik kodlama ve ölçekleme (yalnız eğitimde fit)
- [ ] Yaklaşım 1: Gradient boosting (+ SHAP); ek baseline kararı (Tsafrir)
- [ ] Karar 14 maliyet deneyi (GBM, eğitim 7–45 vs 19–45)
- [ ] Yaklaşım 2: Yalnızca M1
- [ ] M2 encoder seçimi (LSTM / TCN / Transformer)
- [ ] Yaklaşım 3: Sabit birleştirme
- [ ] Yaklaşım 4: Adaptif birleştirme
- [ ] Hedef yapısı kararları (hurdle, bellek kuyruğu, kayıp ağırlıkları, asimetrik kayıp)
- [ ] Model seçim ölçütü kararı
- [ ] Tahmin metrikleriyle değerlendirme (AS1, AS2'nin tahmin tarafı); M1 ve M2 ablation'ları; havuz ve tarihçe alt kümeleri
- **Çıktı:** Eğitilmiş modeller + tahmin sonuçları tablosu

### Aşama 4 — Simülatör
- [ ] Olay tabanlı çekirdek; arka plan yükü; simülasyonda M2 üretimi
- [ ] Zamanlayıcı (sıralama + instance seviyesinde yerleştirme + güvenlik payı)
- [ ] Ceza modeli
- [ ] Küme ölçeklendirme ve havuz bazında bekleme doğrulaması
- **Çıktı:** Doğrulanmış simülatör

### Aşama 5 — Zamanlama deneyleri
- [ ] Tüm yaklaşımlar × α × tohumlar
- [ ] Yük, havuz ve patlama dahil/hariç ayrıştırılmış analiz
- [ ] Sağlamlık analizi (AS4); arka plan duyarlılık koşusu
- [ ] Ablation
- **Çıktı:** Sonuç tabloları ve grafikler

### Aşama 6 — Analiz ve rapor
- [ ] AS1–AS4 yorumları; sınırlılıklar ve geçerlilik tehditleri; rapor; depo temizliği ve belgeleme
- **Çıktı:** Final rapor + açık kaynak depo

---

## 10. Riskler ve önlemler

| Risk | Etki | Önlem |
|---|---|---|
| Kapsamın büyümesi | Proje bitmez | Kesin / opsiyonel ayrımı; aşama aşama |
| **M2'nin süre/kullanım tahminine katkısının küçük çıkması** (Aşama 2'de ön kanıt var) | Ana hipotez zayıflar | Olumsuz sonuç da bulgudur; ölçüm aracı doğrulandı (baskı ↔ bekleme), dolayısıyla sonuç ölçüm hatası değil; katkı havuz, yoğunluk ve tarihçe alt kümelerinde aranacak; gating analizi; diğer katkılar (kullanım tahmini, GPU paylaşımlı simülasyon, AS3) bundan bağımsız |
| Simülatörün gerçekçi olmaması | Sonuçlar inandırıcı olmaz | Havuz bazında bekleme doğrulaması; varsayımların açık yazılması; ceza modeli |
| Veri sızıntısı | Sonuçlar yapay iyi | Zamansal bölme + purging; as-of tarihçe; M2'de kesin "önce" kuralı; 13 birim testi |
| Seçilim yanlılığı (yalnız Terminated) | Genellenebilirlik sınırlı | Dışlanan işlerin istatistikleri; geçerlilik tehditleri |
| Eksik Failed yükü (M2) | Küme durumu olduğundan hafif | Duyarlılık analizi (doluluk ρ = 0,93, baskı ρ = 0,86); raporlama |
| Test dönemindeki patlama | Yanıltıcı ortalamalar | Patlama dahil/hariç ayrık raporlama |
| Eğitim ile simülasyon arasında M2 farkı | AS4 karışır | Kanal bazlı kaynak (D); aynı fonksiyonlar; farkın ayrıca ölçülmesi |
| Uzun işlerin doğrulamada az temsili | Model seçimi yanlı | Uzun işlerin ayrı raporlanması; model seçim ölçütü kararı |
| "Multimodal" teriminin sorgulanması | Akademik itiraz | Tablo + zaman serisi = farklı yapı, farklı encoder, farklı bilgi kaynağı; hocayla görüşme |
| Hesaplama / Colab / RAM | Yavaşlama, kayıp | Küçük modeller; Drive'da ara dosyalar; ızgarada saklama (pencereler eğitimde kesiliyor) |

---

## 11. Karar günlüğü

| Tarih | Karar | Gerekçe |
|---|---|---|
| 27.10.2026 | Modaliteler: tablo (M1) + küme dinamikleri zaman serisi (M2) | Farklı yapı ve encoder → savunulabilir multimodalite |
| 27.10.2026 | Zaman serisi küme seviyesinde trace'ten üretilecek | İş başına kullanım verisi özet halinde |
| 27.10.2026 | Tahmin hedefi: süre + gerçek kaynak kullanımı (multi-task) | JCT ve verimlilik hedeflerine doğrudan bağlı |
| 27.10.2026 | Model tahmin eder, karar kural tabanlı zamanlayıcıda | Adil karşılaştırma; yorumlanabilirlik |
| 27.10.2026 | Baseline seti: tahminsiz, klasik ML, kâhin + DL basamakları | Alt sınır, "neden DL", üst sınır |
| 27.10.2026 | Graf modalitesi opsiyonel | Kapsam kontrolü |
| 30.10.2026 | Proje adı: Tevzi | — |
| 30.10.2026 | Hesaplama Colab'da, veri Drive'da, kod GitHub'da | Veri boyutu |
| 30.10.2026 | Depo private; proje sonunda public + lisans | Yayımlanmamış araştırma |
| 30.10.2026 | Kod ve commit dili İngilizce, plan ve rapor Türkçe | Açık kaynak hedefi |
| 7.11.2026 | Aşama bazlı numaralı notebook'lar; tekrar kullanılan kod `src/tevzi/` | Tekrar üretilebilirlik |
| 7.11.2026 | Analiz birimi iş; yerleştirme instance seviyesinde | JCT iş başına; DL zamanlayıcı literatürü |
| 7.11.2026 | Süre hedefi = çalışma süresi, log; bekleme ayrı referans | Bekleme hedefe girerse küme durumu hedefe sızar |
| 7.11.2026 | Eğitim ve simülasyon yalnız Terminated işlerle | Failed bitişi %90 eksik, Running sansürlü |
| 7.11.2026 | Kullanım hedefleri: GPU oranı + tepe bellek oranı | En büyük israf GPU'da; OOM en sert ceza |
| 7.11.2026 | Zamansal bölme, purging, patlama testte | Sızıntı önleme; AS4 stres senaryosu |
| 7.11.2026 | Metrikler birincil/ikincil önceden ayrıldı | Seçici raporlamayı önlemek |
| 7.11.2026 | Dışlama kuralları + akış tablosu | Fiziksel olarak imkânsız/eksik kayıtlar |
| 7.11.2026 | M1 özellik listesi (A–E), gruplar ablation birimi | Grup %88 açıklıyor ama %25 tekil → tarihçe |
| 8.11.2026 | **M2 kaynağı: D "kanal bazlı karma"** — gelişler tüm işlerden, doluluk/kuyruk Terminated işlerden; simülasyonda arka plan yükü | Önce C (tüm trace + arka plan) seçilip doğrulama şartı konuldu; Failed bitişlerinin %84'ü kayıtsız çıktı (instance ve machine_metric de kurtarmadı) → uydurma veri yerine kanal türüne göre kaynak. Running kayıtları gerçekte çalışmıyor |
| 8.11.2026 | Yalnız CPU işleri M2'nin CPU/bellek kanallarında | CPU görevlerinin çoğu GPU işlerinin PS görevleri; GPU makinelerinde yarışıyorlar; MLaaS: CPU darboğaz olabiliyor |
| 8.11.2026 | `machine_metric` model girdisi değil, doğrulama referansı | Simülasyonda yeniden üretilemez → yeni skew |
| 8.11.2026 | Temizlik bayraklarla, iş seviyesinde; kısa işler dışlanmıyor | İzlenebilirlik; gang scheduling; SJF kazancı |
| 8.11.2026 | **Ana görev:** toplam GPU talebi; eşitlikte plan_gpu → rol önceliği → ad | 8.360 eşitlikli işte alfabetik eşitlik bozma evaluator/PS seçiyordu |
| 8.11.2026 | **Eğitim 19. günde başlıyor** (7–19 uzatılmış ısınma) | Sensor kaydı gün 18'de başlıyor (öncesi %0, sonrası %94–99); tüm modeller aynı ölçüm koşullarında; maliyet Aşama 3'te ayrı deneyle ölçülecek |
| 8.11.2026 | **M1:** medyan yerine `last`/`last2`; başarısızlık oranı çıkarıldı; kullanım tarihçesi eklendi | Tsafrir ve ark. 2007 (yakın geçmiş); Failed bitişlerinin bilinme anı belirsiz (sızıntı/yanlılık) |
| 8.11.2026 | **M2 adımı 5 dk**, son adım gönderilme anının tam durumu, ızgarada saklama | İş süresi medyanı ~9 dk; patlama ölçeği; anlık görüntü baskı–bekleme ilişkisini artırdı (V100 0,29 → 0,37) |
| 8.11.2026 | **M2 penceresi 2 saat (24 adım)** | Süre artıklarıyla ilişki 2 saatte yoğun, 12 saatte yok; günlük döngü M1 zaman özelliklerinde |
| 8.11.2026 | **M2 havuz bazlı** | Küme geneli durum hiçbir şeyi açıklamadı (beklemeyi bile); önceden konan ölçüt (bekleme ile |ρ| ≥ 0,10) havuz baskısıyla sağlandı (V100M32 0,38, V100 0,29, P100 0,14); MLaaS ve DeepShare ile tutarlı. Havuzlar birleşince ilişki tersine dönüyor (Simpson) |
| 8.11.2026 | **M2 kanal listesi** (7 küme + 4 kendi havuzu); MISC → tüm havuzlar; standartlaştırma eğitimden | Literatürdeki anlık görüntü özellikleri + havuz baskısı + kullanım |
| 8.11.2026 | Failed işlerin kuyruğu baskıya eklenmiyor | Simülatör Failed işleri oynatmıyor → eklemek eğitim–simülasyon uyumsuzluğu yaratır; taviz duyarlılıkla ölçüldü |
| 8.11.2026 | Kullanım bazlı kanallar tutuldu | Önceden konan ölçüt (makine ölçümüyle ρ ≥ 0,5) referansın ölçtüğü günlerde (≥ 24) sağlandı: ρ = 0,63 |

---

## 12. Hâlâ açık olan konular

- [ ] Hocadan "multimodal" çerçevesi için onay (Aşama 2 bulgusu da paylaşılmalı: M2'nin katkısı küçük ve havuza bağlı olabilir)
- [ ] M2 encoder türü (Aşama 3)
- [ ] Gating mi attention mı (Aşama 3)
- [ ] Hedef yapısı: hurdle, bellek kuyruğu, kayıp ağırlıkları, asimetrik kayıp (Aşama 3)
- [ ] Model seçim ölçütü (Aşama 3)
- [ ] Tsafrir ek baseline'ı (Aşama 3)
- [ ] Ceza modelinin kesin biçimi ve parametreleri (Aşama 4)
- [ ] Simüle edilecek küme boyutu ve ölçekleme oranı (Aşama 4)
- [ ] Rapor formatı (dil: Türkçe)
