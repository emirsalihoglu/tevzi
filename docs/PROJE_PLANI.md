# Adaptif Multimodal Kaynak Zamanlama — Proje Planı

**Proje adı:** Tevzi — https://github.com/emirsalihoglu/tevzi
**İngilizce başlık:** Adaptive Multimodal Resource Scheduling for Resource Efficiency and Job Completion Time
**Belge sürümü:** v1.2 — 7 Kasım 2026
**Durum:** Aşama 1 tamamlandı, Aşama 2'ye geçiliyor
**İlgili belge:** `docs/KESIF_OZETI.md` (Aşama 1 bulguları)

---

## 1. Projenin özü (tek paragraf)

GPU kümelerinde yapay zekâ işlerini sunuculara dağıtan bir zamanlayıcı için, işin **süresini** ve **gerçek kaynak kullanımını** tahmin eden bir derin öğrenme modeli geliştiriyoruz. Model iki farklı yapıdaki veriyi, farklı encoder'larla işliyor: **statik iş özellikleri (tablo)** ve **küme dinamikleri (zaman serisi)**. Bu iki kaynağın birleştirme ağırlıkları, kümenin o anki durumuna göre **adaptif** olarak ayarlanıyor. Asıl amaç yalnızca iyi tahmin yapmak değil; **tahmin kalitesinin ve adaptifliğin, gerçek zamanlama kararlarına (verimlilik ve tamamlanma süresi) ne kadar katkı sağladığını** dikkatli deneylerle ölçmek.

**Keşiften gelen motivasyon:** Trace'te kullanıcılar sistematik olarak fazla kaynak istiyor. Medyan deneme, istediği GPU'nun yalnızca %4'ünü, istediği belleğin %20'sini kullanıyor. İş süreleri 17 saniyeden 7 güne yayılıyor (medyan ~11 dk). Hem yerleştirmede (kullanım tahmini) hem sıralamada (süre tahmini) kazanılacak alan büyük.

---

## 2. Araştırma soruları

### Ana soru
GPU kümelerinde, statik iş özelliklerini ve küme dinamiklerini (zaman serisi) farklı encoder'larla işleyip birleştirme ağırlıklarını küme durumuna göre ayarlayan adaptif bir multimodal model, zamanlama kararlarında kaynak verimliliği ile iş tamamlanma süresi arasındaki dengeyi iyileştirir mi?

### Alt sorular
| Kod | Soru | Nasıl cevaplanacak |
|---|---|---|
| **AS1 – Modalite katkısı** | Küme dinamiklerini zaman serisi olarak eklemek, yalnızca iş özelliklerine dayalı tahmine göre doğruluğu artırıyor mu? | Yaklaşım 2 ile 3'ün tahmin metrikleri karşılaştırması + ablation |
| **AS2 – Adaptiflik** | Durum-bağımlı birleştirme, sabit birleştirmeden daha iyi mi? Fark hangi koşullarda ortaya çıkıyor? | Yaklaşım 3 ile 4'ün, yük koşullarına göre ayrıştırılmış karşılaştırması + gating ağırlıklarının analizi |
| **AS3 – Tahminden karara** | Tahmin doğruluğundaki iyileşme, zamanlama metriklerine yansıyor mu? Yansımıyorsa neden? | Tüm yaklaşımların simülasyon sonuçları; tahmin hatası ve sıralama kalitesi ↔ JCT/verimlilik ilişkisi |
| **AS4 – Sağlamlık** | Model, zamanlayıcının kendi kararlarıyla oluşan ve eğitimde görülmemiş küme durumlarında tutarlı mı? | Simülasyonda üretilen durumlarda tahmin hatasının trace'teki durumlardaki hatayla karşılaştırılması; test dönemindeki gün 59–60 patlamasının ayrık analizi |

### İlke
Önerilen yöntemin daha iyi olacağını **baştan varsaymıyoruz**. Olumsuz veya karışık sonuçlar da "neden ve hangi koşullarda" analiziyle birlikte raporlanacak. Sonuçları yorumlayacağımız **birincil metrikler, sonuçlar görülmeden önce** belirlendi (Bölüm 7).

---

## 3. Veri

### Kaynak
- **Alibaba Cluster Trace — cluster-trace-gpu-v2020** (Alibaba PAI; ~6.500 GPU, 1.897 makine; 68,9 günlük kayıt).
- Atıf: Weng ve ark., *MLaaS in the Wild*, NSDI 2022.
- Depo ayrıca bir GPU zamanlama simülatörü kodu içeriyor; referans olarak incelenecek.

### Doğrulanmış içerik (Aşama 1)
Ayrıntılar `docs/KESIF_OZETI.md` içinde.
- **Hiyerarşi:** iş (1.055.501) → görev (1.261.050) → instance denemesi (7.522.002). Anahtarlar tutarlı. `inst_id` iş anahtarı, `worker_name` deneme anahtarı.
- **Talepler** görev seviyesinde ve instance başına tanımlı: `plan_cpu` (çekirdek %), `plan_mem` (GB), `plan_gpu` (GPU %, kesirli talepler yaygın), istenen GPU tipi.
- **Gerçek kullanım** (sensor): deneme başına özet (CPU %, GPU kullanım %, ortalama/tepe bellek GB, GPU belleği GB). İşlerin %79'unu kapsıyor.
- **Zamanlar** göreli saniye. job `start_time` gönderilme anı, ilk instance başlangıcı çalışmanın başlangıcı.
- **Makineler:** 6 tip. Kapasiteler çekirdek, GB bellek ve GPU adedi olarak verilmiş. GPU belleği kapasitesi yok.
- **Ek etiketler** (group_tag): kullanıcı, grup (tekrar eden iş imzası), workload (%90 boş).

> **Not:** Kullanım verisi iş başına **özetlenmiş** olarak geliyor; tek bir işin zaman içindeki kullanım eğrisi yok. Zaman serisi modalitesi bu yüzden **küme seviyesinde bizim tarafımızdan üretiliyor** (Bölüm 4.2).

### Kapsam ve dışlama kuralları
| Kural | Uygulama |
|---|---|
| Durum | Yalnızca **Terminated** işler (Failed: bitiş %90 eksik; Running: sansürlü; Waiting: başlamamış) |
| Yalnız CPU işleri (`gpu_type = None`) | Kapsam dışı (çalışma GPU zamanlaması üzerine) |
| Eksik talep (CPU/bellek/GPU) | Dışlanır |
| GPU talebi makine kapasitesini aşan görev (169) | Dışlanır |
| `plan_cpu = 0` (11 görev) | Dışlanır |
| Gözlenen kullanımı makine kapasitesini aşan iş | Dışlanır (ölçüm hatası) |
| İlk instance başlangıcı gönderilmeden önce görünen iş | Dışlanır (zaman tutarsızlığı) |
| Instance karşılığı olmayan sensor satırları | Yok sayılır |
| Ana görevinde kullanım verisi olmayan iş | Süre eğitiminde kalır, kullanım kaybı maskelenir; **simülasyona girmez** |

Her kuraldan sonra kalan iş sayısı Aşama 2'de bir **dışlama akışı tablosu** ile raporlanacak. Dışarıda kalan işlerin (özellikle Failed) tanımlayıcı istatistikleri, seçilim yanlılığının boyutunu göstermek için ayrıca verilecek.

### Veri bölme (zamansal)
Gün sayıları ilk işin gönderildiği andan (t₀ = 494.319 s) itibaren verilmiştir.

| Bölüm | Günler | Gerekçe |
|---|---|---|
| Isınma (dışarıda) | 0–7 | Trace bu dönemde eksik kayıtlı. M2 için geçmiş penceresi olarak kullanılır |
| Eğitim | 7–45 | Sakin dönemler ve orta büyüklükteki yoğunluk (gün 32–34) |
| Doğrulama | 45–52 | Tam bir hafta. Hiperparametreler ve kayıp ağırlıkları burada seçilir |
| Test | 52–69 | Sakin günler ve gün 59–60 patlaması |

- **Rastgele değil zamansal bölme:** Rastgele bölme gelecekten bilgi sızdırır.
- **Purging:** Eğitim (ve doğrulama) kümesine yalnızca o dönemde gelen **ve** dönem bitmeden tamamlanan işler alınır.
- **Patlama:** Gün 58,8–60,5 arasındaki patlama (tek kullanıcı, %79,5 Failed) testte kalır. Test sonuçları hem tüm dönem için hem patlama hariç raporlanır.

---

## 4. Model tasarımı

### 4.1 Modalite 1 (M1) — Statik iş özellikleri
- **Yapı:** Tablo (sayısal + kategorik), iş başına bir satır.
- **Kural:** Yalnızca iş **geldiği anda** bilinen bilgiler kullanılır. İşin durumu, bitişi ve kendi kullanım verisi asla girdi olamaz.
- **Encoder (aday):** MLP; kategorik alanlar için embedding katmanları.
- **Özellik grupları** (her grup aynı zamanda bir **ablation birimi**):

| Grup | Özellikler |
|---|---|
| **A. Talepler** | Ana görevin instance başına `plan_gpu`, `plan_cpu`, `plan_mem`; `inst_num`; iş toplamı GPU/CPU/bellek talebi; istenen GPU tipi; talebin şekli (kesirli GPU mu, GPU başına bellek ve CPU) |
| **B. İş yapısı** | Görev sayısı; PS var mı; PS ve worker instance sayıları; ana görev türü |
| **C. Kimlik ve iş türü** | Kullanıcı (seyrek ve görülmemiş kullanıcılar "bilinmeyen" kategorisinde); workload (+ eksiklik göstergesi) |
| **D. Tarihçe** | Grup ve kullanıcı bazında, yalnızca gelişten **önce bitmiş** işlerden: geçmiş iş sayısı, log-süre ortalaması/medyanı/standart sapması, son biten işin süresi, GPU ve bellek kullanım oranı ortalaması, başarısızlık oranı; grup daha önce görüldü mü; grubun son başvurusundan bu yana geçen süre; grubun son 1 saatteki başvuru sayısı |
| **E. Zaman** | Geliş saati ve haftanın günü (göreli saat, sin/cos kodlama) |

- **Bilerek dışarıda bırakılanlar:** `gpu_type_spec` (%98 boş); ham `group` kimliği (317 bin değer, %25'i tekil; bilgisi D grubuyla taşınıyor); iş/görev kimlikleri.
- **Sızıntı kontrolü:** Tarihçe hesaplamaları için birim testleri yazılacak (gelecekteki işlerin geçmiş özelliğe karışmadığının kanıtı).

### 4.2 Modalite 2 (M2) — Küme dinamikleri (zaman serisi)
- **Yapı:** Çok değişkenli zaman serisi.
- **Nasıl üretilecek:** İşlerin gönderilme, başlangıç ve bitiş zamanlarından, her zaman adımında kümenin durumu hesaplanacak. İş geldiğinde, ondan önceki son **N zaman adımı** modele verilecek. Isınma dönemi (gün 0–7), sonraki işlerin geçmiş penceresini doldurur.
- **Kanallar (aday):**
  - Toplam GPU / CPU / bellek doluluk oranı (talep bazlı ve kullanım bazlı)
  - Çalışan iş sayısı
  - Kuyrukta bekleyen iş sayısı ve kuyruktaki toplam talep
  - Son pencerede iş geliş hızı
  - (Opsiyonel) GPU tipine göre ayrıştırılmış doluluk
  - (Opsiyonel) `machine_metric` tabanlı makine seviyesinde kullanım
- **Encoder (aday):** LSTM, TCN veya küçük bir Transformer. Aşama 3'te karşılaştırılıp biri seçilecek.
- **Açık parametreler:** Zaman adımı uzunluğu (ör. 1 / 5 / 10 dk), pencere uzunluğu N, **M2'nin hangi işlerden hesaplanacağı** (tüm trace mi, simülasyonla tutarlı iş kümesi mi; AS4 ile bağlantılı, Aşama 2'de karar verilecek).
- **Kural:** Yalnızca iş gelmeden **önceki** zaman adımları kullanılır.

### 4.3 Birleştirme (fusion)
- **Sabit birleştirme (Yaklaşım 3):** M1 ve M2 temsilleri uç uca eklenir (concatenation) → ortak katmanlar.
- **Adaptif birleştirme (Yaklaşım 4):** M2'den çıkan küme durumu temsili, bir **gating** veya **attention** ağı aracılığıyla her modalitenin ağırlığını belirler.
  - Basit hali: durum vektöründen softmax ile modalite ağırlıkları üretilir.
  - Alternatif: hedef başına (süre / kullanım) ayrı ağırlıklar.
- **Yorumlanabilirlik:** Gating ağırlıkları kaydedilecek ve küme yoğunluğuna göre nasıl değiştikleri analiz edilecek (AS2).

### 4.4 Tahmin hedefleri (multi-task)
| Hedef | Tanım | Ne işe yarar | Not |
|---|---|---|---|
| **Çalışma süresi** | İş bitişi − ilk instance başlangıcı (yeniden denemeler dahil) | Kuyruk sıralaması → bekleme ve JCT | Log dönüşümü. Bekleme süresi hedefe dahil değil |
| **GPU kullanım oranı** | `gpu_wrk_util` / `plan_gpu`, ana görevin instance ve denemeleri arasındaki maksimum | GPU paylaşımlı yerleştirme → verimlilik | Medyan 0,04 |
| **Tepe bellek oranı** | `max_mem` / `plan_mem`, aynı şekilde maksimum | Bellek taşması (OOM) riskinin kontrolü | Medyan 0,20 |

- **Ana görev:** İşin en çok GPU talep eden görevi. PS ve evaluator görevleri kendi talepleriyle yerleştirilir.
- **Tahmin edilmeyenler:** CPU (GPU kümesinde darboğaz değil, talep kullanılır) ve GPU belleği (makine tablosunda kapasitesi yok). Gerekçeler raporda sınırlılık olarak belirtilir.
- **Kayıp fonksiyonu:** Hedeflerin ağırlıklı toplamı; ağırlıklar doğrulama setinde ayarlanacak. Kullanım verisi olmayan işlerde kullanım kaybı **maskelenir**.
- **Bekleme süresi** (ilk instance başlangıcı − gönderilme) hedef değil; simülatör doğrulaması için referans olarak saklanır.
- **Yedek plan:** Uygulamada zorluk çıkarsa önce yalnızca süre ile başlanacak, sonra kullanım hedefleri eklenecek.
- **Beklenti uyarısı:** M2'nin süre tahminine katkısı (kaynak çekişmesi nedeniyle) kullanım tahminine katkısından daha belirgin olabilir. Bu da raporlanacak bir bulgu.

---

## 5. Zamanlayıcı ve simülatör

### 5.1 Zamanlayıcı (tüm yaklaşımlar için aynı)
Model **tahmin eder**, kararı basit ve şeffaf bir kural verir. Yaklaşımlar arasında **yalnızca tahmin kaynağı değişir.**
- **Birim:** İş. Bir işin görevleri birlikte başlatılır (gang scheduling).
- **Sıralama:** Kuyruktaki işler tahmini çalışma süresine göre sıralanır (en kısa önce). Uzun işlerin sonsuza kadar beklememesi için yaşlanma (aging) kuralı eklenecek.
- **Yerleştirme:** Instance başına. Ana görevin instance'ları tahmini kullanım (oran × talep) + güvenlik payına göre, kapasiteyi aşmayan ve sunucuyu en iyi dolduran sunucuya yerleştirilir (best-fit). GPU bölünebilir kaynaktır (kapasite = 100 × GPU adedi). CPU ve diğer görevler talepleriyle yerleştirilir. İstenen GPU tipi kısıtı uygulanır (MISC = herhangi bir tip).
- **Güvenlik payı (α):** Tahmine eklenen pay (ör. %0, %10, %20, %30, %50). Farklı α değerleri, verimlilik–JCT **denge eğrisini** oluşturur.

### 5.2 Simülatör
- **Tür:** Olay tabanlı (discrete-event) simülasyon; olaylar: iş gelişi, iş bitişi, zamanlama kararı.
- **Girdi:** Test dönemindeki, dışlama kurallarından geçmiş, **kullanım verisi olan Terminated** işlerin gerçek gönderilme zamanları, talepleri, gerçek çalışma süreleri ve gerçek kullanımları.
- **Küme modeli:** Trace'teki makine tipleri ve kapasitelerinden (Bölüm 3). Simülasyona giren iş kümesi gerçek yükten küçük olduğundan, gerçekçi bir yük seviyesi için küme orantılı olarak küçültülecek.
- **Yanlış tahminin bedeli (kritik):** Bir sunucuda işlerin gerçek kullanımı kapasiteyi aşarsa ceza uygulanır. Aday modeller:
  - Yavaşlama: aşırı yüklenen sunucudaki işlerin süresi uzar (GPU aşımı için doğal aday)
  - Yeniden başlatma: işlerden biri durdurulup kuyruğa geri gönderilir (bellek aşımı / OOM için doğal aday; trace'te gözlenen yeniden deneme davranışıyla uyumlu)
  - Seçilen model ve parametreleri raporda açıkça belirtilecek.
- **Doğrulama:** Tahminsiz yaklaşımla çalıştırıldığında trace'teki bekleme süresi dağılımına (medyan 7 s, %90 109 s, uzun kuyruk) makul ölçüde yakın sonuç vermeli.
- **Varsayımlar** (raporda açıkça listelenecek): ağ/topoloji etkisi yok sayılır, gang scheduling basitleştirilir, GPU belleği modellenmez vb.

---

## 6. Karşılaştırılacak yaklaşımlar

| # | Yaklaşım | Tahmin kaynağı | Amaç |
|---|---|---|---|
| 0 | **Tahminsiz** | Kullanıcı talebi; sıralama geliş sırasına göre | Alt sınır, bugünkü standart davranış |
| 1 | **Klasik ML** | Gradient boosting (M1 özellikleri, tarihçe dahil) | "Neden derin öğrenme?" sorusuna cevap; güçlü temel model |
| 2 | **Yalnızca M1** (DL) | MLP | Zaman serisi katkısının referansı |
| 3 | **M1 + M2, sabit birleştirme** | MLP + zaman serisi encoder'ı | Multimodalitenin katkısı |
| 4 | **M1 + M2, adaptif birleştirme** | + gating/attention | **Önerilen yöntem** |
| 5 | **Kâhin (oracle)** | Gerçek süre ve kullanım | Üst sınır: mükemmel tahminle ulaşılabilecek en iyi sonuç |

**Temel ilke:** Her adım bir öncekinden yalnızca bir şeyi değiştirir; böylece her farkın kaynağı net görülür.

---

## 7. Değerlendirme metrikleri

### 7.1 Birincil ve ikincil metrikler
Sonuçlar **birincil** metriklerle yorumlanır. İkincil metrikler destekleyici olarak tablolarda verilir. Bu ayrım, sonuçlar görülmeden önce yapıldı.

| Alan | Birincil | İkincil |
|---|---|---|
| Süre tahmini | log ölçekte MAE; aynı saat içinde gelen iş çiftlerinde Kendall τ | RMSE, R², MAPE (dikkatle), tüm test üzerinde Spearman ρ |
| Kullanım tahmini | MAE; **az tahmin oranı** (tahmin < gerçek; zamanlama açısından en kritik hata) | Yoğun / sakin dönemlere göre ayrıştırılmış hata |
| Zamanlama | Ortalama JCT; GPU kullanım oranı (denge eğrisi üzerinde) | Medyan ve 95. yüzdelik JCT, bekleme süresi, kaynak israfı, taşma/ceza oranı, makespan |

- **Sıralama metriklerinin gerekçesi:** Zamanlayıcı süre tahminini yalnızca sıralamak için kullanıyor. Saat içi Kendall τ, yalnızca aynı anda kuyrukta bulunabilecek işleri karşılaştırdığı için tahmin kalitesi ile zamanlama sonucu arasındaki köprüyü kurar (AS3).
- Tüm sonuçlar ayrıca **yoğun / sakin** dönemlere ve **patlama dahil / hariç** test dönemine göre ayrıştırılır.

### 7.2 Zamanlama metriklerinin tanımları
| Metrik | Tanım |
|---|---|
| **JCT** | İşin gönderilmesinden bitişine kadar geçen süre (ortalama + medyan + 95. yüzdelik) |
| **Bekleme süresi** | Gönderilmeden çalışmaya başlamaya kadar geçen süre |
| **GPU kullanım oranı** | Kümede gerçekten kullanılan GPU / toplam GPU (zaman ortalaması) |
| **Kaynak israfı** | Ayrılan ama kullanılmayan kaynak miktarı |
| **Taşma / ceza oranı** | Yanlış tahmin nedeniyle cezaya uğrayan iş yüzdesi |
| **Makespan** | Tüm işlerin bitmesi için geçen toplam süre |

### 7.3 Ana sonuç grafikleri (hedef)
1. **Denge eğrisi:** x = kaynak verimliliği, y = ortalama JCT; her yaklaşım için α değiştikçe bir eğri.
2. **Kâhin farkının kapatılma oranı:** (Yaklaşım − Tahminsiz) / (Kâhin − Tahminsiz).
3. **Gating ağırlıkları – küme yoğunluğu** ilişkisi.
4. **Ablation tablosu:** M1 özellik grupları (A–E) ve modaliteler / bileşenler çıkarıldığında sonuçlar.
5. **Tahmin kalitesi ↔ zamanlama** ilişkisi (sıralama metriği vs. JCT).

### 7.4 Güvenilirlik ve yorumlanabilirlik
- Her model en az 3–5 farklı rastgele tohum (seed) ile eğitilecek (GPU bütçesi uygunsa daha fazla); ortalama ± standart sapma raporlanacak.
- Farklar için istatistiksel anlamlılık kontrolü yapılacak.
- GBM için SHAP ile özellik katkısı analizi (Aşama 3).

---

## 8. Kapsam

### Kesin yapılacaklar
- Trace'in incelenmesi, temizlenmesi ve özellik çıkarımı ✅ (inceleme) / Aşama 2 (temizleme, özellikler)
- M2 zaman serisinin üretilmesi
- 6 yaklaşımın (0–5) uygulanması
- Olay tabanlı küme simülatörü (ceza modeli dahil)
- Tahmin ve zamanlama deneyleri, ablation, denge eğrileri
- Rapor ve açık kaynak kod deposu

### Zaman kalırsa
- Graf modalitesi (M3) + GNN encoder
- Ek sezgisel baseline'lar (ör. Tiresias tarzı)
- Farklı yük senaryoları (yapay olarak yoğunlaştırılmış trace)
- Sansürlü (Running) işleri kullanan süre modelleri (survival analysis)
- Trace'in daha yeni sürümleriyle genelleme testi (ör. cluster-trace-gpu-v2026)

### Kapsam dışı
- Gerçek bir kümede çalıştırma (yalnızca simülasyon)
- Pekiştirmeli öğrenme (RL) ile zamanlama
- Sıfırdan yeni bir zamanlama algoritması icat etmek
- Yalnız CPU işleri; işin başarısız olup olmayacağının tahmini

---

## 9. Yol haritası (aşamalar)

> Süre kısıtı belirlenmedi; aşamalar sırayla ilerler ve her aşama, çıktıları tamamlanınca kapanır.

### Aşama 0 — Kurulum ✅ (30.10.2026)
- [x] Proje klasör yapısı
- [x] Python ortamı (Colab + Google Drive; yerelde yalnızca kod)
- [x] Git deposu (GitHub, private)
- **Çıktı:** Çalışan boş proje iskeleti

### Aşama 1 — Veri edinme ve keşif ✅ (7.11.2026)
- [x] Trace'in Colab üzerinden Drive'a indirilmesi (sha256 doğrulamalı)
- [x] Ham tabloların Parquet'e dönüştürülmesi (Drive: `Tevzi/data/interim`)
- [x] Yükleyici modül (`src/tevzi/data/loader.py`)
- [x] Tabloların ve alanların incelenmesi; beklentilerin doğrulanması
- [x] Temel istatistikler: süre dağılımı, talep vs. gerçek kullanım, geliş örüntüleri, bekleme
- [x] Veri kalitesi: eksik değerler, tutarsız kayıtlar
- [x] Kapsam, hedef ve bölme kararları; M1 özellik listesi
- **Çıktı:** `notebooks/02_exploration.ipynb`, `docs/KESIF_OZETI.md`, plan v1.2

### Aşama 2 — Veri hazırlama
- [ ] M2'nin kaynağına karar (tüm trace vs. simülasyonla tutarlı iş kümesi)
- [ ] `machine_metric` incelemesi; yalnız CPU görevlerinin sensor kapsamının teyidi
- [ ] Dışlama kurallarının uygulanması + dışlama akışı tablosu
- [ ] İş seviyesinde birleştirme; hedeflerin oluşturulması (çalışma süresi, kullanım oranları, bekleme referansı)
- [ ] M1 özelliklerinin çıkarılması (tarihçe özellikleri dahil) + sızıntı birim testleri
- [ ] M2 zaman serisinin üretilmesi (zaman adımı ve N'nin seçimi)
- [ ] Zamansal eğitim / doğrulama / test bölmesi (purging dahil)
- **Çıktı:** Modele hazır veri setleri (Drive: `Tevzi/data/processed`), `notebooks/03_data_preparation.ipynb`, `src/tevzi/data` ve `src/tevzi/features` modülleri

### Aşama 3 — Tahmin modelleri
- [ ] Yaklaşım 1: Gradient boosting (+ SHAP)
- [ ] Yaklaşım 2: Yalnızca M1
- [ ] M2 encoder seçimi (LSTM / TCN / Transformer kısa karşılaştırma)
- [ ] Yaklaşım 3: Sabit birleştirme
- [ ] Yaklaşım 4: Adaptif birleştirme
- [ ] Tahmin metrikleriyle değerlendirme (AS1, AS2'nin tahmin tarafı) ve özellik grubu ablation'ı
- **Çıktı:** Eğitilmiş modeller + tahmin sonuçları tablosu

### Aşama 4 — Simülatör
- [ ] Olay tabanlı çekirdek
- [ ] Zamanlayıcı (sıralama + instance seviyesinde yerleştirme + güvenlik payı)
- [ ] Ceza modeli
- [ ] Küme ölçeklendirme ve tahminsiz/kâhin yaklaşımlarıyla doğrulama (bekleme süresi referansı)
- **Çıktı:** Doğrulanmış simülatör

### Aşama 5 — Zamanlama deneyleri
- [ ] Tüm yaklaşımlar × α değerleri × tohumlar
- [ ] Yük koşullarına ve patlama dahil/hariç teste göre ayrıştırılmış analiz
- [ ] Sağlamlık analizi (AS4)
- [ ] Ablation
- **Çıktı:** Sonuç tabloları ve grafikler

### Aşama 6 — Analiz ve rapor
- [ ] Sonuçların yorumlanması (AS1–AS4)
- [ ] Sınırlılıklar ve geçerlilik tehditleri
- [ ] Rapor yazımı
- [ ] Kod deposunun temizlenmesi ve belgelenmesi (README, tekrar üretim talimatları)
- **Çıktı:** Final rapor + açık kaynak depo

---

## 10. Riskler ve önlemler

| Risk | Etki | Önlem |
|---|---|---|
| Kapsamın büyümesi | Proje bitmez | Kesin / opsiyonel ayrımına sadık kalmak; aşama aşama ilerlemek |
| M2'nin katkısının düşük çıkması | Ana hipotez zayıflar | Olumsuz sonuç da bulgudur. Bekleme çoğunlukla kısa olduğundan farklar yoğun dönemlerde aranacak (koşullara göre ayrıştırılmış analiz) |
| Simülatörün gerçekçi olmaması | Sonuçlar inandırıcı olmaz | Bekleme süresi referansıyla doğrulama; varsayımları açıkça yazmak; ceza modeli |
| Veri sızıntısı (leakage) | Sonuçlar yapay olarak iyi çıkar | Zamansal bölme + purging; girdilerde yalnızca geliş anında bilinen bilgiler; tarihçe özellikleri için birim testleri; süre hedefinden bekleme süresinin çıkarılması |
| Seçilim yanlılığı (yalnızca Terminated işler) | Sonuçların genellenebilirliği sınırlanır | Dışlanan işlerin tanımlayıcı istatistikleri; geçerlilik tehditlerinde açık raporlama |
| Test dönemindeki patlama sonuçlara baskın çıkar | Yanıltıcı ortalamalar | Patlama dahil/hariç ayrık raporlama |
| Eğitim ile simülasyon arasında M2 dağılım farkı | AS4 sonuçları karışır | M2 kaynağına Aşama 2'de bilinçli karar; farkın ayrıca ölçülmesi |
| "Multimodal" teriminin sorgulanması | Akademik itiraz | Tablo + zaman serisi = farklı yapı, farklı encoder, farklı bilgi kaynağı; hocayla erken görüşme |
| Hesaplama kaynağı yetersizliği | Eğitim yavaşlar | Küçük modeller; gerekirse veri alt kümesi; Colab işlem birimleri (GPU yalnızca eğitimde) |
| Colab oturumunun kapanması | Ara çıktılar kaybolur | Veri ve sonuçlar Drive'da; düzenli checkpoint; kod GitHub'da |
| Bellek (RAM) sınırı | Büyük tablolar işlenemez | Sütun seçerek yükleme; ara nesnelerin silinmesi; gerekirse daha verimli veri tipleri |

---

## 11. Karar günlüğü

| Tarih | Karar | Gerekçe |
|---|---|---|
| 27.10.2026 | Modaliteler: tablo (M1) + küme dinamikleri zaman serisi (M2) | Gerçekten farklı yapı ve encoder → savunulabilir multimodalite |
| 27.10.2026 | Zaman serisi küme seviyesinde trace'ten üretilecek | İş başına kullanım verisi özet halinde |
| 27.10.2026 | Tahmin hedefi: süre + gerçek kaynak kullanımı (multi-task) | Araştırma sorusundaki iki hedefe (JCT ve verimlilik) doğrudan bağlanıyor |
| 27.10.2026 | Model tahmin eder, karar basit kural tabanlı zamanlayıcıda | Yaklaşımlar arası adil karşılaştırma; yorumlanabilirlik |
| 27.10.2026 | Baseline seti: tahminsiz, klasik ML, kâhin + DL basamakları | Alt sınır, "neden DL" sorusu ve üst sınır |
| 27.10.2026 | Graf modalitesi opsiyonel | Kapsam kontrolü |
| 30.10.2026 | Proje adı: Tevzi | — |
| 30.10.2026 | Hesaplama Colab'da, veri ve sonuçlar Google Drive'da; kod yerelde yazılıp GitHub üzerinden Colab'a çekiliyor | Veri boyutu ve eğitim yükü yerel bilgisayar için ağır |
| 30.10.2026 | Depo private; proje sonunda public + lisans | Yayımlanmamış araştırma |
| 30.10.2026 | Kod ve commit dili İngilizce, plan ve rapor Türkçe | Açık kaynak hedefi |
| 7.11.2026 | Aşama bazlı, numaralı, bağımsız notebook'lar; tekrar kullanılan kod `src/tevzi/` içinde; paket ortamdan bağımsız | Tekrar üretilebilirlik ve test edilebilirlik (Cookiecutter Data Science pratiği) |
| 7.11.2026 | **Analiz birimi iş**; görev yapısı M1'e özellik olarak giriyor; yerleştirme instance seviyesinde | JCT iş başına tanımlı; DL küme zamanlayıcıları (Tiresias, Gandiva) işi temel birim alıyor; görev birimi birbirini tekrar eden örnekler ve bölmede sızıntı riski yaratırdı; talepler instance başına (kapasite aşımı 169 / 1,04 M) |
| 7.11.2026 | **Süre hedefi = çalışma süresi** (iş bitişi − ilk instance başlangıcı), log ölçekte; bekleme süresi ayrı referans | Simülatörün ihtiyacı çalışma süresi; bekleme zamanlayıcının çıktısı. Hedefe dahil edilseydi küme durumu hedefe sızar, M2'nin katkısı yapay olarak şişerdi. Yeniden denemeler gerçek kaynak kullanımı olarak süre içinde kalıyor |
| 7.11.2026 | **Eğitim ve simülasyon yalnızca Terminated işlerle** | Failed işlerde bitiş %90 eksik, Running sansürlü; uydurma süreler sonuçları varsayımlara bağlardı. Seçilim yanlılığı geçerlilik tehditlerinde raporlanacak |
| 7.11.2026 | **Kullanım hedefleri: GPU kullanım oranı ve tepe bellek oranı** (talebe göre, ana görevin instance'ları arasında maksimum); CPU ve GPU belleği tahmin edilmiyor | En büyük israf GPU'da (medyan 0,04); bellek aşımı en sert ceza (OOM); maksimum ihtiyatlı ve instance başına yerleştirmeyle uyumlu; GPU belleğinin kapasite karşılığı yok |
| 7.11.2026 | **Zamansal bölme:** ısınma 0–7, eğitim 7–45, doğrulama 45–52, test 52–69. gün; purging; patlama testte, ayrık raporlanıyor | Isınma döneminde eksik kayıt; doğrulama tam bir hafta; sınırda bitmemiş işler sızıntı yaratır; patlama AS4 için gerçek bir stres senaryosu |
| 7.11.2026 | **Metrikler birincil/ikincil olarak önceden ayrıldı**; sıralama metrikleri (Spearman ρ, saat içi Kendall τ) eklendi | Seçici raporlamayı önlemek; zamanlayıcı süreyi sıralama için kullanıyor (AS3 köprüsü) |
| 7.11.2026 | **Dışlama kuralları** (Bölüm 3) ve dışlama akışı tablosu; yalnız CPU işleri kapsam dışı; kullanım verisi olmayan işler süre eğitiminde maskeli kayıpla kalıyor, simülasyona girmiyor | Fiziksel olarak imkânsız veya eksik kayıtlar; çalışma GPU zamanlaması üzerine; kâhin ve ceza hesabı gerçek kullanım gerektiriyor |
| 7.11.2026 | **M1 özellik listesi** (grup A–E, Bölüm 4.1); tarihçe özellikleri yalnızca geçmişte bitmiş işlerden; özellik grupları ablation birimi | Grup log-sürenin %88'ini açıklıyor ama %25'i tekil → ham kimlik yerine tarihçe; tarihçe GBM'e de verilerek güçlü temel model; ablation ile katkıların sayısal ölçümü |

---

## 12. Hâlâ açık olan konular

- [ ] Hocadan "multimodal" çerçevesi için onay
- [ ] M2'nin kaynağı: tüm trace vs. simülasyonla tutarlı iş kümesi (Aşama 2)
- [ ] Zaman adımı uzunluğu ve pencere uzunluğu N (Aşama 2)
- [ ] M2 encoder türü (Aşama 3)
- [ ] Gating mi attention mı (Aşama 3)
- [ ] Ceza modelinin kesin biçimi ve parametreleri (Aşama 4)
- [ ] Simüle edilecek küme boyutu ve ölçekleme oranı (Aşama 4)
- [ ] Rapor formatı (dil: Türkçe)
