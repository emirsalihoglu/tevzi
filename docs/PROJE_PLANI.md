# Adaptif Multimodal Kaynak Zamanlama — Proje Planı

**İngilizce başlık:** Adaptive Multimodal Resource Scheduling for Resource Efficiency and Job Completion Time
**Belge sürümü:** v1.0 — 27 Ekim 2026
**Durum:** Planlama tamamlandı, Aşama 1'e geçilecek

---

## 1. Projenin özü (tek paragraf)

GPU kümelerinde yapay zekâ işlerini sunuculara dağıtan bir zamanlayıcı için, işin **süresini** ve **gerçek kaynak kullanımını** tahmin eden bir derin öğrenme modeli geliştiriyoruz. Model iki farklı yapıdaki veriyi, farklı encoder'larla işliyor: **statik iş özellikleri (tablo)** ve **küme dinamikleri (zaman serisi)**. Bu iki kaynağın birleştirme ağırlıkları, kümenin o anki durumuna göre **adaptif** olarak ayarlanıyor. Asıl amaç yalnızca iyi tahmin yapmak değil; **tahmin kalitesinin ve adaptifliğin, gerçek zamanlama kararlarına (verimlilik ve tamamlanma süresi) ne kadar katkı sağladığını** dikkatli deneylerle ölçmek.

---

## 2. Araştırma soruları

### Ana soru
GPU kümelerinde, statik iş özelliklerini ve küme dinamiklerini (zaman serisi) farklı encoder'larla işleyip birleştirme ağırlıklarını küme durumuna göre ayarlayan adaptif bir multimodal model, zamanlama kararlarında kaynak verimliliği ile iş tamamlanma süresi arasındaki dengeyi iyileştirir mi?

### Alt sorular
| Kod | Soru | Nasıl cevaplanacak |
|---|---|---|
| **AS1 – Modalite katkısı** | Küme dinamiklerini zaman serisi olarak eklemek, yalnızca iş özelliklerine dayalı tahmine göre doğruluğu artırıyor mu? | Yaklaşım 2 ile 3'ün tahmin metrikleri karşılaştırması + ablation |
| **AS2 – Adaptiflik** | Durum-bağımlı birleştirme, sabit birleştirmeden daha iyi mi? Fark hangi koşullarda ortaya çıkıyor? | Yaklaşım 3 ile 4'ün, yük koşullarına göre ayrıştırılmış karşılaştırması + gating ağırlıklarının analizi |
| **AS3 – Tahminden karara** | Tahmin doğruluğundaki iyileşme, zamanlama metriklerine yansıyor mu? Yansımıyorsa neden? | Tüm yaklaşımların simülasyon sonuçları; tahmin hatası ↔ JCT/verimlilik ilişkisi |
| **AS4 – Sağlamlık** | Model, zamanlayıcının kendi kararlarıyla oluşan ve eğitimde görülmemiş küme durumlarında tutarlı mı? | Simülasyon sırasında üretilen durumlarda tahmin hatasının, trace'teki durumlardaki hatayla karşılaştırılması |

### İlke
Önerilen yöntemin daha iyi olacağını **baştan varsaymıyoruz**. Olumsuz veya karışık sonuçlar da "neden ve hangi koşullarda" analiziyle birlikte raporlanacak.

---

## 3. Veri

### Kaynak
- **Alibaba Cluster Trace — cluster-trace-gpu-v2020** (Alibaba PAI; ~6.500 GPU, ~1.800 makine; Temmuz–Ağustos 2020).
- Atıf: Weng ve ark., *MLaaS in the Wild*, NSDI 2022.
- Depo ayrıca bir GPU zamanlama simülatörü kodu içeriyor; referans olarak incelenecek.

### Beklenen içerik (Aşama 1'de doğrulanacak)
- İş / görev / instance hiyerarşisi
- Talep edilen kaynaklar: CPU, GPU (yüzde), bellek
- Gerçek kullanım (özet istatistikler): CPU kullanımı, GPU kullanım oranı, ortalama/maksimum bellek, GPU belleği
- Zaman bilgisi: başlangıç ve bitiş zamanları
- Makine bilgileri: GPU tipi, kapasite
- İş yükü / görev etiketleri

> **Önemli not:** Kullanım verisi iş başına **özetlenmiş** olarak geliyor; tek bir işin zaman içindeki kullanım eğrisi büyük olasılıkla yok. Bu yüzden zaman serisi modalitesini **küme seviyesinde biz üreteceğiz** (Bölüm 4.2).

### Veri bölme
- **Zamansal bölme** (rastgele değil): ilk dönem eğitim, sonraki dönem doğrulama, son dönem test.
- Gerekçe: Rastgele bölme gelecekten bilgi sızdırır (data leakage) ve sonuçları gerçekçi olmaktan çıkarır.

---

## 4. Model tasarımı

### 4.1 Modalite 1 (M1) — Statik iş özellikleri
- **Yapı:** Tablo (sayısal + kategorik)
- **İçerik (aday):** Talep edilen GPU/CPU/bellek, görev tipi, instance sayısı, iş yükü etiketi, talep edilen GPU tipi, geliş saati/gün bilgisi
- **Encoder (aday):** MLP; kategorik alanlar için embedding katmanları
- **Kural:** Yalnızca iş **geldiği anda** bilinen bilgiler kullanılır (gerçek kullanım, süre vb. girdi olamaz).

### 4.2 Modalite 2 (M2) — Küme dinamikleri (zaman serisi)
- **Yapı:** Çok değişkenli zaman serisi
- **Nasıl üretilecek:** İşlerin başlangıç/bitiş zamanlarından, her zaman adımında kümenin durumu hesaplanacak. İş geldiğinde, ondan önceki son **N zaman adımı** modele verilecek.
- **Kanallar (aday):**
  - Toplam GPU / CPU / bellek doluluk oranı (talep bazlı ve kullanım bazlı)
  - Çalışan iş sayısı
  - Kuyrukta bekleyen iş sayısı ve kuyruktaki toplam talep
  - Son pencerede iş geliş hızı
  - (Opsiyonel) GPU tipine göre ayrıştırılmış doluluk
- **Encoder (aday):** LSTM, TCN veya küçük bir Transformer — Aşama 3'te karşılaştırılıp biri seçilecek.
- **Açık parametreler:** Zaman adımı uzunluğu (ör. 1 / 5 / 10 dk), pencere uzunluğu N.
- **Kural:** Yalnızca iş gelmeden **önceki** zaman adımları kullanılır.

### 4.3 Birleştirme (fusion)
- **Sabit birleştirme (Yaklaşım 3):** M1 ve M2 temsilleri uç uca eklenir (concatenation) → ortak katmanlar.
- **Adaptif birleştirme (Yaklaşım 4):** M2'den çıkan küme durumu temsili, bir **gating** veya **attention** ağı aracılığıyla her modalitenin ağırlığını belirler.
  - Basit hali: durum vektöründen softmax ile modalite ağırlıkları üretilir.
  - Alternatif: hedef başına (süre / kullanım) ayrı ağırlıklar.
- **Yorumlanabilirlik:** Gating ağırlıkları kaydedilecek ve küme yoğunluğuna göre nasıl değiştikleri analiz edilecek (AS2).

### 4.4 Tahmin hedefleri (multi-task)
| Hedef | Ne işe yarar | Not |
|---|---|---|
| **İş süresi** | Kuyruk sıralaması → bekleme süresi ve JCT | Dağılımı çok çarpık olacağından log-dönüşümü uygulanacak |
| **Gerçek kaynak kullanımı** (GPU kullanım oranı, tepe bellek — talebe oranla) | Sunucuya yerleştirme → kaynak verimliliği | Talep ile gerçek kullanım arasındaki farkı yakalamak hedefleniyor |

- **Kayıp fonksiyonu:** Hedeflerin ağırlıklı toplamı; ağırlıklar doğrulama setinde ayarlanacak.
- **Yedek plan:** Uygulamada zorluk çıkarsa önce yalnızca süre ile başlanacak, sonra kullanım hedefi eklenecek.
- **Beklenti uyarısı:** M2'nin süre tahminine katkısı (kaynak çekişmesi nedeniyle) kullanım tahminine katkısından daha belirgin olabilir. Bu da raporlanacak bir bulgu.

---

## 5. Zamanlayıcı ve simülatör

### 5.1 Zamanlayıcı (tüm yaklaşımlar için aynı)
Model **tahmin eder**, kararı basit ve şeffaf bir kural verir. Yaklaşımlar arasında **yalnızca tahmin kaynağı değişir.**
- **Sıralama:** Kuyruktaki işler tahmini süreye göre sıralanır (en kısa önce). Uzun işlerin sonsuza kadar beklememesi için yaşlanma (aging) kuralı eklenecek.
- **Yerleştirme:** İş, tahmini gerçek kullanım + güvenlik payına göre, kapasiteyi aşmayan ve sunucuyu en iyi dolduran sunucuya yerleştirilir (best-fit).
- **Güvenlik payı (α):** Tahmine eklenen pay (ör. %0, %10, %20, %30, %50). Farklı α değerleri, verimlilik–JCT **denge eğrisini** oluşturur.

### 5.2 Simülatör
- **Tür:** Olay tabanlı (discrete-event) simülasyon; olaylar: iş gelişi, iş bitişi, zamanlama kararı.
- **Girdi:** Test dönemindeki işlerin gerçek geliş zamanları, talepleri, gerçek süreleri ve gerçek kullanımları.
- **Küme modeli:** Trace'teki makine tipleri ve kapasitelerinden; gerekirse küçültülmüş ama oranları korunmuş bir küme.
- **Yanlış tahminin bedeli (kritik):** Bir sunucuda işlerin gerçek kullanımı kapasiteyi aşarsa ceza uygulanır. Aday modeller:
  - Yavaşlama: aşırı yüklenen sunucudaki işlerin süresi uzar
  - Yeniden başlatma: işlerden biri durdurulup kuyruğa geri gönderilir
  - Seçilen model ve parametreleri raporda açıkça belirtilecek.
- **Doğrulama:** Simülatör, tahminsiz yaklaşımla çalıştırıldığında trace'teki genel davranışa (bekleme süresi dağılımı vb.) makul ölçüde yakın sonuç vermeli.
- **Varsayımlar** (raporda açıkça listelenecek): ağ/topoloji etkisi yok sayılır, gang scheduling basitleştirilir vb.

---

## 6. Karşılaştırılacak yaklaşımlar

| # | Yaklaşım | Tahmin kaynağı | Amaç |
|---|---|---|---|
| 0 | **Tahminsiz** | Kullanıcı talebi; sıralama geliş sırasına göre | Alt sınır, bugünkü standart davranış |
| 1 | **Klasik ML** | Gradient boosting (M1 özellikleri) | "Neden derin öğrenme?" sorusuna cevap |
| 2 | **Yalnızca M1** (DL) | MLP | Zaman serisi katkısının referansı |
| 3 | **M1 + M2, sabit birleştirme** | MLP + zaman serisi encoder'ı | Multimodalitenin katkısı |
| 4 | **M1 + M2, adaptif birleştirme** | + gating/attention | **Önerilen yöntem** |
| 5 | **Kâhin (oracle)** | Gerçek süre ve kullanım | Üst sınır: mükemmel tahminle ulaşılabilecek en iyi sonuç |

**Temel ilke:** Her adım bir öncekinden yalnızca bir şeyi değiştirir; böylece her farkın kaynağı net görülür.

---

## 7. Değerlendirme metrikleri

### 7.1 Tahmin metrikleri
- Süre: log ölçekte MAE / RMSE, MAPE (dikkatle), R²
- Kullanım: MAE, **az tahmin oranı** (tahmin < gerçek olan iş yüzdesi — zamanlama açısından en kritik hata türü)
- Koşullara göre ayrıştırılmış hata: yoğun / sakin küme dönemleri

### 7.2 Zamanlama metrikleri
| Metrik | Tanım |
|---|---|
| **Ortalama JCT** | İşin gelişinden bitişine kadar geçen süre (ortalama + medyan + 95. yüzdelik) |
| **Bekleme süresi** | Gelişten çalışmaya başlamaya kadar geçen süre |
| **GPU kullanım oranı** | Kümede gerçekten kullanılan GPU / toplam GPU (zaman ortalaması) |
| **Kaynak israfı** | Ayrılan ama kullanılmayan kaynak miktarı |
| **Taşma / ceza oranı** | Yanlış tahmin nedeniyle cezaya uğrayan iş yüzdesi |
| **Makespan** | Tüm işlerin bitmesi için geçen toplam süre |

### 7.3 Ana sonuç grafikleri (hedef)
1. **Denge eğrisi:** x = kaynak verimliliği, y = ortalama JCT; her yaklaşım için α değiştikçe bir eğri.
2. **Kâhin farkının kapatılma oranı:** (Yaklaşım − Tahminsiz) / (Kâhin − Tahminsiz).
3. **Gating ağırlıkları – küme yoğunluğu** ilişkisi.
4. **Ablation tablosu:** her modalite / bileşen çıkarıldığında sonuçlar.

### 7.4 Güvenilirlik
- Her model en az 3–5 farklı rastgele tohum (seed) ile eğitilecek; ortalama ± standart sapma raporlanacak.
- Farklar için basit istatistiksel anlamlılık kontrolü yapılacak.

---

## 8. Kapsam

### Kesin yapılacaklar
- Trace'in incelenmesi, temizlenmesi ve özellik çıkarımı
- M2 zaman serisinin üretilmesi
- 6 yaklaşımın (0–5) uygulanması
- Olay tabanlı küme simülatörü (ceza modeli dahil)
- Tahmin ve zamanlama deneyleri, ablation, denge eğrileri
- Rapor ve açık kaynak kod deposu

### Zaman kalırsa
- Graf modalitesi (M3) + GNN encoder
- Ek sezgisel baseline'lar (ör. Tiresias tarzı)
- Farklı yük senaryoları (yapay olarak yoğunlaştırılmış trace)
- Trace'in daha yeni sürümleriyle genelleme testi

### Kapsam dışı
- Gerçek bir kümede çalıştırma (yalnızca simülasyon)
- Pekiştirmeli öğrenme (RL) ile zamanlama
- Sıfırdan yeni bir zamanlama algoritması icat etmek

---

## 9. Yol haritası (aşamalar)

> Süre kısıtı belirlenmedi; aşamalar sırayla ilerler ve her aşama, çıktıları tamamlanınca kapanır.

### Aşama 0 — Kurulum
- [ ] Proje klasör yapısı
- [ ] Python ortamı (sanal ortam, temel kütüphaneler)
- [ ] Git deposu
- **Çıktı:** Çalışan boş proje iskeleti

### Aşama 1 — Veri edinme ve keşif
- [ ] Trace'in indirilmesi
- [ ] Tabloların ve alanların incelenmesi; Bölüm 3'teki beklentilerin doğrulanması
- [ ] Temel istatistikler: iş süresi dağılımı, talep vs. gerçek kullanım farkı, geliş örüntüleri
- [ ] Veri kalitesi: eksik değerler, tutarsız kayıtlar
- **Çıktı:** Veri keşif raporu (notebook + kısa özet); kesinleşmiş özellik listesi

### Aşama 2 — Veri hazırlama
- [ ] Temizleme ve iş seviyesinde birleştirme
- [ ] M1 özelliklerinin çıkarılması
- [ ] M2 zaman serisinin üretilmesi (zaman adımı ve N'nin seçimi)
- [ ] Hedef değişkenlerin oluşturulması
- [ ] Zamansal eğitim / doğrulama / test bölmesi
- **Çıktı:** Modele hazır veri setleri

### Aşama 3 — Tahmin modelleri
- [ ] Yaklaşım 1: Gradient boosting
- [ ] Yaklaşım 2: Yalnızca M1
- [ ] M2 encoder seçimi (LSTM / TCN / Transformer kısa karşılaştırma)
- [ ] Yaklaşım 3: Sabit birleştirme
- [ ] Yaklaşım 4: Adaptif birleştirme
- [ ] Tahmin metrikleriyle değerlendirme (AS1, AS2'nin tahmin tarafı)
- **Çıktı:** Eğitilmiş modeller + tahmin sonuçları tablosu

### Aşama 4 — Simülatör
- [ ] Olay tabanlı çekirdek
- [ ] Zamanlayıcı (sıralama + yerleştirme + güvenlik payı)
- [ ] Ceza modeli
- [ ] Tahminsiz ve kâhin yaklaşımlarıyla doğrulama
- **Çıktı:** Doğrulanmış simülatör

### Aşama 5 — Zamanlama deneyleri
- [ ] Tüm yaklaşımlar × α değerleri × tohumlar
- [ ] Yük koşullarına göre ayrıştırılmış analiz
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
| Trace'te beklenen alanların eksik olması | Tasarım değişir | Aşama 1'de erken doğrulama; tasarımı veriye göre güncellemek |
| M2'nin katkısının düşük çıkması | Ana hipotez zayıflar | Olumsuz sonuç da bulgudur; koşullara göre ayrıştırılmış analizle "nerede işe yarıyor" sorusu |
| Simülatörün gerçekçi olmaması | Sonuçlar inandırıcı olmaz | Tahminsiz ile doğrulama; varsayımları açıkça yazmak; ceza modeli |
| Veri sızıntısı (leakage) | Sonuçlar yapay olarak iyi çıkar | Zamansal bölme; girdilerde yalnızca geliş anında bilinen bilgiler |
| "Multimodal" teriminin sorgulanması | Akademik itiraz | Tablo + zaman serisi = farklı yapı, farklı encoder; hocayla erken görüşme |
| Hesaplama kaynağı yetersizliği | Eğitim yavaşlar | Küçük modeller; gerekirse veri alt kümesi; ücretsiz bulut GPU seçenekleri |

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

---

## 12. Hâlâ açık olan konular

- [ ] Hocadan "multimodal" çerçevesi için onay
- [ ] Zaman adımı uzunluğu ve pencere uzunluğu N (Aşama 2)
- [ ] M2 encoder türü (Aşama 3)
- [ ] Gating mi attention mı (Aşama 3)
- [ ] Ceza modelinin kesin biçimi (Aşama 4)
- [ ] Simüle edilecek küme boyutu (Aşama 4)
- [ ] Rapor dili ve formatı
