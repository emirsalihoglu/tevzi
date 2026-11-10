# Tevzi — Literatür Notları

**Başlangıç:** 8 Kasım 2026 (Aşama 2) — son güncelleme 8.11 (Aşama 2 kapanışı)
**Amaç:** Benzer çalışmaların veri hazırlama, hedef tanımı, bölme, özellik ve değerlendirme yöntemlerini kaydetmek; kararlarımızı bunlarla karşılaştırmak. Rapordaki "İlgili çalışmalar" ve yöntem gerekçeleri için kaynak.

---

## 1. Doğrudan aynı trace'i kullanan çalışmalar

### ATLAS (Jiang, Zhao, Xia, Zomaya — ICLR 2026)
*ATLAS: Alibaba Dataset and Benchmark for Learning-Augmented Scheduling.* Kod/veri: github.com/zhiyunjiang0810/non-clairvoyant-with-predictions
- **Veri:** Alibaba PAI-2020 GPU trace'inden ~730 bin **tamamlanmış** iş (bizim Terminated sayımız 732.355 → uyumlu).
- **Etiket:** fork-join süresi; p = E − S. → Bizim tanımımızla aynı fikir.
- **Dışarıda bırakılan sütunlar:** gerçek atanan GPU tipi, instance kimlikleri, sensor/kullanım, ağ, makine yerleşimi.
- **Özellikler:** gönderilme anı: log talepler, sin/cos saat, kullanıcı, grup etiketleri, instance sayıları; tarihçe `shift(1)` ile; Empirical Bayes shrinkage (λ=5).
  - **Bizim farkımız:** tarihçe yalnızca **gelişten önce bitmiş** işlerden (as-of, kesin "önce"), birim testli → daha katı.
  - **Ayrıca:** ATLAS küme durumu (M2) kullanmıyor.
- **Bölme:** kronolojik 70/15/15. **Metrikler:** RMSLE, Coverage@τ, Spearman ρ; en iyi ρ = 0,912.
- **Sınırlılıklar:** etiketler orijinal zamanlayıcıdan ve CPU çekişmesinden etkileniyor; simülasyonlarda **kesirli GPU paylaşımı yok.**

### "Better Predictions, Worse Schedules" (NeurIPS 2026)
- Tahmin edicinin doğruluğa göre seçilmesi yanlış ölçüt; uzun işler domine eder. Alibaba/Google'da %18/%26 fark.
- **Bizde ölçüldü:** doğrulamada >1 gün süren işlerin ~%33'ü purge ediliyor (eğitimde %5,6).

### MLaaS in the Wild (Weng ve ark. — NSDI 2022) — trace'in kaynak makalesi
- Görevlerin çoğu gang-scheduled ve tekrarlı; grup etiketi özelleştirilmiş girdinin hash'i; %65 ≥ 5 tekrar.
- **CPU darboğaz olabiliyor** → M2 CPU kanalları.
- **Seçici gereksinimli, üst seviye GPU isteyen görevler zor zamanlanıyor** → **bizde doğrulandı:** %90'lık bekleme T4 4 s, V100 5.214 s, V100M32 7.602 s.

## 2. Benzer GPU kümesi tahmin + zamanlama çalışmaları
- **Helios / QSSF (SC 2021):** tahmin tabanlı QSSF (JCT 6,5×'e kadar); Cluster Energy Saving.
- **Lucid (ASPLOS 2023):** GA²M; baseline LightGBM/XGBoost/RF/DNN; politika baseline'ları FIFO, SJF, QSSF, Tiresias.
- **SchedMate (2025):** seyrek meta veri sınırlılığı → M2 motivasyonu.

## 3. Veri temizliği pratikleri (karşılaştırma)
| Çalışma | Temizlik |
|---|---|
| ATLAS | Tamamlanmış işler; yürütme sonrası sütunlar çıkarılır |
| Eva (2025, Alibaba v2023) | Failed ve hiçbir instance tipine sığmayan işler çıkarılır → R1, R5 |
| AIOps model seçimi (2025) | Bitmemiş ve <5 dk işler çıkarılır (farklı görev) |
| DCcluster-Opt (2025) | <15 dk görevler çıkarılır (farklı amaç) |

## 4. Tarihçe tabanlı süre tahmini (HPC literatürü)
- **Tsafrir, Etsion, Feitelson — IEEE TPDS 2007 (EASY++):** kullanıcının son iki (gönderilmiş ve zaten bitmiş) işinin ortalaması; yakın geçmiş, benzer işleri aramaktan önemli. → `*_rt_last2`, medyan yerine `last`. **Bizde doğrulandı:** `grp_rt_last2` ρ = 0,932 > `last` 0,928 > `mean` 0,926.
- **Gaussier ve ark. — SC 2015:** ML + probleme uygun maliyet fonksiyonları → Aşama 3 kayıp kararı.
- **CMU-PDL-19-103:** kullanıcı davranışı en tahmin edici ortak özellik; mantıksal iş adı güçlü; saat/gün.
- **MOD 2017:** zamansal yerellik. **OSTI 1889001 / 2017:** az tahmin zamanlamayı bozar.

## 5. Küme durumu (M2)
### Kuyruk/sistem durumu anlık görüntüsü
- **Brown, Gibb, Belikov, Nash (arXiv 2204.13543):** gönderilme anındaki kuyruk/çalışan iş durumu özellik olarak; katkı genelde pozitif ama **sınırlı**. "Kalan iş" özellikleri gerçek sürelerle hesaplanmış → **sızıntı**; bizde kullanılmadı. **Bizim bulgumuzla tutarlı:** küme durumu tahmine küçük katkı.
- **Hutchison ve ark. (2022):** ayrılmış CPU, ayrılmış bellek, kuyruk derinliği.
- **Anvil/TROUT (2025):** istenen kaynaklar, kuyruk doluluğu, sistem kullanımı; hiyerarşik model.
- **Farkımız:** zaman serisi penceresi + gönderilme anının tam durumu; sızıntısız; **havuz bazlı**; modelden önce, artık ve dilim analizleriyle hangi bilginin taşındığını ölçtük.

### Çekişme
- **Horus (TPDS):** GPU paylaşımında ortalama %18, en fazla 2,4–3,4× yavaşlama → simülatör cezası referansı.
- **Multi-agent graph RL (arXiv 2112.13354):** PCIe ve CPU kullanımı çekişmenin en önemli göstergeleri.
- **DeepShare (arXiv 2609.16682):** baskı = bekleyen GPU talebi / boş GPU → **bizde `pressure` ve `pool_pressure`**; havuz bazında beklemeyi açıklıyor (V100 ρ = 0,37, V100M32 0,38).
- **Inference interference (2512.18725):** statik görüntü sonraki değişimleri görmez → sınırlılığımız.

### Bizim M2 bulgularımız (kıyas için)
- Küme geneli durum süre/bekleme/kullanımı açıklamıyor; havuzlara ayrıştırınca bekleme ilişkisi ortaya çıkıyor (birleşik veride Simpson paradoksu: −0,19).
- Havuz durumu çalışma süresini çok az etkiliyor → M2'nin süre/kullanım tahminine katkısının küçük olması bekleniyor (Brown ve ark. ile tutarlı).

## 6. Yapılacaklar
- [ ] Aşama 3: kategorik kodlama — Empirical Bayes shrinkage (ATLAS λ=5); yalnız eğitimde fit.
- [ ] Aşama 3: GBM sonucunu ATLAS'ın ρ = 0,912 ile karşılaştır (dikkatle).
- [ ] Aşama 3: Tsafrir "son iki iş" ek baseline'ı — karar olarak sor.
- [ ] Aşama 3: model seçim ölçütü (NeurIPS 2026) ve asimetrik kayıp (Gaussier 2015).
- [x] Adım 7: türetilmiş çekişme kanalı (DeepShare) → `pressure`, `pool_pressure`.
- [x] Adım 7: "kalan iş" gibi gerçek süre gerektiren kanallardan kaçınıldı.
- [ ] Aşama 4: yavaşlama cezası için Horus referansı.
- [ ] Aşama 5–6: uzun işler ayrıca; purging yanlılığı; M2'nin "çalışma boyunca yük" görmemesi.
- [ ] İlgili çalışmalar: ATLAS'a göre farklarımız — M2 (havuz bazlı zaman serisi), kullanım tahmini, GPU paylaşımlı simülasyon, birim testli tarihçe.
