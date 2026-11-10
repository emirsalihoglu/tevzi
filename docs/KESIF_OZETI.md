# Tevzi — Keşif Özeti (Aşama 1 ve 2)

**Tarih:** 7 Kasım 2026 (Aşama 1) — güncelleme 8 Kasım 2026 (Aşama 2 bulguları, Bölüm 11)
**Veri:** Alibaba `cluster-trace-gpu-v2020` (Weng ve ark., *MLaaS in the Wild*, NSDI 2022)
**Notebook'lar:** `notebooks/02_exploration.ipynb` (Aşama 1), `notebooks/03_data_preparation.ipynb` (Aşama 2)
**İlgili belge:** `docs/PROJE_PLANI.md` (v1.3) — bu özetten çıkan kararlar planın 11. bölümündeki karar günlüğüne işlendi.

Bu belge, veriyi modellemeden önce neyi doğruladığımızı, neyi beklemediğimizi ve bunların tasarıma nasıl yansıdığını özetler. Tüm sayılar notebook çıktılarından alınmıştır. **8.11 düzeltmeleri** ilgili yerlerde işaretlendi.

---

## 1. Tablolar ve hiyerarşi

| Tablo | Satır | İçerik |
|---|---|---|
| job | 1.055.501 | İş: kullanıcı, durum, gönderilme ve bitiş zamanı |
| task | 1.261.050 | Görev: instance sayısı, kaynak talepleri, istenen GPU tipi |
| instance | 7.522.002 | Instance denemesi: worker, makine, zamanlar |
| sensor | 3.033.232 | Deneme başına kullanım özeti (CPU, GPU, bellek, GPU belleği, I/O) |
| group_tag | 1.055.032 | İş etiketi: grup, workload (**anahtar `inst_id`**, `job_name` yok) |
| machine_spec | 1.897 | Makine kapasiteleri |
| machine_metric | 2.009.423 | Worker başına makine ölçümleri (Bölüm 11.3) |

- **Anahtarlar tutarlı.** `job_name` ve `inst_id` her işte tekil ve bire bir eşleşiyor. Sensor satırları denemelere `worker_name` ile bağlanıyor.
- **İşlerin çoğu tek görevli.** İş başına 1–4 görev, ortalama 1,19. İki yapı: tek görevli işler (`tensorflow`, `PyTorchWorker`…) ve parameter-server yapısındaki dağıtık işler (`worker` + `ps` [+ `evaluator`]).
- **Yeniden denemeler yaygın.** 231.632 instance birden fazla kez kayıtlı; tekrarlanan satırların %78,4'ü Failed.
- **Instance'ı olmayan küçük bir grup var:** 130 görev (69 iş).

## 2. Durumlar ve zaman bilgisinin güvenilirliği

| İş durumu | Pay | Başlangıç eksik | Bitiş eksik |
|---|---|---|---|
| Terminated | %69,4 | %0 | %0 |
| Failed | %24,3 | %0 | %90,0 |
| Running | %6,0 | %0 | %98,3 |
| Waiting | %0,3 | %100 | %98,8 |

- **Güvenilir süre yalnızca Terminated işlerde var** (732.355 iş).
- **8.11 eki:** Failed işlerin bitişi instance ve `machine_metric` tablolarıyla da kurtarılamıyor (herhangi bir kaynaktan bitiş: %15,7). **Running** işler gerçekte trace sonunda çalışmıyor: gönderilme günlerinin medyanı 36,7, son 2 günde gönderilenler yalnız %1,4 → **durumu güncellenmemiş kayıtlar** (Bölüm 11.1).

## 3. Süre dağılımı (Terminated işler)

> **8.11 düzeltmesi:** Aşağıdaki ilk tablo `end − submit`, yani **bekleme süresi dahil** hesaplanmıştı. Plandaki hedef **çalışma süresi** (`end − ilk instance başlangıcı`); onun dağılımı ikinci tabloda.

**Bekleme dahil (`end − submit`):**

| Yüzdelik | %1 | %5 | %25 | %50 | %75 | %95 | %99 | %99,9 |
|---|---|---|---|---|---|---|---|---|
| Süre (s) | 17 | 28 | 144 | 663 | 3.871 | 23.922 | 68.884 | 205.781 |

**Çalışma süresi (`end − first_start`):**

| Küme | %1 | %25 | %50 | %75 | %90 | %99 |
|---|---|---|---|---|---|---|
| Tüm Terminated | 11 | 104 | 551 | 3.571 | — | 66.153 |
| Model kümesi (714.273) | 12 | 103 | **534** | 3.550 | 11.902 | 65.199 |

- **Log dönüşümü şart.** Süreler dört kattan fazla büyüklük farkına yayılıyor.
- **Dağılım çok tepeli.** Kısa ve uzun işlerin bir arada bulunması, en kısa iş önce sıralamasının kazanç potansiyelini gösteriyor.

## 4. Kaynak talepleri ve kapasiteler

| Alan | Birim | Kanıt |
|---|---|---|
| `plan_cpu` | Çekirdek yüzdesi (600 = 6 çekirdek) | 100'ün katlarında yoğunlaşma |
| `plan_mem` | GB | 29,296875 = 30000/1024 |
| `plan_gpu` | GPU yüzdesi (100 = 1 GPU) | En büyük değer 800 |

- **Kesirli GPU talebi baskın** (medyan 50) → simülatörde GPU bölünebilir.
- **Talepler instance başına.** GPU talebi 1,04 milyon görevin 169'unda kapasiteyi aşıyor.
- **Yalnız CPU görevleri:** `gpu_type = 'None'` (string) olan 217.738 görev. **8.11 eki:** bunların büyük çoğunluğu GPU işlerinin içindeki `ps` (172.335), `evaluator`, `chief` görevleri; **tamamen CPU olan iş** yalnız 31.514 (Terminated 17.452).

| Makine tipi | Adet | Çekirdek | Bellek (GB) | GPU | Havuz GPU |
|---|---|---|---|---|---|
| CPU | 83 | 96 | 512 | 0 | — |
| MISC | 280 | 96 | 512 | 8 | 2.240 |
| P100 | 798 | 64 | 512 | 2 | 1.596 |
| T4 | 497 | 96 | 512 | 2 | 994 |
| V100 | 104 | 96 | 384 / 512 | 8 | 832 |
| V100M32 | 135 | 96 | 384 | 8 | 1.080 |

Toplam GPU 6.742 (kaynak makaleyle aynı).

## 5. Gerçek kullanım (sensor)

- **Kapsam:** Denemelerin %40'ı, işlerin %79'u. **8.11 eki: sensor kaydı gün 18'de başlıyor** (gün 0–16 %0, gün 17 %1, gün 18 %15, gün 19'dan itibaren %94–99). Sensor çöken (Failed) işleri de ölçmüş (GPU görevlerinde %84,5); yalnız CPU görevlerinde kayıt yok (%0).
- **Birimler:** `cpu_usage` ve `gpu_wrk_util` taleplerle aynı yüzde biriminde; bellek GB.

| Kaynak | Kullanım / talep, medyan (deneme başına) | %75 | Talebi aşan denemeler |
|---|---|---|---|
| CPU | 0,23 | 0,63 | %15,4 |
| Tepe bellek | 0,20 | 0,61 | %12,0 |
| GPU | **0,04** | 0,23 | %2,1 |

**8.11 eki — iş başına hedefler** (ana görevin tüm denemeleri arasında maksimum, model kümesi): GPU oranı medyan 0,050, **%34,3'ü tam 0** (GPU hiç kullanılmamış), %3,5 > 1; tepe bellek oranı medyan 0,169, %99 3,08, **%10,6 > 1**. Bellek medyanının deneme başına değerden (0,20) düşük olması, deneme ağırlıklı ölçümde çok denemeli/çöken işlerin baskın olmasıyla açıklanabilir.

## 6. Zaman ekseni, geliş örüntüsü ve bekleme

- **Trace 68,9 gün.** Gün sayıları ilk gönderilmeden itibaren (t₀ = 494.319 s).
- **Bekleme süresi** (ilk instance başlangıcı − gönderilme): medyan 7 s, %90 109–110 s, %99 ~5.700 s.
- **Geliş grafiği:** gün 0–7 neredeyse boş; gün 7–58 saatte ~500–1.000 iş, gün 32–34 orta yoğunluk; **gün 59–60 patlama** (99.195 iş, %82,5 tek kullanıcı, %79,5 Failed).
- **8.11 düzeltmesi — döngüler:** Aşama 1'de "belirgin gece/gündüz farkı yok" demiştik. 5 dk'lık ızgarada özilinti bunu düzeltiyor: talep bazlı GPU doluluğu 12 saatte 0,73, **24 saatte 0,84**; gelişler 12 saatte 0,07, **24 saatte 0,37** → **günlük döngü var.** Saatlik geliş grafiği bunu görünür kılmıyordu.
- **8.11 eki — bekleme heterojen:** istenen GPU tipine göre %90'lık bekleme T4 4 s, MISC 109 s, P100 927 s, **V100 5.214 s, V100M32 7.602 s** (Bölüm 11.4).

## 7. Kullanıcı ve grup sinyali

`group_tag` işlerin %99,86'sını kapsıyor: 1.431 kullanıcı, 316.956 grup.

| Değişken | Kategori sayısı | Açıklanan varyans | Tekil kategori payı |
|---|---|---|---|
| group | 217.981 | 0,880 | %25,3 |
| user | 1.307 | 0,343 | %0 |
| workload | 9 | 0,308 (dolu %12'de) | %0 |
| task_name | 17 | 0,041 | %0 |

**8.11 eki:** Sızıntısız tarihçe özellikleriyle (yalnız gelişten önce bitmiş işler) eğitim döneminde, grup tarihçesi olan işlerde grubun son iki biten işinin ortalaması ile log-süre arasında Spearman ρ = **0,932**; kullanıcı için 0,486. İşlerin ~%28'i grup tarihçesi olmadan geliyor.

## 8. Veri kalitesi notları

| Bulgu | Büyüklük | İşlem |
|---|---|---|
| Yalnız CPU işleri | 31.514 iş (görevlerin ~%17'si) | Tahmin/zamanlama kapsamı dışı; M2'de CPU/bellek kanallarında |
| Eksik talepler | ~%1,5 | Dışlanır (R3; model kümesinde 0 — hepsi zaten Terminated değil) |
| GPU talebi kapasiteyi aşan | 169 görev | Dışlanır (R5: 83 iş) |
| `plan_cpu = 0` | 10–11 görev | Dışlanır (R4) |
| Kapasiteyi aşan gözlenen kullanım | 2.075 iş (bayrak) | Dışlanır (R7: 540 iş) |
| Gönderilmeden önce başlama / süre ≤ 0 | Çok az | Dışlanır (R6: 7 iş) |
| Instance'ta karşılığı olmayan sensor satırları | 13.883 | Yok sayılır |
| `group_tag` anahtarı `inst_id` | — | İş tablosu üzerinden bağlanır |
| `machine_metric` eksikleri | `machine_gpu` %12, `machine_num_worker` %14,7 | Doğrulamada temizlenir |
| Dağıtık işlerde ilk instance PS (CPU makinesi) olabilir | — | İşin havuzu: instance'larının en sık GPU makine tipi |

## 9. Tasarıma yansıyan sonuçlar (Aşama 1)

1. Analiz birimi iş; yerleştirme instance seviyesinde.
2. Süre hedefi çalışma süresi, log; bekleme referans.
3. Eğitim ve simülasyon yalnız Terminated işlerle.
4. İki kullanım hedefi: GPU kullanım oranı ve tepe bellek oranı.
5. Zamansal bölme (8.11: eğitim 19. günde başlıyor); purging; patlama ayrık.
6. Metrikler birincil/ikincil önceden ayrıldı.
7. Dışlama akışı tablosu (uygulandı, plan Bölüm 3).

## 10. Aşama 1'den açık kalanlar → Aşama 2'de kapatıldı

| Açık konu | Sonuç |
|---|---|
| M2'nin hangi işlerden hesaplanacağı | Kanal bazlı karma (plan 4.2) |
| `machine_metric` incelemesi | Doğrulama referansı; gün 24'te başlıyor |
| Yalnız CPU görevlerinin sensor kaydı | Yok (%0) |
| M2 zaman adımı ve pencere | 5 dk; 2 saat + gönderilme anının tam durumu |

## 11. Aşama 2 bulguları (8.11.2026)

### 11.1 Failed ve Running işlerin zamanları
- **Failed (256.555):** iş bitişi %10,0, instance bitişi %11,8, herhangi bir bitiş %15,7 (`machine_metric` katkısı yok). Bitişi bilinenlerde iş bitişi = son instance bitişi (fark medyanı 0). Bitişi bilinenlerin çalışma süresi medyanı 111–183 s (alt örnek yanlı olabilir); %1'lik dilimde negatif süre.
- **Running (62.928):** bitiş %12,4; gönderilme günü medyanı 36,7 (%10: 13,7; %90: 60,7); son 2 günde gönderilen %1,4 → gerçekte çalışmıyor.
- **Waiting (3.663):** gönderilme zamanı yok.

### 11.2 Dışlama akışı ve analiz kümeleri
1.055.501 → R1 −323.146 → 732.355 → R2 −17.452 → 714.903 → R3 0 → R4 0 → R5 −83 → R6 −7 → R7 −540 → **714.273**. Sıradan bağımsız bayraklar: R2 31.514, R3 16.911, R4 10, R5 169, R6 296.813 (çoğu bitişsiz Failed/Running), R7 2.075. Kümeler: `m2_arrival` 1.051.838, `m2_occupancy` 732.257 (GPU 714.813 + CPU 17.444), `model` 714.273.

### 11.3 Ölçüm sistemlerinin başlangıçları
- **Sensor:** gün 18 (gün 19'dan itibaren %94–99).
- **`machine_metric`:** gün 24 (gün 19–23 günlük ortalama GPU kullanımı 0,00–0,01, gün 24'ten itibaren 0,09–0,15). Satırlar yalnız sensor'lu worker'lar çalışırken tutulmuş; Terminated kapsamı %78,5, Failed %0,9. `machine_gpu` makinedeki GPU yüzdelerinin toplamı (0–787).

### 11.4 Küme durumu (M2 keşfi, eğitim dönemi)
- **Küme geneli, talep bazlı:** GPU doluluğu medyan 0,45 / p99 0,62; CPU 0,27 / 0,39; bellek 0,17 / 0,26 → küme genelinde doyma yok. Süre artıkları, bekleme ve kullanım artıklarıyla ilişki |ρ| ≤ 0,04; doluluk dilimlerine göre medyan bekleme 6–8 s (düz); eşik etkisi yok.
- **Havuz bazlı:** havuz doluluğu medyan / p99 — P100 0,41 / 0,70, T4 0,36 / 0,81, V100 0,26 / 0,64, V100M32 0,27 / 0,62. **Havuz baskısı** (kuyruktaki GPU talebi / havuzdaki boş GPU) ile bekleme, gönderilme anının tam durumunda: V100 ρ = **0,372**, V100M32 **0,380**, P100 **0,159**, T4 0,089, MISC (tüm havuzlar) 0,061. Havuzlar birleştirildiğinde doluluk–bekleme ilişkisi −0,19'a dönüyor (**Simpson paradoksu**: T4 en dolu ama en hızlı başlayan havuz).
- **Süre üzerindeki etki:** havuz durumu süre artıklarıyla büyük ölçüde ilişkisiz (V100M32 baskısı +0,10 dışında |ρ| ≤ 0,03) → tıkanma beklemeyi belirliyor, çalışma süresini çok az etkiliyor.
- **Kullanım bazlı GPU doluluğu** (ana görev tepe kullanımı × instance): medyan 0,17 (talep bazlı 0,45). `machine_metric` ile (gün ≥ 24, 5 dk) Spearman 0,63; talep bazlı doluluk 0,75.
- **Failed yükünün etkisi (duyarlılık):** Failed işler eklenince GPU doluluğu medyan 0,45 → 0,54 (Spearman 0,93), baskı 0,041 → 0,123 (Spearman 0,86).

### 11.5 Tasarıma yansıyan sonuçlar (Aşama 2)
1. M2: gelişler tüm işlerden, doluluk/kuyruk Terminated işlerden; simülasyonda arka plan yükü.
2. Eğitim dönemi gün 19'da başlıyor (sensor).
3. M2 havuz bazlı; baskı kanalları; MISC → tüm havuzlar.
4. M2'nin süre/kullanım tahminine katkısının küçük ve havuza bağlı olması bekleniyor → Aşama 3'te havuza göre ayrıştırılmış değerlendirme.
5. GPU oranı sıfır-şişkin → Aşama 3'te hurdle yapısı değerlendirilecek.
