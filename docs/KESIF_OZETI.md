# Tevzi — Aşama 1 Keşif Özeti

**Tarih:** 7 Kasım 2026
**Veri:** Alibaba `cluster-trace-gpu-v2020` (Weng ve ark., *MLaaS in the Wild*, NSDI 2022)
**Notebook:** `notebooks/02_exploration.ipynb`
**İlgili belge:** `docs/PROJE_PLANI.md` (v1.2) — bu özetten çıkan kararlar planın 11. bölümündeki karar günlüğüne işlendi.

Bu belge, veriyi modellemeye başlamadan önce neyi doğruladığımızı, neyi beklemediğimizi ve bunların tasarıma nasıl yansıdığını özetler. Tüm sayılar notebook çıktılarından alınmıştır.

---

## 1. Tablolar ve hiyerarşi

| Tablo | Satır | İçerik |
|---|---|---|
| job | 1.055.501 | İş: kullanıcı, durum, gönderilme ve bitiş zamanı |
| task | 1.261.050 | Görev: instance sayısı, kaynak talepleri, istenen GPU tipi |
| instance | 7.522.002 | Instance denemesi: worker, makine, zamanlar |
| sensor | 3.033.232 | Deneme başına kullanım özeti (CPU, GPU, bellek, GPU belleği, I/O) |
| group_tag | 1.055.032 | İş etiketi: grup, workload |
| machine_spec | 1.897 | Makine kapasiteleri |
| machine_metric | 2.009.423 | Makine seviyesinde ölçümler (Aşama 2'de incelenecek) |

- **Anahtarlar tutarlı.** `job_name` ve `inst_id` her işte tekil ve bire bir eşleşiyor. Her görev bilinen bir işe, her instance bilinen bir göreve ait. Sensor satırları instance denemelerine `worker_name` ile bağlanıyor (`worker_name` deneme başına tekil).
- **İşlerin çoğu tek görevli.** İş başına 1–4 görev düşüyor, ortalaması 1,19. Görev adlarına göre iki yapı var: tek görevli işler (`tensorflow`, `PyTorchWorker`…) ve parameter-server yapısındaki dağıtık işler (`worker` + `ps` [+ `evaluator`]). Toplam 17 farklı görev adı var.
- **Yeniden denemeler yaygın.** 231.632 instance birden fazla kez kayıtlı (toplam 906.751 satır). Deneme sayısının medyanı 2, en fazlası 338. Denemelerin %99,8'i farklı bir worker'da başlatılmış. Tekrarlanan satırların %78,4'ü Failed (tekil satırlarda bu oran %3,9). Tipik desen: deneme çöküyor, birkaç dakika içinde başka bir worker'da yeniden başlatılıyor.
- **Instance'ı olmayan küçük bir grup var:** 130 görev (69 iş). Bunlar hiç başlamamış.

## 2. Durumlar ve zaman bilgisinin güvenilirliği

| İş durumu | Pay | Başlangıç eksik | Bitiş eksik |
|---|---|---|---|
| Terminated | %69,4 | %0 | %0 |
| Failed | %24,3 | %0 | %90,0 |
| Running | %6,0 | %0 | %98,3 |
| Waiting | %0,3 | %100 | %98,8 |

- **Güvenilir süre yalnızca Terminated işlerde var** (732.355 iş). Failed işlerin bitişi büyük çoğunlukla kayıtlı değil. Running işler trace sonunda hâlâ çalışıyor, yani sağdan sansürlü.
- **Instance seviyesinde zamanlar dağınık.** Running instance'ların %85'inde, Interrupted instance'ların ise tamamında başlangıç zamanı yok. Bu yüzden süre iş seviyesinden hesaplanmalı.
- **Küçük tutarsızlıklar:** Running işlerin %1,7'sinde ve Running görevlerin %22'sinde bitiş zamanı dolu.

## 3. Süre dağılımı (Terminated işler)

| Yüzdelik | %1 | %5 | %25 | %50 | %75 | %95 | %99 | %99,9 |
|---|---|---|---|---|---|---|---|---|
| Süre (s) | 17 | 28 | 144 | 663 | 3.871 | 23.922 | 68.884 | 205.781 |

- Ortalama 5.382 s (~90 dk), medyan 663 s (~11 dk), en uzun 7,25 gün. Negatif ya da sıfır süreli kayıt yok.
- **Log dönüşümü şart.** Süreler dört kattan fazla büyüklük farkına yayılıyor, ortalama medyanın 8 katı.
- **Dağılım çok tepeli.** 20–60 saniyede sivri çıkıntılar var (tekrarlanan veya şablonla gönderilen işler), ~3–5 dakikada bir tepe, 10 dakikadan 2 saate geniş bir plato ve ~15 saat civarında bir tümsek. Kısa ve uzun işlerin bir arada bulunması, en kısa iş önce sıralamasının kazanç potansiyelini gösteriyor.

## 4. Kaynak talepleri ve kapasiteler

| Alan | Birim | Kanıt |
|---|---|---|
| `plan_cpu` | Çekirdek yüzdesi (600 = 6 çekirdek) | 100'ün katlarında yoğunlaşma; en büyük değer 9000 ≤ 96 çekirdek |
| `plan_mem` | GB | 29,296875 = 30000/1024 (kullanıcı MB istemiş); en büyük değer 400 ≤ 512 GB |
| `plan_gpu` | GPU yüzdesi (100 = 1 GPU) | En büyük değer 800 = 8 GPU |

- **Kesirli GPU talebi baskın.** `plan_gpu`'nun medyanı 50. En yaygın değerler 100, 25, 50, 10 ve 5. Görevlerin çoğu bir GPU'yu paylaşıyor, bu yüzden simülatörde GPU bölünebilir bir kaynak olarak modellenmeli.
- **Talepler instance başına tanımlı.** İstenen tipin en büyük makinesine göre kontrol ettik: CPU ve bellek talebi hiçbir görevde kapasiteyi aşmıyor, GPU talebi ise 1,04 milyon görevin yalnızca 169'unda aşıyor.
- **Yalnız CPU görevleri:** `gpu_type = None` olan 217.738 görevin 217.737'sinde `plan_gpu` boş.
- **İstenen GPU tipleri:** MISC %55, T4 227 bin, None 218 bin, P100 73 bin, V100 29 bin, V100M32 19 bin görev.

| Makine tipi | Adet | Çekirdek | Bellek (GB) | GPU |
|---|---|---|---|---|
| CPU | 83 | 96 | 512 | 0 |
| MISC | 280 | 96 | 512 | 8 |
| P100 | 798 | 64 | 512 | 2 |
| T4 | 497 | 96 | 512 | 2 |
| V100 | 104 | 96 | 384 / 512 | 8 |
| V100M32 | 135 | 96 | 384 | 8 |

## 5. Gerçek kullanım (sensor)

- **Kapsam:** Denemelerin %40'ında, işlerin ise %79'unda kullanım verisi var. Terminated işlerde bu oran %78,6 (575.912 iş). Kapsanan Terminated işlerin yarısından fazlasında denemelerin tamamı, dörtte üçünde en az %84'ü kayıtlı.
- **Birimler:** `cpu_usage` ve `gpu_wrk_util` taleplerle aynı yüzde biriminde (en büyük değerler 9.206 ve 792). Bellek sütunları GB cinsinden. `max_mem` için 2.757 GB gibi fiziksel olarak imkânsız aykırı değerler var.

| Kaynak | Kullanım / talep, medyan | %75'lik dilim | Talebi aşan denemeler |
|---|---|---|---|
| CPU | 0,23 | 0,63 | %15,4 |
| Tepe bellek | 0,20 | 0,61 | %12,0 |
| GPU | **0,04** | 0,23 | %2,1 |

**Yorum:** Kullanıcılar sistematik olarak fazla kaynak istiyor. Medyan deneme, istediği GPU'nun yalnızca %4'ünü kullanıyor. Kullanımı doğru tahmin edebilen bir zamanlayıcı, aynı donanıma çok daha fazla iş sığdırabilir. Projenin verimlilik hipotezinin dayanağı bu. Öte yandan talebi aşan denemelerin oranı (CPU %15, bellek %12), talebin altında tahmin yapmanın gerçek bir risk taşıdığını ve güvenlik payının (α) neden gerekli olduğunu gösteriyor.

## 6. Zaman ekseni, geliş örüntüsü ve bekleme

- **Trace 68,9 gün sürüyor.** Saat göreli. Bu belgede gün sayıları ilk işin gönderildiği andan itibaren veriliyor (t₀ = 494.319 s).
- **job `start_time` gönderilme anı.** İlk instance her zaman ondan sonra başlıyor. Aradaki fark bekleme süresi: medyan 7 s, %90'lık dilim 109 s, %99'luk dilim 5.622 s, en uzun 6,4 gün. Çok az sayıda negatif kayıt var. Küme çoğunlukla rahat, ara ara tıkanıyor. Bu yüzden zamanlama kararlarının etkisi yoğun dönemlerde yoğunlaşacak.
- **Geliş grafiği:**
  - **Gün 0–7:** Neredeyse boş. Trace bu dönemde tam kayıt tutmuyor (ısınma dönemi).
  - **Gün 7–58:** Saatte ~500–1.000 iş. Gün 32–34'te orta büyüklükte bir yoğunluk var (~2.500/saat).
  - **Gün 59–60:** Saatte 4.000, ardından 7.500 işlik patlama. Patlama penceresindeki 99.195 işin %82,5'i tek bir kullanıcıdan geliyor ve %79,5'i Failed. Otomatik bir döngünün sürekli çöken işleri yeniden göndermesi gibi görünüyor.
- **Döngüler zayıf.** Belirgin bir gece/gündüz farkı yok, göreli saat 16'da keskin bir tepe var. Belirgin bir hafta sonu çukuru da yok.

## 7. Kullanıcı ve grup sinyali

`group_tag` işlerin %99,86'sını kapsıyor: 1.431 kullanıcı, 316.956 grup. `workload` 9 değer alıyor (bert, ctr, nmt…) ve %90 boş. `gpu_type_spec` %98 boş.

Kategori bilindiğinde log-sürenin ne kadarının açıklandığına baktık (Terminated işler, örneklem içi, iyimser bir üst sınır):

| Değişken | Kategori sayısı | Açıklanan varyans | Tekil kategori payı |
|---|---|---|---|
| group | 217.981 | 0,880 | %25,3 |
| user | 1.307 | 0,343 | %0 |
| workload | 9 | 0,308 (yalnızca %12'lik dolu kısımda) | %0 |
| task_name | 17 | 0,041 | %0 |

**Yorum:** Aynı grup etiketini taşıyan işler büyük ölçüde aynı işin tekrarları ve benzer sürüyor. Ancak grupların dörtte biri yalnızca bir kez görülüyor, yani ham grup kimliği yeni gruplara genellenemez. Bu bulgu M1 tasarımında **tarihçe özelliklerine** yol açtı: grubun ve kullanıcının geçmişte biten işlerinin istatistikleri (bkz. plan Bölüm 4.1).

## 8. Veri kalitesi notları

| Bulgu | Büyüklük | İşlem (plan Bölüm 3) |
|---|---|---|
| Yalnız CPU görevleri | Görevlerin ~%17'si | Kapsam dışı |
| Eksik talepler | CPU/bellek %1,5; GPU tipi belli ama `plan_gpu` boş ~6.200 görev | Dışlanır |
| GPU talebi kapasiteyi aşan görevler | 169 görev | Dışlanır |
| `plan_cpu = 0` | 11 görev | Dışlanır |
| Kapasiteyi aşan gözlenen kullanım | Az (ör. `max_mem` 2.757 GB) | Dışlanır |
| İlk instance gönderilmeden önce başlamış | Çok az (en küçük değer −15.479 s) | Dışlanır |
| Instance'ta karşılığı olmayan sensor satırları | 13.883 satır (%0,5) | Yok sayılır |
| Running/Waiting kayıtlarında zaman tutarsızlıkları | Küçük | Bu işler zaten kapsam dışı |

## 9. Tasarıma yansıyan sonuçlar

1. **Analiz birimi iş.** Görev yapısı M1'e özellik olarak giriyor, yerleştirme instance seviyesinde yapılıyor.
2. **Süre hedefi çalışma süresi:** bitiş − ilk instance başlangıcı, log ölçekte. Yeniden denemeler sürenin içinde kalıyor. Bekleme süresi hedef değil, simülatörü doğrulamak için referans.
3. **Eğitim ve simülasyon yalnızca Terminated işlerle yapılıyor.** Simülasyona ayrıca kullanım verisi olan işler giriyor.
4. **İki kullanım hedefi var:** GPU kullanım oranı ve tepe bellek oranı (talebe göre). İkisi de ana görevin instance'ları arasındaki maksimumdan alınıyor.
5. **Zamansal bölme:** ısınma 0–7. gün, eğitim 7–45, doğrulama 45–52, test 52–69. Sınırda bitmemiş işler dışarıda bırakılıyor (purging). Patlama testte kalıyor ama ayrıca raporlanıyor.
6. **Metrikler baştan birincil ve ikincil olarak ayrıldı.** Sıralama metrikleri eklendi.
7. **Dışlama kuralları Aşama 2'de bir "dışlama akışı" tablosuyla sayısal olarak raporlanacak.**

## 10. Açık bırakılanlar (Aşama 2)

- M2'nin hangi işlerden hesaplanacağı (tüm trace mi, simülasyonla tutarlı iş kümesi mi)
- `machine_metric` tablosunun incelenmesi (kullanım bazlı doluluk kanalı için)
- Yalnız CPU görevlerinin sensor kaydı taşıyıp taşımadığının teyidi
- M2 için zaman adımı ve pencere uzunluğu (N)
