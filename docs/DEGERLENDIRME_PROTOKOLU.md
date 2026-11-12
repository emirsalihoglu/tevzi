# Tevzi — Aşama 3 Değerlendirme Protokolü

**Tarih:** 12 Kasım 2026 — **hiçbir Aşama 3 modeli eğitilmeden önce yazıldı** (git commit zamanı kayıt olarak geçerlidir).
**Kod:** `src/tevzi/eval/metrics.py` (7 birim testi: `tests/test_metrics.py`)
**İlgili:** `docs/PROJE_PLANI.md` Bölüm 7 (birincil/ikincil metrikler), `docs/LITERATUR_NOTLARI.md` Bölüm 7.

İstisna: M3 (grup tarihçesi dizisi) denemesi bu protokolden önce, kendi önceden yazılmış ölçütüyle yapıldı ve kapandı (LITERATUR 7.2).

---

## 1. Veri kullanımı
- Model eğitimi: eğitim dönemi (gün 19–45), purge edilmemiş işler.
- Model seçimi, hiperparametre ve tüm Aşama 3 kararları: **yalnız doğrulama dönemi** (gün 45–52).
- **Test dönemi (gün 52+) Aşama 3 boyunca kapalı.** Yalnız Aşama 3 sonunda, seçilmiş son modellerle bir kez değerlendirilir.
- Eşikler ve ölçekleyiciler yalnız eğitim döneminden hesaplanır.

## 2. Metrikler
| Hedef | Metrik | Rol |
|---|---|---|
| Süre (log1p s) | **SJF pişmanlığı** (`sjf_regret`) | **Model seçimi** |
| | log-MAE, aynı saatte gelen işlerde Kendall τ | Birincil, raporlanır (plan 7.1) |
| | RMSE, R², Spearman ρ, `sjf_gap_closed` | İkincil |
| GPU oranı, bellek oranı | MAE + az tahmin oranı | Model seçimi ve birincil |
| | Ortalama eksik tahmin, sıfır / pozitif ayrımında MAE | İkincil (hurdle değerlendirmesi) |

**SJF pişmanlığı:** İşler gönderildikleri saate göre gruplanır. Her grupta tahmin edilen süreye göre (en kısa önce; eşitlikte geliş sırası) tek sunucuda art arda çalıştırılır. Toplam tamamlanma süresi, gerçek sürelerle yapılan sıralamanın (kâhin) toplamına bölünür. 1,00 mükemmel; küçük daha iyi. `sjf_gap_closed` = geliş sırası (FIFO) ile kâhin arasındaki farkın kapatılan oranı.

## 3. Model seçim kuralı (karar g)
1. Süre modelleri doğrulamada **SJF pişmanlığına** göre seçilir.
2. Kullanım modelleri doğrulamada **MAE** ile seçilir; az tahmin oranı birlikte raporlanır. MAE farkı gürültü içindeyse az tahmin oranı düşük olan seçilir.
3. Aşama 4–5'te her yaklaşımın son adayları (≤ 3) doğrulama döneminde **simüle JCT** ile yeniden karşılaştırılır. Vekil ölçüt ile simüle JCT farklı sıra verirse **simüle JCT kazanır** ve tutarsızlık raporlanır.
4. Bilinen sınırlılık: doğrulamada > 1 günlük işlerin ~%33'ü purge edildiği için vekil ölçüt uzun işleri eksik görür → `rt_ge_1d` alt kümesi ayrıca raporlanır; 3. madde bunu telafi eder.

## 4. Tohumlar ve karşılaştırma kuralı
- Her model **5 tohum** (0–4).
- İki model eşleştirilmiş karşılaştırılır (aynı tohum = bir çift; `paired_comparison`). Gürültü = **eşleştirilmiş farkın** tohumlar arası yayılımı.
- B, A'dan **daha iyi** sayılır ancak: (i) ortalama eşleştirilmiş iyileşmenin %95 t-aralığı sıfırı içermiyorsa **ve** (ii) göreli kazanç ≥ **%1** ise. (%1: M3 denemesinde bilginin olamayacağı alt kümede bile ~%0,3–1,1 fark görüldü.)
- Raporlarda ortalama ± std ve güven aralığı birlikte verilir.

## 5. Değerlendirme alt kümeleri (`make_subsets`)
`all`, `no_burst` / `burst` (gün 59–60, yalnız testte), `hist_none` / `hist_some` (grup tarihçesi), `busy` / `calm` (gönderilme anında işin havuzundaki `pool_pressure` ≥ eğitim döneminin %75'lik değeri), `rt_ge_1h`, `rt_ge_1d`, `full_coverage` (`usage_coverage == 1`), havuz başına (`pool_ALL`, `pool_P100`, `pool_T4`, `pool_V100`, `pool_V100M32`), `pool_V100_busy` (V100 veya V100M32 ve busy).

## 6. Önceden adlandırılmış hipotezler ("iyi sonuç beklenen yerler")
Sonuçlar görülmeden yazıldı. Her biri doğrulamada test edilir, testte bir kez raporlanır; sonuç ne çıkarsa raporlanır.

| # | Hipotez | Başarı ölçütü |
|---|---|---|
| **H1** | Kullanım tahmini, "kullanım = talep" varsayımından (oran = 1) çok daha iyi | GBM'in GPU ve bellek oranı MAE'si, sabit 1 tahmininin MAE'sinin yarısından küçük |
| **H2** | Tahminli sıralama geliş sırasından belirgin şekilde iyi | Seçilen süre modelinin `sjf_gap_closed` ≥ 0,5 |
| **H3** | M2'nin katkısı koşullu: kalabalık üst seviye havuzlarda daha büyük | Hem 1b vs 1 (GBM) hem 3 vs 2 (DL) için: `pool_V100_busy` alt kümesinde §4 kuralına göre anlamlı iyileşme **ve** göreli kazanç `all`'dakinden büyük. Süre (SJF pişmanlığı) ve kullanım (MAE) ayrı ayrı raporlanır |
| **H4** | Adaptif birleştirme M2'ye kalabalıkta daha fazla ağırlık veriyor | Doğrulamada M2 gating ağırlığı ile `pool_pressure` arasında Spearman ρ > 0 ve %95 güven aralığı sıfırı içermiyor |

Ek (Aşama 4–5): M2'nin değerinin küme sıkışıklığıyla artıp artmadığı, simülatörde küçültülmüş kümelerle ayrıca test edilecek (ölçüt o aşamada sonuçlardan önce yazılacak).
