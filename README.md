<div align="center">
# Karanlıkta Güvende Hissetme Algısının Makine Öğrenmesi ile Tahmini
 
**TGSS 2024 verisiyle suç korkusu (fear of crime) sınıflandırması ve algoritmik adalet analizi**
 
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Veri](https://img.shields.io/badge/Veri-TGSS%202024-0A66C2)](https://doi.org/10.5281/zenodo.18721350)
[![Lisans](https://img.shields.io/badge/Kod-MIT-green)](LICENSE)
[![Veri Lisansı](https://img.shields.io/badge/Veri-CC%20BY--NC%204.0-lightgrey)](https://creativecommons.org/licenses/by-nc/4.0/)
 
*Türkiye Yapay Zekâ Akademisi × Huawei Student Development (HSD)*
*Yapay Zekâ ve Veri Bilimi Bootcamp — Final Projesi*
 
</div>
---
 
## İçindekiler
 
- [Özet](#özet)
- [Problem Tanımı](#problem-tanımı)
- [Veri Seti](#veri-seti)
- [Yöntem](#yöntem)
- [Sonuçlar](#sonuçlar)
- [Algoritmik Adalet Analizi](#algoritmik-adalet-analizi)
- [Kurulum ve Çalıştırma](#kurulum-ve-çalıştırma)
- [Proje Yapısı](#proje-yapısı)
- [Sınırlılıklar](#sınırlılıklar)
- [Kaynakça](#kaynakça)
---
 
## Özet
 
Bu çalışma, Türkiye genelini temsil eden **TGSS 2024** anket verisini kullanarak bir bireyin
yaşadığı mahallede **karanlıkta yalnız yürürken kendini güvende hissedip hissetmediğini**
tahmin eden sınıflandırma modelleri geliştirir.
 
Beş farklı algoritma karşılaştırılmış, en iyi model test setinde **0,742 ROC AUC** elde etmiştir.
Çalışmanın asıl katkısı ise sadece model performansı değildir: modelin **cinsiyet grupları
arasındaki hata dağılımı** incelendiğinde, accuracy farkı yalnızca %1,1 olmasına rağmen
yanlış pozitif oranları arasında **41 puanlık** büyük bir fark tespit edilmiştir.
 
> **English summary:** This project predicts perceived safety after dark (a standard *fear of
> crime* indicator) using the nationally representative Turkish General Social Survey (TGSS) 2024.
> Five classifiers are compared; the best model reaches 0.742 ROC AUC on the test set. Beyond
> predictive performance, a group-wise fairness audit reveals that while accuracy parity holds
> across gender (1.1 pp difference), equalized odds is strongly violated (41 pp gap in false
> positive rates).
 
---
 
## Problem Tanımı
 
**Problem türü:** İkili sınıflandırma (binary classification)
 
**Hedef değişken:** `nbdark` → `guvende`
 
| Değer | Anlam |
|:-----:|-------|
| `1` | Karanlıkta yalnız yürürken kendini güvende hissediyor |
| `0` | Hissetmiyor |
 
Kullanılan anket sorusu, uluslararası literatürde **suç korkusunun (fear of crime)** standart
göstergesidir ve ESS, GSS gibi büyük ölçekli araştırmalarda da yer alır. Bu sayede bulgular
uluslararası karşılaştırmaya açıktır.
 
---
 
## Veri Seti
 
**Türkiye Genel Sosyal Saha Araştırması (TGSS) 2024**
İSAR Araştırma Merkezi · DOI: [10.5281/zenodo.18721350](https://doi.org/10.5281/zenodo.18721350)
 
| Özellik | Değer |
|---|---|
| Örneklem | 2.615 kişi (18 yaş üstü, Türkiye geneli temsili) |
| Örnekleme | Olasılıklı çok aşamalı tabakalı küme örneklemesi |
| Saha dönemi | 17 Mayıs – 2 Haziran 2024 |
| Yöntem | CAPI / CASI |
| Değişken sayısı | 664 |
| Analiz örneklemi | 1.742 kişi (hedef soruyu yanıtlayanlar) |
| Sınıf dengesi | %52,1 güvende — dengeli |
| Lisans | CC BY-NC 4.0 |
 
> ⚠️ **Veri dosyası bu repoda yer almaz.** CC BY-NC 4.0 lisansı gereği veriyi doğrudan
> yeniden dağıtmak yerine kaynağına yönlendiriyoruz. 
 
### Anket verisine özgü not
 
TGSS'te eksik yanıtlar `NaN` olarak değil, **özel kodlarla** saklanır:
 
| Kod | Anlamı |
|:---:|--------|
| `-99` | Cevap vermek istemiyorum |
| `-90` | Uygun değil |
| `-88` | Bilmiyorum |
 
Bu kodlar temizlenmeden yapılacak her analiz hatalıdır. Ayrıca anket üç forma bölündüğü ve her
katılımcı bunlardan ikisini yanıtladığı için bazı sorulardaki eksiklik **tasarım gereğidir**,
veri kalitesi sorunu değildir.
 
---
 
## Yöntem
 
```
Veri yükleme → Özel kod temizliği → Hedef oluşturma → EDA
     → Öznitelik mühendisliği → Encoding → Train/Val/Test bölme
     → Ölçekleme → Öznitelik seçimi → Modelleme → Çapraz doğrulama
     → Hiperparametre optimizasyonu → Test değerlendirmesi → Adalet analizi
```
 
**Kullanılan algoritmalar:** Lojistik Regresyon · KNN · SVM (RBF) · Karar Ağacı · Random Forest
· DummyClassifier (baseline)
 
**Veri bölme:** %60 eğitim / %20 doğrulama / %20 test, `stratify` ile sınıf oranı korunarak.
Model seçimi **yalnızca doğrulama setinde** yapılmış, test seti tek seferlik nihai ölçüm için
saklanmıştır.
 
**Öznitelik mühendisliği:** Tekil anket sorularının gürültüsünü azaltmak için üç bileşik endeks
türetilmiştir:
 
| Endeks | Bileşenler |
|---|---|
| `SosyalGuvenEndeksi` | `trustpeople`, `trustfair`, `trusthelp` |
| `IliskiMemnuniyeti` | `satsoc`, `satneı`, `satfam` |
| `OznelRefah` | `health`, `happy`, `lifesat` |
 
Üç endeks de en önemli altı öznitelik arasına girmiştir.
 
---
 
## Sonuçlar
 
### Doğrulama seti
 
| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| **Random Forest** | 0,6504 | 0,6531 | 0,7033 | 0,6772 | **0,6742** |
| Lojistik Regresyon | 0,6361 | 0,6396 | 0,6923 | 0,6649 | 0,6740 |
| SVM | 0,6390 | 0,6474 | 0,6758 | 0,6613 | 0,6728 |
| Karar Ağacı | 0,6218 | 0,6168 | 0,7253 | 0,6667 | 0,6504 |
| KNN | 0,6046 | 0,6279 | 0,5934 | 0,6102 | 0,6211 |
| *Baseline* | *0,5215* | *0,5215* | *1,0000* | *0,6855* | *0,5000* |
 
> **Metrik uyarısı:** Baseline modelin F1 skoru (0,6855) tüm modellerden yüksektir; çünkü herkese
> pozitif sınıfı atadığı için recall'ü 1,000'dir. Buna karşılık ROC AUC'si 0,500'dür. Bu, tek
> metriğe bakmanın neden yanıltıcı olduğunun somut bir örneğidir.
 
**Çapraz doğrulama (5 katlı, ROC AUC):** Random Forest **0,7137 ± 0,0202**
 
### Test seti (nihai performans)
 
| Metrik | Değer |
|---|:---:|
| ROC AUC | **0,7419** |
| Accuracy | 0,7192 |
| Precision | 0,7188 |
| Recall | 0,7582 |
| F1 | 0,7380 |
 
### Öne çıkan bulgular
 
| Bulgu | Değer |
|---|---|
| Cinsiyet farkı | **25,7 puan** (Erkek %64,8 — Kadın %39,1) |
| En kırılgan grup | 18-24 yaş kadınlar: **%28,0** |
| Yaş örüntüsü | 18-24 yaş %41,8 → 65+ yaş %74,8 |
| Komşuya güven etkisi | Güvenenler %71,5 — güvenmeyenler %40,8 |
| En yüksek bölge | Doğu Karadeniz %71,2 |
| En düşük bölge | İstanbul %44,0 |
 
---
 
## Algoritmik Adalet Analizi
 
Projenin ana katkısı bu bölümdedir. Model, test setinde cinsiyet gruplarına ayrılarak
yeniden değerlendirilmiştir:
 
| Metrik | Erkek (n=178) | Kadın (n=171) | Fark |
|---|:---:|:---:|:---:|
| Gerçek güvende oranı | 0,6798 | 0,3567 | — |
| Modelin tahmin ettiği oran | 0,7865 | 0,3041 | — |
| Accuracy | 0,7247 | 0,7135 | **0,0113** |
| Yanlış pozitif oranı (FPR) | **0,5965** | 0,1818 | **0,4147** |
| Yanlış negatif oranı (FNR) | 0,1240 | **0,4754** | **0,3514** |
 
**Yorum:** Accuracy açısından **eşitlik (accuracy parity)** sağlanmış görünmektedir; fark yalnızca
1,1 puandır. Ancak **equalized odds** ölçütü ciddi biçimde ihlal edilmiştir. Model, "erkekse
güvendedir, kadınsa değildir" örüntüsünü öğrenmiş ve bunu bireylere uygulamaktadır. Sonuç olarak
gerçekten tedirgin olan erkekler görünmez olmakta, kendini güvende hisseden kadınların ise
neredeyse yarısı yanlış sınıflandırılmaktadır.
 
Bu model bir kamu politikası kararında (örneğin sokak aydınlatması önceliklendirmesi)
kullanılsaydı, sistematik bir yanlılık üretecek ve bu yanlılık raporda görünen **%72 accuracy**
değerinin arkasında gizli kalacaktı.
 
**Ölçüm yanlılığı ihtimali:** Literatürde erkeklerin suç korkusunu olduğundan düşük bildirdiğine
dair bulgular vardır (*male discounting of fear*; Sutton & Farrall, 2005). Bu durumda modelin
"hatası" sandığımız şey, etiketin kendisindeki yanlılığın bir kopyası olabilir. Bu veriyle
test edilemez, ancak yorumda göz ardı edilmemelidir.
 
---
 
## Kurulum ve Çalıştırma
 
### Gereksinimler
 
```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
 
pip install -r requirements.txt
```
 
### Veri setinin indirilmesi
1. https://www.tgss.org.tr/ sayfasına gidip veri setini bulun
2. `TGSS_2024_dataset_V01.0.sav` dosyasını (2,1 MB) indirin
3. Proje kök dizinine kopyalayın
> Script klasördeki `.sav` uzantılı dosyayı otomatik bulur; dosya adını değiştirmeniz gerekmez.
 
### Çalıştırma
 
```bash
python guvenlik_algisi_tahmini.py
```
 
**Çalışma süresi:** ~2-3 dakika
**Üretilen çıktılar:** 8 grafik (PNG), `model_sonuclari.csv`, `adalet_analizi.csv`
 
Doğrulama için çıktının sonunda şu değerleri görmelisiniz:
 
```
Test ROC AUC      : 0.7419
Test Accuracy     : 0.7192
Cinsiyet farkı    : 25.7 yüzde puanı
```
 
---
 
## Proje Yapısı
 
```
.
├── guvenlik_algisi_tahmini.py    # Ana analiz scripti (22 adım)
├── requirements.txt
├── model_sonuclari.csv           # Model karşılaştırma çıktısı
├── adalet_analizi.csv            # Grup bazlı adalet metrikleri
├── grafikler/                    # Üretilen 8 görsel
│   ├── grafik_01_cinsiyet_yas.png
│   ├── grafik_02_bolgeler.png
│   ├── grafik_03_sosyal_guven.png
│   ├── grafik_04_model_karsilastirma.png
│   ├── grafik_05_confusion_roc.png
│   ├── grafik_06_feature_importance.png
│   ├── grafik_07_katsayilar.png
│   └── grafik_08_algoritmik_adalet.png
├── .gitignore
└── README.md
```
 
---
 
## Sınırlılıklar
 
- **Anket ağırlıkları kullanılmamıştır.** TGSS'in örneklem ağırlıkları uygulanmadığı için sonuçlar
  Türkiye nüfusuna doğrudan genellenemez.
- **Kesitsel veri.** Bulgular nedensellik değil, **ilişki** ifade eder.
- **Etiket yanlılığı.** Hedef değişken bir öz-bildirimdir; sosyal beğenilirlik etkisi ölçülememiştir.
- **Örneklem büyüklüğü.** Test seti 349 kişidir; metrikler dalgalanmaya açıktır. Test ROC AUC'nin
  doğrulamadan yüksek çıkması bu dalgalanmayla açıklanabilir; gerçek performans ~0,70–0,74 bandındadır.
- **Kapsam dışı değişken.** Dışarıda ayrımcılığa maruz kalma (`cntrace`) değişkeni, yanıt oranı
  düşük olduğu için (%50) modele dâhil edilmemiştir.
---
 
## Kaynakça
 
**Veri seti**
 
> Nişancı, Z., Kılavuz, M. T., Aysan, M. F., Ovayurt, M. E. T., Yüce, B., Aydın, A. B., Kavdır, A. Y.,
> Alboğa, M. H., Akbulut, Y., Atik, E. ve Aydın, R. N. (2026).
> *Türkiye Genel Sosyal Saha Araştırması (TGSS) 2024: Dataset.*
> https://doi.org/10.5281/zenodo.18721350 — CC BY-NC 4.0
 
**Literatür**
 
- Warr, M. (1984). *Fear of Victimization: Why Are Women and the Elderly More Afraid?*
  Social Science Quarterly, 65.
- Smith, W. R. & Torstensson, M. (1997). *Gender Differences in Risk Perception and Neutralizing
  Fear of Crime.* The British Journal of Criminology, 37(4).
- Sutton, R. M. & Farrall, S. (2005). *Gender, Socially Desirable Responding and the Fear of Crime.*
  British Journal of Criminology, 45(2).

  
## Lisans
 
**Kod:** MIT Lisansı
**Veri:** TGSS 2024 verisi CC BY-NC 4.0 lisanslıdır ve bu repoda dağıtılmamaktadır. Veriyi
kullanırken yukarıdaki atfı yapmanız gerekmektedir.
 
---
 
<div align="center">
📝 Projenin ayrıntılı anlatımı için **[Medium yazısına]()** göz atabilirsiniz.
 
</div>
