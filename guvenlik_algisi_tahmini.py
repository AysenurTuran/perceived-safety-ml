"""
=============================================================================
TÜRKİYE YAPAY ZEKA AKADEMİSİ x HUAWEI STUDENT DEVELOPMENT (HSD)
Yapay Zeka ve Veri Bilimi Bootcamp - Final Projesi
=============================================================================

Proje: Türkiye'de Karanlıkta Güvende Hissetme Algısının Makine Öğrenmesi ile Tahmini

Amaç:
    - TGSS 2024 (Türkiye Genel Sosyal Saha Araştırması) verisini kullanarak bir kişinin
      yaşadığı mahallede karanlıkta yalnız yürürken kendini güvende hissedip
      hissetmediğini tahmin eden sınıflandırma modelleri geliştirmek
    - Lojistik regresyon, KNN, SVM, karar ağacı ve rastgele orman modellerini karşılaştırmak
    - Güvenlik algısını sürükleyen demografik ve sosyal faktörleri ortaya çıkarmak
    - Modelin cinsiyet grupları arasındaki başarım farkını (algoritmik adalet) incelemek
       Bu model ayrıca hem mimarlık alanında kent içi aydınlatma kararlarının verilmesinde
       hem de güvenlik kaygısına dair farklı disiplinlerde çalışmalar yapılmasına katkı sağlayabilir.

Veri seti:
    - TGSS 2024: Turkish General Social Survey (Türkiye Genel Sosyal Saha Araştırması)
    - İSAR Araştırma Merkezi, DOI: 10.5281/zenodo.18721350
    - Lisans: CC BY-NC 4.0
    - 2615 katılımcı, 664 değişken (Türkiye geneli temsili örneklem, 18+ yaş)
    - Saha çalışması: 17 Mayıs - 2 Haziran 2024, CAPI/CASI yöntemi
    - Bu projede kullanılan alt örneklem: nbdark sorusunu yanıtlayan 1742 kişi
    - hedef değişken (nbdark -> guvende)
        - 0: kendini güvende hissetmiyor (çok güvensiz / güvensiz / ne güvenli ne güvensiz)
        - 1: kendini güvende hissediyor (güvenli / çok güvenli)

Plan/program:
    BÖLÜM 1 - VERİ HAZIRLAMA
    1. Gerekli kütüphanelerin içeriye aktarılması
    2. Veri setinin yüklenmesi ve ilk incelenmesi
    3. Özel kodların (bilmiyorum, cevap yok) eksik değere dönüştürülmesi
    4. Hedef değişkenin oluşturulması ve sınıf dengesinin incelenmesi

    BÖLÜM 2 - KEŞİFSEL VERİ ANALİZİ (EDA)
    5. Cinsiyete göre güvenlik algısının incelenmesi
    6. Yaş gruplarına göre güvenlik algısının incelenmesi
    7. Bölgelere göre güvenlik algısının incelenmesi
    8. Sosyal güven değişkenlerinin incelenmesi

    BÖLÜM 3 - ÖZNİTELİK MÜHENDİSLİĞİ VE ÖN İŞLEME
    9. Öznitelik setinin belirlenmesi ve eksik verilerin doldurulması
    10. Bileşik endekslerin türetilmesi (sosyal güven, ilişki memnuniyeti, refah)
    11. One-hot encoding uygulanması
    12. Eğitim, doğrulama ve test veri setlerinin oluşturulması
    13. Standardization uygulanması
    14. Lasso ile öznitelik seçimi

    BÖLÜM 4 - MODELLEME VE DEĞERLENDİRME
    15. Baseline (temel) model kurulması
    16. Beş sınıflandırma modelinin doğrulama setinde karşılaştırılması
    17. Çapraz doğrulama ile model kararlılığının test edilmesi
    18. GridSearchCV ile hiperparametre optimizasyonu
    19. Seçilen modelin test veri setinde nihai değerlendirilmesi

    BÖLÜM 5 - YORUMLAMA VE ALGORİTMİK ADALET
    20. Öznitelik öneminin incelenmesi
    21. Lojistik regresyon katsayıları ile etkinin yönünün yorumlanması
    22. Modelin cinsiyet gruplarındaki başarım farkının incelenmesi

Kurulumlar
pip install pandas numpy scikit-learn matplotlib seaborn pyreadstat
pip install -r requirements.txt
=============================================================================
"""

# =============================================================================
# BÖLÜM 1 - VERİ HAZIRLAMA
# =============================================================================

# 1. Gerekli kütüphanelerin içeriye aktarılması
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyreadstat

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (9, 5)
plt.rcParams["figure.dpi"] = 110

RASTGELE_TOHUM = 42

print("=" * 70)
print("TÜRKİYE'DE GÜVENLİK ALGISININ MAKİNE ÖĞRENMESİ İLE TAHMİNİ")
print("TGSS 2024 verisi ile sınıflandırma projesi")
print("=" * 70)

# 2. Veri setinin yüklenmesi ve ilk incelenmesi
df, meta = pyreadstat.read_sav("TGSS_2024_dataset_V01_0.sav")

print("\n[1] VERİ SETİNİN İLK İNCELEMESİ")
print("-" * 70)
print(f"Ham veri seti boyutu: {df.shape[0]} katılımcı, {df.shape[1]} değişken")

HEDEF_SORU = "nbdark"
print(f"\nHedef soru ({HEDEF_SORU}):")
print(f"  {meta.column_names_to_labels[HEDEF_SORU]}")

print("\nYanıt dağılımı:")
etiketler = meta.variable_value_labels[HEDEF_SORU]
for deger, sayi in df[HEDEF_SORU].value_counts().sort_index().items():
    print(f"  {int(deger)} - {etiketler[deger]:28s}: {sayi:4d}")

print(f"\nSoruyu yanıtlamayan katılımcı sayısı: {df[HEDEF_SORU].isnull().sum()}")
print("Not: TGSS'te her katılımcı üç formdan yalnızca ikisini yanıtladığı için")
print("     bazı sorularda sistematik eksiklik bulunur. Bu bir veri hatası değildir.")

# 3. Özel kodların eksik değere dönüştürülmesi
# TGSS'te -99 (cevap vermek istemiyorum), -90 (uygun değil), -88 (bilmiyorum)
OZEL_KODLAR = [-99, -90, -88]

# hedef soruyu yanıtlayanlarla çalışacağız
df_calisma = df[df[HEDEF_SORU].notna()].copy()
df_calisma = df_calisma.replace(OZEL_KODLAR, np.nan)

print(f"\nÇalışma örneklemi: {len(df_calisma)} katılımcı")

# 4. Hedef değişkenin oluşturulması ve sınıf dengesinin incelenmesi
# 4 (Güvenli) ve 5 (Çok güvenli) -> 1, diğerleri -> 0
df_calisma["guvende"] = (df_calisma[HEDEF_SORU] >= 4).astype(int)

guvende_orani = df_calisma["guvende"].mean()

print("\n[2] HEDEF DEĞİŞKEN")
print("-" * 70)
print(f"Kendini güvende hissedenler   (1): {(df_calisma['guvende'] == 1).sum():4d}")
print(f"Kendini güvende hissetmeyenler (0): {(df_calisma['guvende'] == 0).sum():4d}")
print(f"Güvende hissetme oranı: {guvende_orani:.4f} ({guvende_orani * 100:.2f}%)")
print("\nYorum: Sınıflar dengeli. Bu, accuracy metriğinin bu projede")
print("       anlamlı biçimde kullanılabileceği anlamına gelir.")


# =============================================================================
# BÖLÜM 2 - KEŞİFSEL VERİ ANALİZİ (EDA)
# =============================================================================

print("\n[3] KEŞİFSEL VERİ ANALİZİ")
print("-" * 70)


def etiketle(sutun):
    """Bir değişkenin sayısal kodlarını okunabilir etiketlere çevirir."""

    eslesme = meta.variable_value_labels.get(sutun, {})
    return df_calisma[sutun].map(eslesme)


# 5. Cinsiyete göre güvenlik algısının incelenmesi
df_calisma["cinsiyet"] = etiketle("gender")

cinsiyet_ozet = df_calisma.groupby("cinsiyet")["guvende"].agg(["mean", "count"])
cinsiyet_ozet.columns = ["guvende_orani", "kisi_sayisi"]

print("Cinsiyete göre güvende hissetme oranı:")
print(cinsiyet_ozet.round(4))

cinsiyet_farki = cinsiyet_ozet["guvende_orani"].max() - cinsiyet_ozet["guvende_orani"].min()
print(f"\nCinsiyet farkı: {cinsiyet_farki * 100:.1f} yüzde puanı")

# 6. Yaş gruplarına göre güvenlik algısının incelenmesi
df_calisma["yas_grubu"] = etiketle("agegroup")

yas_ozet = df_calisma.groupby("yas_grubu")["guvende"].agg(["mean", "count"])
yas_ozet.columns = ["guvende_orani", "kisi_sayisi"]

print("\nYaş grubuna göre güvende hissetme oranı:")
print(yas_ozet.round(4))

# cinsiyet ve yaş grubu birlikte
capraz = df_calisma.pivot_table(
    index="yas_grubu", columns="cinsiyet", values="guvende", aggfunc="mean"
)

print("\nYaş grubu x Cinsiyet kırılımı:")
print(capraz.round(3))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.barplot(x=cinsiyet_ozet.index, y=cinsiyet_ozet["guvende_orani"],
            hue=cinsiyet_ozet.index, palette="Set2", legend=False, ax=axes[0])
axes[0].axhline(guvende_orani, color="red", linestyle="--", linewidth=1.2,
                label=f"Genel ortalama ({guvende_orani:.1%})")
axes[0].set_title("Cinsiyete Göre Güvende Hissetme Oranı")
axes[0].set_xlabel("Cinsiyet")
axes[0].set_ylabel("Güvende hissetme oranı")
axes[0].set_ylim(0, 0.8)
axes[0].legend(fontsize=8)

for i, deger in enumerate(cinsiyet_ozet["guvende_orani"]):
    axes[0].text(i, deger, f"{deger:.1%}", ha="center", va="bottom", fontsize=10)

capraz.plot(kind="bar", ax=axes[1], color=["#66c2a5", "#fc8d62"], edgecolor="black", linewidth=0.4)
axes[1].axhline(guvende_orani, color="red", linestyle="--", linewidth=1.2)
axes[1].set_title("Yaş Grubu ve Cinsiyete Göre Güvende Hissetme")
axes[1].set_xlabel("Yaş grubu")
axes[1].set_ylabel("Güvende hissetme oranı")
axes[1].tick_params(axis="x", rotation=30)
axes[1].legend(title="Cinsiyet", fontsize=8)

plt.tight_layout()
plt.savefig("grafik_01_cinsiyet_yas.png", bbox_inches="tight")
plt.close()

# 7. Bölgelere göre güvenlik algısının incelenmesi
df_calisma["bolge"] = etiketle("nuts1")

bolge_ozet = df_calisma.groupby("bolge")["guvende"].agg(["mean", "count"])
bolge_ozet.columns = ["guvende_orani", "kisi_sayisi"]
bolge_ozet = bolge_ozet.sort_values("guvende_orani", ascending=False)

print("\nBölgelere göre güvende hissetme oranı:")
print(bolge_ozet.round(4))

plt.figure(figsize=(11, 6))
sns.barplot(y=bolge_ozet.index, x=bolge_ozet["guvende_orani"],
            hue=bolge_ozet.index, palette="viridis", legend=False)
plt.axvline(guvende_orani, color="red", linestyle="--", linewidth=1.2,
            label=f"Türkiye ortalaması ({guvende_orani:.1%})")
plt.title("NUTS-1 Bölgelerine Göre Güvende Hissetme Oranı")
plt.xlabel("Güvende hissetme oranı")
plt.ylabel("Bölge")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("grafik_02_bolgeler.png", bbox_inches="tight")
plt.close()

# 8. Sosyal güven değişkenlerinin incelenmesi
df_calisma["komsuya_guven"] = etiketle("trustneig")

komsu_ozet = df_calisma.groupby("trustneig")["guvende"].agg(["mean", "count"])
komsu_ozet.columns = ["guvende_orani", "kisi_sayisi"]

print("\nKomşuya güven düzeyine göre güvende hissetme oranı:")
komsu_etiket = meta.variable_value_labels.get("trustneig", {})
for deger, satir in komsu_ozet.iterrows():
    print(f"  {komsu_etiket.get(deger, deger):25s}: "
          f"{satir['guvende_orani']:.3f} (n={int(satir['kisi_sayisi'])})")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

komsu_sirali = komsu_ozet.copy()
komsu_sirali.index = [komsu_etiket.get(i, i) for i in komsu_sirali.index]

sns.barplot(x=komsu_sirali.index, y=komsu_sirali["guvende_orani"],
            hue=komsu_sirali.index, palette="Set2", legend=False, ax=axes[0])
axes[0].axhline(guvende_orani, color="red", linestyle="--", linewidth=1.2)
axes[0].set_title("Komşuya Güven ve Güvende Hissetme")
axes[0].set_xlabel("Komşuya güven düzeyi")
axes[0].set_ylabel("Güvende hissetme oranı")
axes[0].tick_params(axis="x", rotation=25)

# genel insan güveni (0-10 ölçeği)
guven_ozet = df_calisma.groupby("trustpeople")["guvende"].mean()

axes[1].plot(guven_ozet.index, guven_ozet.values, marker="o", linewidth=2, color="#4c72b0")
axes[1].axhline(guvende_orani, color="red", linestyle="--", linewidth=1.2)
axes[1].set_title("Genel İnsan Güveni ve Güvende Hissetme")
axes[1].set_xlabel("İnsanlara güven (0 = çok dikkatli, 10 = çok güvenir)")
axes[1].set_ylabel("Güvende hissetme oranı")

plt.tight_layout()
plt.savefig("grafik_03_sosyal_guven.png", bbox_inches="tight")
plt.close()


# =============================================================================
# BÖLÜM 3 - ÖZNİTELİK MÜHENDİSLİĞİ VE ÖN İŞLEME
# =============================================================================

print("\n[4] ÖZNİTELİK MÜHENDİSLİĞİ")
print("-" * 70)

# 9. Öznitelik setinin belirlenmesi ve eksik verilerin doldurulması
SAYISAL_OZNITELIKLER = [
    "age",           # yaş
    "hhsize",        # hane büyüklüğü
    "trustpeople",   # insanlara güven (0-10)
    "trustfair",     # insanlar kandırır mı (0-10)
    "trusthelp",     # insanlar yardımsever mi (0-10)
    "trustneig",     # komşuya güven (1-5)
    "health",        # sağlık durumu (1-5)
    "happy",         # mutluluk (1-5)
    "lifesat",       # yaşam memnuniyeti (1-5)
    "inchhrel",      # göreli hane geliri algısı (1-5)
    "satsoc",        # sosyal hayat memnuniyeti (1-5)
    "satneı",        # komşu ilişkileri memnuniyeti (1-5)
    "satfam",        # aile ilişkileri memnuniyeti (1-5)
    "religious",     # dindarlık düzeyi (1-5)
    "educlt",        # eğitim seviyesi
]

KATEGORIK_OZNITELIKLER = [
    "gender",     # cinsiyet
    "nuts1",      # bölge
    "marital",    # medeni durum
    "area14",     # 14 yaşına kadar yaşanan yerleşim türü
    "areasame",   # hâlâ aynı yerde mi yaşıyor
    "enrolled",   # öğrenci mi
]

X_ham = df_calisma[SAYISAL_OZNITELIKLER + KATEGORIK_OZNITELIKLER].copy()
y = df_calisma["guvende"]

print(f"Başlangıç öznitelik sayısı: {X_ham.shape[1]}")
print(f"  Sayısal  : {len(SAYISAL_OZNITELIKLER)}")
print(f"  Kategorik: {len(KATEGORIK_OZNITELIKLER)}")

eksik_ozet = X_ham.isnull().sum()
eksik_ozet = eksik_ozet[eksik_ozet > 0].sort_values(ascending=False)

print("\nEksik değer içeren öznitelikler:")
print(eksik_ozet)

# sayısal değişkenler medyan ile doldurulur (aykırı değerlerden etkilenmez)
for sutun in SAYISAL_OZNITELIKLER:
    if X_ham[sutun].isnull().any():
        X_ham[sutun] = X_ham[sutun].fillna(X_ham[sutun].median())

# kategorik değişkenler mod ile doldurulur
for sutun in KATEGORIK_OZNITELIKLER:
    if X_ham[sutun].isnull().any():
        X_ham[sutun] = X_ham[sutun].fillna(X_ham[sutun].mode()[0])

print(f"\nDoldurma sonrası kalan eksik değer: {X_ham.isnull().sum().sum()}")

# 10. Bileşik endekslerin türetilmesi
# tek tek sorular gürültülü olabilir; aynı olguyu ölçen soruların ortalaması
# daha kararlı bir sinyal verir (ölçek güvenilirliği mantığı)

# sosyal güven endeksi: 0-10 ölçeğindeki üç güven sorusunun ortalaması
X_ham["SosyalGuvenEndeksi"] = X_ham[["trustpeople", "trustfair", "trusthelp"]].mean(axis=1)

# ilişki memnuniyeti endeksi: sosyal, komşu ve aile memnuniyeti ortalaması
X_ham["IliskiMemnuniyeti"] = X_ham[["satsoc", "satneı", "satfam"]].mean(axis=1)

# öznel refah endeksi: sağlık, mutluluk ve yaşam memnuniyeti ortalaması
X_ham["OznelRefah"] = X_ham[["health", "happy", "lifesat"]].mean(axis=1)

YENI_OZNITELIKLER = ["SosyalGuvenEndeksi", "IliskiMemnuniyeti", "OznelRefah"]

print("\nTüretilen bileşik endeksler:")
for oznitelik in YENI_OZNITELIKLER:
    korelasyon = X_ham[oznitelik].corr(y)
    print(f"  {oznitelik:20s}: hedef ile korelasyon = {korelasyon:+.4f}")

print("\nBileşenlerin tekil korelasyonları (karşılaştırma için):")
for sutun in ["trustpeople", "trustfair", "trusthelp", "satneı", "health"]:
    korelasyon = X_ham[sutun].corr(y)
    print(f"  {sutun:20s}: hedef ile korelasyon = {korelasyon:+.4f}")

# 11. One-hot encoding uygulanması
# kategorik kodları önce okunabilir etiketlere çevirelim ki sütun adları anlamlı olsun
for sutun in KATEGORIK_OZNITELIKLER:
    eslesme = meta.variable_value_labels.get(sutun, {})
    X_ham[sutun] = X_ham[sutun].map(eslesme).fillna("Bilinmiyor").astype(str)

X = pd.get_dummies(X_ham, columns=KATEGORIK_OZNITELIKLER, drop_first=True)

print(f"\nOne-hot encoding sonrası öznitelik sayısı: {X.shape[1]}")

# 12. Eğitim, doğrulama ve test veri setlerinin oluşturulması
X_gecici, X_test, y_gecici, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RASTGELE_TOHUM, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_gecici, y_gecici, test_size=0.25, random_state=RASTGELE_TOHUM, stratify=y_gecici
)

print(f"\nEğitim seti   : {X_train.shape[0]} kişi (%{X_train.shape[0] / len(X) * 100:.0f}), "
      f"hedef oranı: {y_train.mean():.4f}")
print(f"Doğrulama seti: {X_val.shape[0]} kişi (%{X_val.shape[0] / len(X) * 100:.0f}), "
      f"hedef oranı: {y_val.mean():.4f}")
print(f"Test seti     : {X_test.shape[0]} kişi (%{X_test.shape[0] / len(X) * 100:.0f}), "
      f"hedef oranı: {y_test.mean():.4f}")

# 13. Standardization uygulanması
olceklenecek = SAYISAL_OZNITELIKLER + YENI_OZNITELIKLER

scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_val_scaled = X_val.copy()
X_test_scaled = X_test.copy()

# scaler SADECE eğitim verisi ile fit edilir (veri sızıntısını önlemek için)
X_train_scaled[olceklenecek] = scaler.fit_transform(X_train[olceklenecek])
X_val_scaled[olceklenecek] = scaler.transform(X_val[olceklenecek])
X_test_scaled[olceklenecek] = scaler.transform(X_test[olceklenecek])

print("\nÖlçekleme tamamlandı (fit sadece eğitim verisinde yapıldı).")

# 14. Lasso ile öznitelik seçimi
print("\n[4.1] ÖZNİTELİK SEÇİMİ (Lasso / L1)")
print("-" * 70)

lasso_secici = LogisticRegression(
    penalty="l1", solver="liblinear", C=0.1, random_state=RASTGELE_TOHUM
)
lasso_secici.fit(X_train_scaled, y_train)

lasso_katsayilari = pd.Series(lasso_secici.coef_[0], index=X_train_scaled.columns)

elenen = lasso_katsayilari[lasso_katsayilari == 0].index.tolist()
secilen = lasso_katsayilari[lasso_katsayilari != 0].index.tolist()

print(f"Elenen öznitelik sayısı : {len(elenen)}")
print(f"Seçilen öznitelik sayısı: {len(secilen)}")

if elenen:
    print("\nElenen öznitelikler:")
    for oznitelik in elenen:
        print(f"  - {oznitelik}")

# seçimin etkisini doğrulama setinde ölçelim
test_modeli = LogisticRegression(C=1, max_iter=1000, random_state=RASTGELE_TOHUM)

test_modeli.fit(X_train_scaled, y_train)
tum_f1 = f1_score(y_val, test_modeli.predict(X_val_scaled))

test_modeli.fit(X_train_scaled[secilen], y_train)
secilmis_f1 = f1_score(y_val, test_modeli.predict(X_val_scaled[secilen]))

print(f"\nDoğrulama F1 (tüm {X_train_scaled.shape[1]} öznitelik)   : {tum_f1:.4f}")
print(f"Doğrulama F1 (seçilen {len(secilen)} öznitelik): {secilmis_f1:.4f}")


# =============================================================================
# BÖLÜM 4 - MODELLEME VE DEĞERLENDİRME
# =============================================================================

print("\n[5] MODELLEME")
print("-" * 70)


def model_degerlendir(model_adi, y_gercek, y_tahmin, y_olasilik):
    """Sınıflandırma metriklerini hesaplar ve sözlük olarak döndürür."""

    return {
        "Model": model_adi,
        "Accuracy": accuracy_score(y_gercek, y_tahmin),
        "Precision": precision_score(y_gercek, y_tahmin),
        "Recall": recall_score(y_gercek, y_tahmin),
        "F1": f1_score(y_gercek, y_tahmin),
        "ROC AUC": roc_auc_score(y_gercek, y_olasilik),
    }


# 15. Baseline (temel) model kurulması
dummy_clf = DummyClassifier(strategy="most_frequent", random_state=RASTGELE_TOHUM)
dummy_clf.fit(X_train_scaled, y_train)

dummy_pred = dummy_clf.predict(X_val_scaled)
dummy_proba = dummy_clf.predict_proba(X_val_scaled)[:, 1]

dummy_sonuc = model_degerlendir("Baseline (çoğunluk)", y_val, dummy_pred, dummy_proba)

print("BASELINE MODEL (herkese çoğunluk sınıfını atayan model):")
print(f"  Accuracy: {dummy_sonuc['Accuracy']:.4f}")
print(f"  ROC AUC : {dummy_sonuc['ROC AUC']:.4f}")
print("\nYorum: Sınıflar dengeli olduğu için baseline accuracy ~%52'de kalıyor.")
print("       Bu, modellerimizin gerçekten öğrenip öğrenmediğini net biçimde gösterecek.")

# 16. Beş sınıflandırma modelinin doğrulama setinde karşılaştırılması
modeller = {
    "Lojistik Regresyon": LogisticRegression(
        C=1, max_iter=1000, random_state=RASTGELE_TOHUM
    ),
    "KNN": KNeighborsClassifier(n_neighbors=25, weights="distance"),
    "SVM": SVC(kernel="rbf", C=1, probability=True, random_state=RASTGELE_TOHUM),
    "Karar Ağacı": DecisionTreeClassifier(
        criterion="gini", max_depth=5, min_samples_leaf=20, random_state=RASTGELE_TOHUM
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=3,
        random_state=RASTGELE_TOHUM, n_jobs=-1
    ),
}

val_sonuclari = [dummy_sonuc]
val_tahminleri = {}

print("\nModeller eğitiliyor ve DOĞRULAMA setinde değerlendiriliyor...")

for model_adi, model in modeller.items():
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_val_scaled)
    y_proba = model.predict_proba(X_val_scaled)[:, 1]

    val_tahminleri[model_adi] = {"tahmin": y_pred, "olasilik": y_proba}
    val_sonuclari.append(model_degerlendir(model_adi, y_val, y_pred, y_proba))

    print(f"  {model_adi} eğitildi.")

sonuclar_df = pd.DataFrame(val_sonuclari).sort_values("ROC AUC", ascending=False)

print("\nMODEL KARŞILAŞTIRMA TABLOSU (doğrulama seti):")
print(sonuclar_df.round(4).to_string(index=False))

# görselleştirme
metrikler = ["Accuracy", "Precision", "Recall", "F1", "ROC AUC"]
grafik_df = sonuclar_df.set_index("Model")[metrikler]

plt.figure(figsize=(12, 6))
grafik_df.plot(kind="bar", colormap="Set2", edgecolor="black", linewidth=0.5, ax=plt.gca())
plt.axhline(dummy_sonuc["Accuracy"], color="red", linestyle="--", linewidth=1.2,
            label=f"Baseline accuracy ({dummy_sonuc['Accuracy']:.2f})")
plt.title("Modellerin Metrik Bazında Karşılaştırılması (doğrulama seti)")
plt.xlabel("Model")
plt.ylabel("Skor")
plt.xticks(rotation=15)
plt.ylim(0, 1)
plt.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig("grafik_04_model_karsilastirma.png", bbox_inches="tight")
plt.close()

# 17. Çapraz doğrulama ile model kararlılığının test edilmesi
print("\n[6] ÇAPRAZ DOĞRULAMA (5 katlı, ROC AUC)")
print("-" * 70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RASTGELE_TOHUM)
cv_sonuclari = []

for model_adi, model in modeller.items():
    skorlar = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="roc_auc")

    cv_sonuclari.append({
        "Model": model_adi,
        "Ortalama ROC AUC": skorlar.mean(),
        "Std Sapma": skorlar.std(),
    })

    print(f"{model_adi:20s}: {skorlar.mean():.4f} (+/- {skorlar.std():.4f})")

cv_df = pd.DataFrame(cv_sonuclari).sort_values("Ortalama ROC AUC", ascending=False)

print("\nÇapraz doğrulama özeti:")
print(cv_df.round(4).to_string(index=False))

# 18. GridSearchCV ile hiperparametre optimizasyonu
print("\n[7] HİPERPARAMETRE OPTİMİZASYONU (GridSearchCV)")
print("-" * 70)

parametre_izgarasi = {
    "n_estimators": [200, 300],
    "max_depth": [6, 10, None],
    "min_samples_leaf": [1, 3, 5],
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=RASTGELE_TOHUM, n_jobs=-1),
    param_grid=parametre_izgarasi,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
)

print(f"Denenecek kombinasyon sayısı: "
      f"{2 * 3 * 3}")
print("Arama yapılıyor...")

grid_search.fit(X_train_scaled, y_train)

print(f"\nEn iyi parametreler: {grid_search.best_params_}")
print(f"En iyi çapraz doğrulama ROC AUC: {grid_search.best_score_:.4f}")

en_iyi_model = grid_search.best_estimator_

optimize_pred = en_iyi_model.predict(X_val_scaled)
optimize_proba = en_iyi_model.predict_proba(X_val_scaled)[:, 1]

optimize_sonuc = model_degerlendir("Random Forest (optimize)", y_val, optimize_pred, optimize_proba)
val_tahminleri["Random Forest (optimize)"] = {"tahmin": optimize_pred, "olasilik": optimize_proba}

tum_sonuclar_df = pd.concat(
    [sonuclar_df, pd.DataFrame([optimize_sonuc])], ignore_index=True
).sort_values("ROC AUC", ascending=False)

print("\nDOĞRULAMA SETİ NİHAİ KARŞILAŞTIRMA TABLOSU:")
print(tum_sonuclar_df.round(4).to_string(index=False))

# 19. Seçilen modelin test veri setinde nihai değerlendirilmesi
print("\n[7.1] SEÇİLEN MODELİN TEST SETİNDE DEĞERLENDİRİLMESİ")
print("-" * 70)

secilen_model_adi = tum_sonuclar_df.iloc[0]["Model"]
print(f"Doğrulama setine göre seçilen model: {secilen_model_adi}")

if secilen_model_adi == "Random Forest (optimize)":
    secilen_model = en_iyi_model
elif secilen_model_adi == "Baseline (çoğunluk)":
    secilen_model = dummy_clf
else:
    secilen_model = modeller[secilen_model_adi]

# test seti burada ilk ve tek kez kullanılıyor
test_pred = secilen_model.predict(X_test_scaled)
test_proba = secilen_model.predict_proba(X_test_scaled)[:, 1]

test_sonuc = model_degerlendir(secilen_model_adi, y_test, test_pred, test_proba)

print("\nTEST SETİ SONUÇLARI (nihai performans):")
for anahtar, deger in test_sonuc.items():
    if anahtar != "Model":
        print(f"  {anahtar:10s}: {deger:.4f}")

val_auc = tum_sonuclar_df.iloc[0]["ROC AUC"]
test_auc = test_sonuc["ROC AUC"]

print(f"\nDoğrulama ROC AUC: {val_auc:.4f} | Test ROC AUC: {test_auc:.4f} | "
      f"Fark: {abs(val_auc - test_auc):.4f}")

if abs(val_auc - test_auc) < 0.05:
    print("Yorum: Doğrulama ve test performansı yakın, model genelleme yapabiliyor.")
else:
    print("Yorum: Doğrulama ve test arasında belirgin fark var, dikkatli yorumlanmalı.")

print("\nSınıflandırma raporu (test seti):")
print(classification_report(y_test, test_pred,
                            target_names=["Güvende değil", "Güvende"]))

# sonuçları kaydet
kayit_df = tum_sonuclar_df.copy()
kayit_df["Değerlendirme"] = "Doğrulama"

test_kayit = pd.DataFrame([test_sonuc])
test_kayit["Değerlendirme"] = "Test"

pd.concat([kayit_df, test_kayit], ignore_index=True).round(4).to_csv(
    "model_sonuclari.csv", index=False
)

# confusion matrix ve ROC eğrisi
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

conf_matrix = confusion_matrix(y_test, test_pred)
sns.heatmap(conf_matrix, annot=True, fmt="g", cmap="Blues", cbar=False,
            xticklabels=["Güvende değil", "Güvende"],
            yticklabels=["Güvende değil", "Güvende"], ax=axes[0])
axes[0].set_xlabel("Tahmin edilen sınıf")
axes[0].set_ylabel("Gerçek sınıf")
axes[0].set_title(f"{secilen_model_adi} - Confusion Matrix (test seti)")

for model_adi in tum_sonuclar_df["Model"]:
    if model_adi == "Baseline (çoğunluk)":
        continue
    fpr, tpr, _ = roc_curve(y_val, val_tahminleri[model_adi]["olasilik"])
    auc_skoru = roc_auc_score(y_val, val_tahminleri[model_adi]["olasilik"])
    axes[1].plot(fpr, tpr, linewidth=2, label=f"{model_adi} ({auc_skoru:.3f})")

axes[1].plot([0, 1], [0, 1], "k--", linewidth=1, label="Rastgele (0.500)")
axes[1].set_xlabel("Yanlış Pozitif Oranı")
axes[1].set_ylabel("Doğru Pozitif Oranı")
axes[1].set_title("ROC Eğrileri (doğrulama seti)")
axes[1].legend(loc="lower right", fontsize=8)

plt.tight_layout()
plt.savefig("grafik_05_confusion_roc.png", bbox_inches="tight")
plt.close()


# =============================================================================
# BÖLÜM 5 - YORUMLAMA VE ALGORİTMİK ADALET
# =============================================================================

print("\n[8] ÖZNİTELİK ÖNEMİ")
print("-" * 70)

# 20. Öznitelik öneminin incelenmesi
if hasattr(secilen_model, "feature_importances_"):
    onemler = secilen_model.feature_importances_
else:
    onemler = np.abs(modeller["Random Forest"].fit(
        X_train_scaled, y_train).feature_importances_)

onem_sirali = sorted(zip(onemler, X_train.columns), reverse=True)

print("En önemli 15 öznitelik:")
for onem, isim in onem_sirali[:15]:
    print(f"  {isim:32s}: {onem:.4f}")

ilk15 = onem_sirali[:15]

plt.figure(figsize=(10, 7))
sns.barplot(x=[i[0] for i in ilk15], y=[i[1] for i in ilk15],
            hue=[i[1] for i in ilk15], palette="viridis", legend=False)
plt.title("En Önemli 15 Öznitelik")
plt.xlabel("Önem derecesi (feature importance)")
plt.ylabel("Öznitelik")
plt.tight_layout()
plt.savefig("grafik_06_feature_importance.png", bbox_inches="tight")
plt.close()

# türetilen endekslerin sıralamadaki yeri
tum_isimler = [i[1] for i in onem_sirali]

print("\nTüretilen endekslerin önem sıralamasındaki yeri:")
for oznitelik in YENI_OZNITELIKLER:
    if oznitelik in tum_isimler:
        print(f"  {oznitelik:20s}: {tum_isimler.index(oznitelik) + 1}. sırada")

# 21. Lojistik regresyon katsayıları ile etkinin yönünün yorumlanması
print("\n[9] ETKİNİN YÖNÜ (Lojistik Regresyon Katsayıları)")
print("-" * 70)

log_reg = modeller["Lojistik Regresyon"]

katsayilar = pd.DataFrame({
    "oznitelik": X_train.columns,
    "katsayi": log_reg.coef_[0],
})
katsayilar["odds_orani"] = np.exp(katsayilar["katsayi"])
katsayilar = katsayilar.sort_values("katsayi", ascending=False)

print("Güvende hissetme olasılığını EN ÇOK ARTIRAN 8 öznitelik:")
print(katsayilar.head(8).round(4).to_string(index=False))

print("\nGüvende hissetme olasılığını EN ÇOK AZALTAN 8 öznitelik:")
print(katsayilar.tail(8).round(4).to_string(index=False))

en_etkili = pd.concat([katsayilar.head(10), katsayilar.tail(10)])

plt.figure(figsize=(10, 8))
renkler = ["#66c2a5" if k > 0 else "#fc8d62" for k in en_etkili["katsayi"]]

plt.barh(en_etkili["oznitelik"], en_etkili["katsayi"],
         color=renkler, edgecolor="black", linewidth=0.4)
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Lojistik Regresyon Katsayıları\n"
          "(Yeşil: güvende hissetmeyi artırır, Turuncu: azaltır)")
plt.xlabel("Katsayı değeri")
plt.ylabel("Öznitelik")
plt.tight_layout()
plt.savefig("grafik_07_katsayilar.png", bbox_inches="tight")
plt.close()

# 22. Modelin cinsiyet gruplarındaki başarım farkının incelenmesi
print("\n[10] ALGORİTMİK ADALET: CİNSİYETE GÖRE MODEL BAŞARIMI")
print("-" * 70)

# test setindeki katılımcıların cinsiyetini geri alalım
cinsiyet_test = df_calisma.loc[X_test.index, "cinsiyet"]

adalet_sonuclari = []

for cinsiyet in cinsiyet_test.dropna().unique():
    maske = (cinsiyet_test == cinsiyet).values

    if maske.sum() < 30:
        continue

    grup_gercek = y_test[maske]
    grup_tahmin = test_pred[maske]
    grup_olasilik = test_proba[maske]

    tn, fp, fn, tp = confusion_matrix(grup_gercek, grup_tahmin, labels=[0, 1]).ravel()

    adalet_sonuclari.append({
        "Grup": cinsiyet,
        "n": int(maske.sum()),
        "Gerçek güvende oranı": grup_gercek.mean(),
        "Tahmin güvende oranı": grup_tahmin.mean(),
        "Accuracy": accuracy_score(grup_gercek, grup_tahmin),
        "ROC AUC": roc_auc_score(grup_gercek, grup_olasilik),
        "Yanlış pozitif oranı": fp / (fp + tn) if (fp + tn) > 0 else np.nan,
        "Yanlış negatif oranı": fn / (fn + tp) if (fn + tp) > 0 else np.nan,
    })

adalet_df = pd.DataFrame(adalet_sonuclari)

print("Cinsiyet gruplarına göre model başarımı (test seti):")
print(adalet_df.round(4).to_string(index=False))

if len(adalet_df) == 2:
    acc_farki = abs(adalet_df["Accuracy"].iloc[0] - adalet_df["Accuracy"].iloc[1])
    fpr_farki = abs(adalet_df["Yanlış pozitif oranı"].iloc[0] - adalet_df["Yanlış pozitif oranı"].iloc[1])
    fnr_farki = abs(adalet_df["Yanlış negatif oranı"].iloc[0] - adalet_df["Yanlış negatif oranı"].iloc[1])

    print(f"\nGruplar arası accuracy farkı            : {acc_farki:.4f}")
    print(f"Gruplar arası yanlış pozitif oranı farkı: {fpr_farki:.4f}")
    print(f"Gruplar arası yanlış negatif oranı farkı: {fnr_farki:.4f}")

    print("\nYorum: Model genel olarak başarılı olsa bile, hata türlerinin gruplar")
    print("       arasında farklı dağılması algoritmik adalet açısından önemlidir.")

adalet_df.round(4).to_csv("adalet_analizi.csv", index=False)

# adalet görselleştirmesi
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

x_pos = np.arange(len(adalet_df))
genislik = 0.35

axes[0].bar(x_pos - genislik / 2, adalet_df["Gerçek güvende oranı"], genislik,
            label="Gerçek", color="#66c2a5", edgecolor="black", linewidth=0.4)
axes[0].bar(x_pos + genislik / 2, adalet_df["Tahmin güvende oranı"], genislik,
            label="Model tahmini", color="#fc8d62", edgecolor="black", linewidth=0.4)
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(adalet_df["Grup"])
axes[0].set_ylabel("Güvende hissetme oranı")
axes[0].set_title("Gerçek ve Tahmin Edilen Oranlar")
axes[0].legend(fontsize=9)

axes[1].bar(x_pos - genislik / 2, adalet_df["Yanlış pozitif oranı"], genislik,
            label="Yanlış pozitif oranı", color="#8da0cb", edgecolor="black", linewidth=0.4)
axes[1].bar(x_pos + genislik / 2, adalet_df["Yanlış negatif oranı"], genislik,
            label="Yanlış negatif oranı", color="#e78ac3", edgecolor="black", linewidth=0.4)
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(adalet_df["Grup"])
axes[1].set_ylabel("Hata oranı")
axes[1].set_title("Cinsiyet Gruplarına Göre Hata Türleri")
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig("grafik_08_algoritmik_adalet.png", bbox_inches="tight")
plt.close()

# =============================================================================
# ÖZET
# =============================================================================

print("\n" + "=" * 70)
print("PROJE ÖZETİ")
print("=" * 70)
print(f"Veri seti           : TGSS 2024, {len(df_calisma)} katılımcı")
print(f"Öznitelik sayısı    : {X.shape[1]} (encoding sonrası)")
print(f"Hedef değişken      : Karanlıkta güvende hissetme (dengeli, %{guvende_orani * 100:.1f})")
print(f"Denenen model       : {len(modeller)} + 1 optimize + 1 baseline")
print(f"Seçilen model       : {secilen_model_adi}")
print(f"  Test ROC AUC      : {test_sonuc['ROC AUC']:.4f}")
print(f"  Test Accuracy     : {test_sonuc['Accuracy']:.4f}")
print(f"  Test F1           : {test_sonuc['F1']:.4f}")
print(f"Cinsiyet farkı      : {cinsiyet_farki * 100:.1f} yüzde puanı")
print("\nÜretilen grafikler: 8 adet PNG dosyası")
print("Çıktı dosyaları    : model_sonuclari.csv, adalet_analizi.csv")
print("=" * 70)
print("Proje tamamlandı.")
