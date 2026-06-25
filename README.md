# 🤖 DyeMind -Tekstil Boyahanelerinde Yapay Zeka Destekli Anomali Tespiti

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.8-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-red?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Bitirme%20Projesi-purple?style=for-the-badge)

**İstanbul Üniversitesi-Cerrahpaşa | Endüstri Mühendisliği Bölümü**
**2025–2026 Bitirme Projesi**

*[Ömer Faruk DAVUTOĞLU] · [Muhammed Rüçhan Durak] · [Furkan Aras]*
*Danışman: [Prof.Dr.Ersin NAMLI]*

</div>

---

## 📌 Projeye Genel Bakış

Tekstil boyahanelerinde üretim partileri, geleneksel yöntemlerle yalnızca süreç **tamamlandıktan sonra** kalite kontrole tabi tutulmaktadır. Bu reaktif yaklaşım:

- ♻️ Yüksek yeniden işlem (rebatching) maliyetlerine
- ⚡ Gereksiz enerji ve kimyasal tüketimine
- 📦 Teslimat gecikmelerine

yol açmaktadır.

**DyeMind**, bu sorunu çözmek için geliştirilmiş iki katmanlı bir sistemdir:

1. **Makine Öğrenmesi Katmanı** → Üretim sırasında anomaliyi tespit eder
2. **Yapay Zeka Ajan Katmanı** → Tespiti operatöre sade bir dille yorumlar

---

## 🏗️ Sistem Mimarisi

```
Ham Sensör Verisi (7 sensör, dakikalık)
          │
          ▼
┌─────────────────────┐
│  Sliding Window     │  ← 30 dk pencere → 210 öznitelik
│  Feature Engineering│
└─────────┬───────────┘
          │
          ▼
┌─────────────────────────────────────────┐
│         Anomali Tespit Modelleri        │
│  ┌──────────────┐ ┌──────────────────┐  │
│  │Isolation     │ │ One-Class SVM    │  │
│  │Forest        │ │ (RBF kernel)     │  │
│  └──────────────┘ └──────────────────┘  │
│  ┌──────────────┐                       │
│  │XGBoost       │  ← MAE tabanlı skor  │
│  │Forecasting   │                       │
│  └──────────────┘                       │
└─────────┬───────────────────────────────┘
          │  Anomali Skoru + Sensör Katkısı
          ▼
┌─────────────────────┐
│   DyeMind Ajanı     │  ← Claude LLM üzeri
│   (Karar Desteği)   │
└─────────────────────┘
          │
          ▼
    Operatör Raporu
```

---

## 📊 Veri Seti

| Özellik | Değer |
|---|---|
| Toplam parti | 12 (6 normal, 6 anormal) |
| Toplam ölçüm | 4.970 satır |
| Örnekleme aralığı | Dakikalık |
| Parti uzunluğu | 166–679 dakika |
| Sensör sayısı | 7 |
| Etiket türü | Parti düzeyi (Kusur: 0/1) |

### 🔬 Sensörler

| Kod | Sensör | Normal Aralık (P5–P95) |
|---|---|---|
| S1 | AK Sıcaklığı | 5.6 – 105.0 °C |
| S2 | BK1 Sıcaklığı | 28.3 – 38.8 °C |
| S3 | BK2 Sıcaklık | 30.1 – 156.7 °C |
| S4 | AK Seviyesi | 4.2 – 6.259 |
| S5 | İletkenlik | 0.0 – 0.94 |
| S6 | BK1 Seviyesi | 0.0 – 105.9 |
| S7 | BK2 Seviyesi | 0.0 – 70.3 |

---

## 🤖 Model Performansı

| Model | ROC-AUC | F1 | Recall | Precision | İlk Alarm |
|---|---|---|---|---|---|
| **Isolation Forest** | 0.861 | **0.857** | **1.000** | 0.750 | t=30–134 dk |
| One-Class SVM | **1.000** | 0.800 | 0.667 | **1.000** | t=30–89 dk |
| XGBoost Forecasting | 0.722 | 0.769 | 0.833 | 0.714 | t=31–321 dk |

> 💡 **Isolation Forest** en dengeli performansı sunmaktadır. **OCSVM** sıfır yanlış alarm üretir. **XGBoost** SHAP ile en zengin açıklanabilirliği sağlar.

### 🔍 Sensör Önem Sıralaması (Açıklanabilirlik)

Üç modelde de tutarlı biçimde öne çıkan sensörler:

1. 🔴 **İletkenlik** - Kimyasal konsantrasyon sapmasının birincil göstergesi
2. 🔴 **BK1 Sıcaklığı** - Termal dengesizliğin en net işareti
3. 🟡 AK Seviyesi, BK1 Seviyesi, BK2 Seviyesi

---

## 🧠 DyeMind Ajanı

DyeMind, **Claude** büyük dil modeli üzerine inşa edilmiş bir yapay zeka karar destek ajanıdır. `agent/` klasöründe talimat seti mevcuttur.

### Özellikler

- ✅ Ham sensör verisi → Kural tabanlı değerlendirme
- ✅ Anomali skoru → Eşik karşılaştırması ve yorumu
- ✅ Yapılandırılmış rapor: Karar · Gerekçe · Sapan Sensörler · Zamanlama · Güven Notu
- ✅ Kapsam dışı sorgulara tutarlı ret
- ✅ Sıfır ek altyapı maliyeti (claude.ai Projects üzerinde çalışır)

### Örnek Çıktı

```
Merhaba, ben boyahane anomali tespit ajanı DyeMind 🤖

1. Karar: Bu partide anomali tespit edildi - partinin orta aşamasında
   birden fazla yüksek öncelikli sensör eş zamanlı olarak normal bandın
   dışına çıkmıştır.

2. Gerekçe: t=200 civarında belirgin bir kırılma yaşanmıştır...

3. Sapan Sensörler:
   🔴 İletkenlik (en yüksek öncelik): t=215'ten itibaren 0.94 eşiğini aşmış
   🔴 BK1 Sıcaklığı (yüksek öncelik): 80 ölçüm üst sınırı aşmış
   ✅ AK Sıcaklığı: Sapma yok
   ...

4. Zamanlama: Anomali t=200 civarında başlamıştır.

5. Güven Notu: Kesin karar için eğitilmiş model çıktısıyla
   doğrulanması önerilir.
```

---

## 📁 Repo Yapısı

```
DyeMind/
│
├── 📓 notebooks/
│   ├── 01_EDA.ipynb                  ← Keşifsel veri analizi
│   ├── 02_IsolationForest.ipynb      ← IF modeli ve sonuçları
│   ├── 03_OCSVM.ipynb                ← OCSVM modeli ve sonuçları
│   └── 04_XGBoost.ipynb              ← XGBoost modeli + SHAP
│
├── 🐍 src/
│   ├── data_prep.py                  ← Veri yükleme ve temizleme
│   ├── feature_engineering.py        ← Sliding window feature extraction
│   └── models.py                     ← Model eğitim ve değerlendirme
│
├── 🤖 agent/
│   └── dyemind_instructions.md       ← DyeMind ajan talimat seti
│
├── 📊 results/
│   └── model_comparison.md           ← Model karşılaştırma özeti
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Kurulum ve Kullanım

### 1. Repoyu klonla

```bash
git clone https://github.com/[omer207]/DyeMind.git
cd DyeMind
```

### 2. Sanal ortam oluştur

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows
```

### 3. Bağımlılıkları yükle

```bash
pip install -r requirements.txt
```

### 4. Notebook'ları çalıştır

```bash
jupyter notebook notebooks/
```

Sırayla çalıştır: `01_EDA` → `02_IsolationForest` → `03_OCSVM` → `04_XGBoost`

### 5. DyeMind ajanını kur

`agent/dyemind_instructions.md` dosyasındaki talimatları **claude.ai Projects** → **Custom Instructions** alanına yapıştır.

---

## 🛠️ Kullanılan Teknolojiler

| Kategori | Araç |
|---|---|
| Dil | Python 3.10+ |
| Veri işleme | pandas, numpy |
| Modelleme | scikit-learn, xgboost |
| Açıklanabilirlik | SHAP |
| Görselleştirme | matplotlib, seaborn |
| Ajan | Claude (Anthropic) - claude.ai Projects |
| Geliştirme ortamı | Google Colab / Jupyter |

---

## 📈 Gelecek Çalışmalar

- [ ] Daha geniş veri seti ile model yeniden eğitimi
- [ ] Zaman noktalı etiketleme ile erken uyarı doğrulaması
- [ ] Çok reçeteli genelleme
- [ ] ERP sistemi entegrasyonu
- [ ] DyeMind → API tabanlı gerçek zamanlı tetikleme

---

## 📄 Lisans

![License](https://img.shields.io/badge/License-Apache%202.0-green?style=for-the-badge)
---

## 📬 İletişim

| İsim | GitHub |
|---|---|
| [Ömer Faruk DAVUTOĞLU] | [@omer207](https://github.com/omer207) ||

---

<div align="center">

**İstanbul Üniversitesi-Cerrahpaşa | Endüstri Mühendisliği | 2025–2026**

*"Reaktif kontrol değil, proaktif tespit."*

</div>
