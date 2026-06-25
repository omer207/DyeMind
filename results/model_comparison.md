# Model Karşılaştırma Sonuçları

## Performans Metrikleri (Parti Düzeyi)

| Model | ROC-AUC | Avg Precision | F1 | Precision | Recall | Eşik |
|---|---|---|---|---|---|---|
| Isolation Forest | 0.8611 | 0.8774 | **0.8571** | 0.7500 | **1.0000** | 0.538 |
| One-Class SVM | **1.0000** | **1.0000** | 0.8000 | **1.0000** | 0.6667 | 4.473 |
| XGBoost Forecasting | 0.7222 | 0.7794 | 0.7692 | 0.7143 | 0.8333 | 248.5 |

## Karmaşıklık Matrisi

| Model | TP | FP | FN | TN | Doğru/Toplam |
|---|---|---|---|---|---|
| Isolation Forest | 6 | 2 | 0 | 4 | 10/12 |
| One-Class SVM | 4 | 0 | 2 | 6 | 10/12 |
| XGBoost Forecasting | 5 | 2 | 1 | 4 | 9/12 |

## Erken Uyarı Süreleri (Anormal Partiler)

| Parti | IF | OCSVM | XGBoost |
|---|---|---|---|
| 129469 | t=32 | ❌ | ❌ |
| 129487 | t=134 | ❌ | t=31 |
| 130246 | t=30 | t=30 | t=31 |
| 130565 | t=43 | t=89 | t=321 |
| 130567 | t=356 | t=30 | t=51 |
| 130569 | t=30 | t=30 | t=56 |

## Sensör Önem Sıralaması

| Sıra | IF (Permütasyon) | OCSVM (Permütasyon) | XGBoost (SHAP) |
|---|---|---|---|
| 1 | İletkenlik (0.0464) | BK1 Sıcaklığı (4.3864) | İletkenlik (3.4262) |
| 2 | BK1 Seviyesi (0.0309) | İletkenlik (2.7851) | BK1 Sıcaklığı (2.3161) |
| 3 | BK1 Sıcaklığı (0.0242) | AK Seviyesi (0.1522) | AK Seviyesi (2.1821) |
| 4 | BK2 Seviyesi (0.0222) | BK2 Seviyesi (0.1443) | BK2 Sıcaklık (1.8497) |
| 5 | BK2 Sıcaklık (0.0129) | AK Sıcaklığı (0.0112) | BK2 Seviyesi (1.8086) |

## Model Seçim Rehberi

| Kullanım Amacı | Önerilen Model |
|---|---|
| Hiçbir anormal partiyi kaçırma | Isolation Forest (Recall=1.000) |
| Yanlış alarm üretmeme | One-Class SVM (Precision=1.000) |
| Kök neden açıklaması | XGBoost (SHAP TreeExplainer) |
| Genel kullanım / üretim | Isolation Forest |
