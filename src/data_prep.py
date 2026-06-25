"""
DyeMind — Veri Hazırlık Modülü
================================
Ham Excel dosyalarını okur, temizler ve birleştirilmiş
bir DataFrame olarak döndürür.
"""

import os
import io
import pandas as pd
import numpy as np


SENSOR_COLS = [
    "S1_AK_Sicakligi",
    "S2_BK1_Sicakligi",
    "S3_BK2_Sicaklik",
    "S4_AK_Seviyesi",
    "S5_Iletkenlik",
    "S6_BK1_Seviyesi",
    "S7_BK2_Seviyesi",
]

SENSOR_LABELS = [
    "AK Sıcaklığı",
    "BK1 Sıcaklığı",
    "BK2 Sıcaklık",
    "AK Seviyesi",
    "İletkenlik",
    "BK1 Seviyesi",
    "BK2 Seviyesi",
]

# Normal parti referans aralıkları (P5–P95, gerçek veriden türetilmiştir)
NORMAL_RANGES = {
    "S1_AK_Sicakligi": (5.6,  105.0),
    "S2_BK1_Sicakligi": (28.3,  38.8),
    "S3_BK2_Sicaklik":  (30.1, 156.7),
    "S4_AK_Seviyesi":   (4.2,  6259.0),
    "S5_Iletkenlik":    (0.0,    0.94),
    "S6_BK1_Seviyesi":  (0.0,  105.9),
    "S7_BK2_Seviyesi":  (0.0,   70.3),
}


def load_and_clean(filepath: str, label: int = None) -> pd.DataFrame:
    """
    Tek bir Excel dosyasını yükler, temizler ve standart formata getirir.

    Parameters
    ----------
    filepath : str
        Excel dosyasının yolu (.xlsx veya .xls)
    label : int, optional
        0=Normal, 1=Anormal. None ise dosyadaki 'Kusur' sütununu kullanır.

    Returns
    -------
    pd.DataFrame
        Sütunlar: batch_id, t, S1..S7, label
    """
    with open(filepath, "rb") as f:
        raw = f.read()

    df = pd.read_excel(io.BytesIO(raw), sheet_name=0, engine="openpyxl")

    # Sensörleri sayısala zorla, kirli değerleri NaN yap
    for col in SENSOR_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    # NaN doldur
    df[SENSOR_COLS] = df[SENSOR_COLS].ffill().fillna(0)

    # t sütunu yoksa oluştur
    if "t" not in df.columns:
        df["t"] = range(1, len(df) + 1)

    # Etiket
    if label is not None:
        df["label"] = label
    elif "Kusur" in df.columns:
        df["label"] = df["Kusur"].astype(int)
    else:
        df["label"] = -1  # bilinmiyor

    # batch_id
    if "batch_id" not in df.columns:
        batch_name = os.path.splitext(os.path.basename(filepath))[0]
        df["batch_id"] = batch_name

    return df[["batch_id", "t"] + SENSOR_COLS + ["label"]]


def load_multiple(file_label_pairs: list) -> pd.DataFrame:
    """
    Birden fazla Excel dosyasını yükleyip birleştirir.

    Parameters
    ----------
    file_label_pairs : list of (filepath, label) tuples

    Returns
    -------
    pd.DataFrame
        Tüm partilerin birleştirilmiş verisi
    """
    frames = []
    for filepath, label in file_label_pairs:
        df = load_and_clean(filepath, label)
        frames.append(df)
        print(f"  ✓ {os.path.basename(filepath):20s} — {len(df):>4} satır, label={label}")

    combined = pd.concat(frames, ignore_index=True)
    print(f"\nToplam: {len(combined):,} satır | {combined['batch_id'].nunique()} parti")
    return combined


def validate_data(df: pd.DataFrame) -> None:
    """Yüklenen veri setini doğrular ve olası sorunları raporlar."""
    print("── VERİ DOĞRULAMA ──────────────────────────")
    print(f"Toplam satır   : {len(df):,}")
    print(f"Toplam parti   : {df['batch_id'].nunique()}")
    label_counts = df.groupby("label")["batch_id"].nunique()
    print(f"Normal parti   : {label_counts.get(0, 0)}")
    print(f"Anormal parti  : {label_counts.get(1, 0)}")

    na_counts = df[SENSOR_COLS].isna().sum()
    if na_counts.any():
        print(f"\n[UYARI] Eksik değerler:\n{na_counts[na_counts > 0]}")
    else:
        print("Eksik değer    : YOK ✓")

    lengths = df.groupby("batch_id")["t"].max()
    print(f"Parti uzunlukları: {lengths.min()} – {lengths.max()} dk "
          f"(ort. {lengths.mean():.0f})")
