"""
DyeMind — Sliding Window Feature Engineering
=============================================
Her zaman noktası için son WINDOW_SIZE dakikayı flatten ederek
model girdisi olarak kullanılacak öznitelik matrisi üretir.
"""

import numpy as np
import pandas as pd
from data_prep import SENSOR_COLS


def extract_window_features(
    df: pd.DataFrame,
    window_size: int = 30,
    target: bool = False,
) -> tuple:
    """
    Kayan pencere ile öznitelik ve (opsiyonel) hedef matrisi üretir.

    Parameters
    ----------
    df : pd.DataFrame
        Sütunlar: batch_id, t, S1..S7, label
    window_size : int
        Pencere büyüklüğü (dakika). Varsayılan: 30
    target : bool
        True ise XGBoost için X=[t-window..t-1], Y=[t] şeklinde ayırır.
        False ise IF/OCSVM için X=[t-window+1..t] (pencere dahil t)

    Returns
    -------
    X : np.ndarray  — (n_samples, n_features)
    meta : pd.DataFrame — batch_id, t, label
    feat_names : list[str]
    Y : np.ndarray (sadece target=True)
    """
    X_list, Y_list, meta_list = [], [], []

    for batch_id, group in df.groupby("batch_id"):
        group   = group.sort_values("t").reset_index(drop=True)
        sensors = group[SENSOR_COLS].values   # (n, 7)
        label   = group["label"].iloc[0]
        n       = len(sensors)

        if target:
            # XGBoost: X = pencere öncesi, Y = t anındaki değer
            start = window_size
        else:
            # IF / OCSVM: X = pencere sonu dahil t
            start = window_size - 1

        for t_idx in range(start, n):
            if target:
                window = sensors[t_idx - window_size : t_idx]   # (30, 7)
                y_val  = sensors[t_idx]                          # (7,)
                Y_list.append(y_val)
            else:
                window = sensors[t_idx - window_size + 1 : t_idx + 1]  # (30, 7)

            X_list.append(window.flatten())   # (210,)
            meta_list.append({
                "batch_id": batch_id,
                "t"       : group["t"].iloc[t_idx],
                "label"   : label,
            })

    X    = np.vstack(X_list)
    meta = pd.DataFrame(meta_list).reset_index(drop=True)
    n_feat = window_size * len(SENSOR_COLS)

    feat_names = [
        f"w{i // len(SENSOR_COLS)}_S{(i % len(SENSOR_COLS)) + 1}"
        for i in range(n_feat)
    ]

    print(f"✓ Feature extraction | window={window_size} dk | "
          f"features={n_feat} | samples={len(X):,}")

    if target:
        Y = np.vstack(Y_list)
        return X, Y, meta, feat_names

    return X, meta, feat_names


def split_train_test(
    df: pd.DataFrame,
    n_train_normal: int = None,
) -> tuple:
    """
    Normal partileri eğitim, kalanları test olarak ayırır.

    Parameters
    ----------
    df : pd.DataFrame
    n_train_normal : int, optional
        Eğitime alınacak normal parti sayısı.
        None ise tüm normal partiler eğitime girer.
    """
    normal_ids = df[df["label"] == 0]["batch_id"].unique()
    anom_ids   = df[df["label"] == 1]["batch_id"].unique()

    if n_train_normal is None or n_train_normal >= len(normal_ids):
        train_ids = normal_ids
    else:
        train_ids = normal_ids[:n_train_normal]

    train_df = df[df["batch_id"].isin(train_ids)].copy()
    test_df  = df.copy()   # tüm partiler test setinde

    print(f"Eğitim : {len(train_ids)} normal parti → {len(train_df):,} satır")
    print(f"Test   : {df['batch_id'].nunique()} parti → {len(test_df):,} satır")

    return train_df, test_df, train_ids, anom_ids
