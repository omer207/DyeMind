"""
DyeMind — Model Eğitim ve Değerlendirme Modülü
===============================================
Isolation Forest, One-Class SVM ve XGBoost Forecasting
modellerini eğitir, skorlar ve değerlendirir.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, average_precision_score,
    precision_recall_curve, roc_curve, f1_score,
)
from xgboost import XGBRegressor


# ─── Eşik Optimizasyonu ──────────────────────────────────────────────────────

def batch_level_eval(meta_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Parti düzeyinde tahmin ve gerçek etiketi karşılaştırır."""
    grp = meta_df.groupby("batch_id").agg(
        true_label = ("label",         "first"),
        max_score  = ("anomaly_score", "max"),
        mean_score = ("anomaly_score", "mean"),
    ).reset_index()
    grp["pred_label"] = (grp["max_score"] > threshold).astype(int)
    return grp


def optimize_threshold(meta_df: pd.DataFrame) -> tuple:
    """F1'i maksimize eden eşiği tarama ile bulur."""
    scores = meta_df["anomaly_score"].values
    thresholds = np.percentile(scores, np.arange(70, 99, 0.5))
    results = []

    for thr in thresholds:
        grp = batch_level_eval(meta_df, thr)
        tp = ((grp["pred_label"]==1) & (grp["true_label"]==1)).sum()
        fp = ((grp["pred_label"]==1) & (grp["true_label"]==0)).sum()
        fn = ((grp["pred_label"]==0) & (grp["true_label"]==1)).sum()
        f1 = tp / (tp + 0.5*(fp + fn) + 1e-9)
        try:
            auc = roc_auc_score(grp["true_label"], grp["max_score"])
        except Exception:
            auc = 0.0
        results.append({"threshold": thr, "f1": f1, "auc": auc})

    thr_df   = pd.DataFrame(results)
    best_row = thr_df.loc[thr_df["f1"].idxmax()]
    return float(best_row["threshold"]), thr_df


def print_report(meta_df: pd.DataFrame, threshold: float, model_name: str) -> None:
    """Performans raporunu yazdırır."""
    batch_results = batch_level_eval(meta_df, threshold)
    auc = roc_auc_score(batch_results["true_label"], batch_results["max_score"])
    ap  = average_precision_score(batch_results["true_label"], batch_results["max_score"])
    print(f"\n{'═'*55}")
    print(f"  {model_name} — PARTİ DÜZEYİ PERFORMANS")
    print(f"{'═'*55}")
    print(classification_report(
        batch_results["true_label"], batch_results["pred_label"],
        target_names=["Normal", "Anormal"], zero_division=0,
    ))
    print(f"ROC-AUC        : {auc:.4f}")
    print(f"Avg Precision  : {ap:.4f}")
    print(f"Eşik           : {threshold:.4f}")


# ─── Isolation Forest ────────────────────────────────────────────────────────

class IsolationForestDetector:
    def __init__(self, n_estimators=300, contamination=0.05, random_state=42):
        self.scaler = StandardScaler()
        self.model  = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X_train: np.ndarray):
        X_sc = self.scaler.fit_transform(X_train)
        self.model.fit(X_sc)
        print(f"✓ Isolation Forest eğitildi | {len(X_train):,} örnek")
        return self

    def score(self, X: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
        X_sc        = self.scaler.transform(X)
        raw_scores  = self.model.score_samples(X_sc)
        anom_scores = -raw_scores
        meta = meta.copy()
        meta["anomaly_score"] = anom_scores
        return meta


# ─── One-Class SVM ───────────────────────────────────────────────────────────

class OCSVMDetector:
    def __init__(self, kernel="rbf", nu=0.05, gamma="scale"):
        self.scaler = StandardScaler()
        self.model  = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)

    def fit(self, X_train: np.ndarray):
        X_sc = self.scaler.fit_transform(X_train)
        self.model.fit(X_sc)
        n_sv = self.model.support_vectors_.shape[0]
        print(f"✓ OCSVM eğitildi | {len(X_train):,} örnek | {n_sv} support vector")
        return self

    def score(self, X: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
        X_sc        = self.scaler.transform(X)
        raw_scores  = self.model.decision_function(X_sc)
        anom_scores = -raw_scores   # yüksek = daha anomalik
        meta = meta.copy()
        meta["anomaly_score"] = anom_scores
        return meta


# ─── XGBoost Forecasting ─────────────────────────────────────────────────────

class XGBoostForecaster:
    def __init__(self, n_estimators=300, max_depth=5, learning_rate=0.05,
                 random_state=42):
        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()
        base_xgb = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
        )
        self.model = MultiOutputRegressor(base_xgb, n_jobs=-1)

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray):
        X_sc = self.scaler_X.fit_transform(X_train)
        Y_sc = self.scaler_Y.fit_transform(Y_train)
        self.model.fit(X_sc, Y_sc)
        print(f"✓ XGBoost eğitildi | {len(X_train):,} örnek | 7 alt model")
        return self

    def score(self, X: np.ndarray, Y_actual: np.ndarray,
              meta: pd.DataFrame) -> pd.DataFrame:
        X_sc     = self.scaler_X.transform(X)
        Y_pred_sc = self.model.predict(X_sc)
        Y_pred   = self.scaler_Y.inverse_transform(Y_pred_sc)
        residuals = np.abs(Y_actual - Y_pred)      # (n, 7)
        anom_scores = residuals.mean(axis=1)        # MAE skoru

        meta = meta.copy()
        meta["anomaly_score"] = anom_scores
        return meta
