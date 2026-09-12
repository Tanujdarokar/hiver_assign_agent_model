"""
Evaluation Baselines Module
===========================
Implements Baseline 1 (Majority Class Classifier) and Baseline 2 (Rule/Keyword Classifier)
and computes standardized classification metrics on the Golden Set.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
GOLDEN_CSV = ROOT / "data" / "golden" / "golden_set.csv"


def calculate_metrics(y_true: list, y_pred: list) -> dict:
    """
    Computes classification evaluation metrics:
    Accuracy, Macro Precision, Macro Recall, Macro F1, Weighted F1.
    """
    acc = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    prec_w, rec_w, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(prec_macro, 4),
        "macro_recall": round(rec_macro, 4),
        "macro_f1": round(f1_macro, 4),
        "weighted_f1": round(f1_weighted, 4),
    }


class MajorityClassBaseline:
    """Baseline 1: Trivial Majority Class Classifier."""

    def __init__(self):
        self.majority_class = "general_inquiry_or_other"

    def fit(self, y_train: list):
        if len(y_train) > 0:
            self.majority_class = pd.Series(y_train).mode()[0]

    def predict(self, X: list) -> list:
        return [self.majority_class] * len(X)


class KeywordRuleBaseline:
    """Baseline 2: Simple Keyword/Rule-Based Classifier."""

    def predict_one(self, text: str) -> str:
        m = text.lower()
        if any(k in m for k in ["ios", "update", "beta"]):
            return "software_update_and_ios"
        if any(k in m for k in ["battery", "charge", "power"]):
            return "battery_and_power"
        if any(k in m for k in ["app store", "download"]):
            return "app_store_and_downloads"
        if any(k in m for k in ["screen", "crack", "display", "camera"]):
            return "hardware_and_display"
        if any(k in m for k in ["itunes", "music", "song"]):
            return "itunes_and_apple_music"
        if any(k in m for k in ["wifi", "wi-fi", "bluetooth", "sim"]):
            return "connectivity_and_cellular"
        if any(k in m for k in ["apple id", "passcode", "password"]):
            return "apple_id_and_security"
        if any(k in m for k in ["icloud", "storage", "backup"]):
            return "icloud_and_storage"
        if any(k in m for k in ["refund", "billing", "charge"]):
            return "apple_pay_and_billing"
        if any(k in m for k in ["airpod", "headphone", "audio"]):
            return "audio_and_airpods"
        return "general_inquiry_or_other"

    def predict(self, X: list) -> list:
        return [self.predict_one(text) for text in X]


def evaluate_baselines():
    print(f"Loading Golden Set from {GOLDEN_CSV} ...")
    df = pd.read_csv(GOLDEN_CSV)

    y_true = df["intent"].tolist()
    X = df["message"].tolist()

    # 1. Baseline 1
    b1 = MajorityClassBaseline()
    b1.fit(y_true)
    y_pred_b1 = b1.predict(X)
    m1 = calculate_metrics(y_true, y_pred_b1)

    # 2. Baseline 2
    b2 = KeywordRuleBaseline()
    y_pred_b2 = b2.predict(X)
    m2 = calculate_metrics(y_true, y_pred_b2)

    print("\n" + "=" * 60)
    print("BASELINE EVALUATION RESULTS (Golden Set N=200)")
    print("=" * 60)
    print("Metric             | Baseline 1 (Majority) | Baseline 2 (Keywords)")
    print("-" * 60)
    for k in m1.keys():
        print(f"{k:18s} | {m1[k]:21.4f} | {m2[k]:21.4f}")
    print("=" * 60)

    return {"baseline_1": m1, "baseline_2": m2}


if __name__ == "__main__":
    evaluate_baselines()
