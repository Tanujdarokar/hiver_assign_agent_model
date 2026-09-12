"""
Phase 7 & 12 — Comprehensive Evaluation Harness
================================================
Evaluates all intent classifiers (Baseline 1, Baseline 2, Real Classifier)
and the escalation decisions against the Golden Evaluation Set.
"""

import sys
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

GOLDEN_CSV = ROOT / "data" / "golden" / "golden_set.csv"
from evaluation.baselines import MajorityClassBaseline, KeywordRuleBaseline, calculate_metrics
from src.intent_classifier import IntentClassifier


def run_full_classifier_evaluation() -> pd.DataFrame:
    print(f"Loading Golden Set (N=200) from {GOLDEN_CSV} ...")
    df = pd.read_csv(GOLDEN_CSV)

    X = df["message"].tolist()
    y_true = df["intent"].tolist()

    # 1. Baseline 1: Majority Class
    b1 = MajorityClassBaseline()
    b1.fit(y_true)
    y_pred_b1 = b1.predict(X)
    m1 = calculate_metrics(y_true, y_pred_b1)

    # 2. Baseline 2: Keyword / Rule-Based
    b2 = KeywordRuleBaseline()
    y_pred_b2 = b2.predict(X)
    m2 = calculate_metrics(y_true, y_pred_b2)

    # 3. Real Intent Classifier
    clf = IntentClassifier()
    clf.load()
    preds_clf = clf.predict_batch(X)
    y_pred_clf = [p["intent"] for p in preds_clf]
    m3 = calculate_metrics(y_true, y_pred_clf)

    # Summary table
    metrics = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]
    summary = []
    for metric in metrics:
        summary.append({
            "Metric": metric,
            "Baseline 1 (Majority)": m1[metric],
            "Baseline 2 (Keywords)": m2[metric],
            "Real Intent Classifier": m3[metric]
        })

    summary_df = pd.DataFrame(summary)

    print("\n" + "=" * 75)
    print("CLASSIFIER EVALUATION COMPARISON ON GOLDEN SET (N=200)")
    print("=" * 75)
    print(summary_df.to_string(index=False))
    print("=" * 75)

    return summary_df


if __name__ == "__main__":
    run_full_classifier_evaluation()
