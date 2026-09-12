"""
Phase 10 — Escalation System Module
====================================
Determines whether an incoming customer inquiry can be AUTO-handled or must be ESCALATED
to a human Apple support agent, providing an explicit, auditable reason.
Evaluates safety metrics (Precision, Recall, False Auto-Handling Rate) on the Golden Set.
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLDEN_CSV = ROOT / "data" / "golden" / "golden_set.csv"

# High-risk intent categories requiring mandatory human agent handling
HIGH_RISK_INTENTS = {
    "apple_id_and_security",    # Account lockout, password reset, 2FA
    "apple_pay_and_billing",    # Financial transactions, unauthorized charges, refunds
    "hardware_and_display"      # Physical screen damage, hardware replacement, Genius Bar
}


class EscalationEngine:
    """Escalation Decision Engine for AppleSupport AI Agent."""

    def __init__(self, confidence_threshold: float = 0.60, min_similarity_threshold: float = 0.25):
        self.confidence_threshold = confidence_threshold
        self.min_similarity_threshold = min_similarity_threshold

    def evaluate(self, message: str, intent: str, confidence: float, top_similarity: float) -> dict:
        """
        Evaluates message and classification parameters to make an AUTO vs ESCALATE decision.
        Returns:
            {"decision": "AUTO" | "ESCALATE", "reason": str}
        """
        # 1. Short or ambiguous messages
        if len(message.strip()) < 15:
            return {
                "decision": "ESCALATE",
                "reason": f"Message is too short ({len(message.strip())} chars) or lacks actionable technical details."
            }

        # 2. Sensitive / High-Risk Intents requiring human verification/action
        if intent in HIGH_RISK_INTENTS:
            return {
                "decision": "ESCALATE",
                "reason": f"High-risk intent '{intent}' requires human agent verification or account/hardware authorization."
            }

        # 3. Low intent classification confidence
        if confidence < self.confidence_threshold:
            return {
                "decision": "ESCALATE",
                "reason": f"Low intent classification confidence ({confidence:.2f} < {self.confidence_threshold})."
            }

        # 4. Low retrieval grounding / insufficient historical precedent
        if top_similarity < self.min_similarity_threshold:
            return {
                "decision": "ESCALATE",
                "reason": f"Insufficient historical precedent (top similarity {top_similarity:.2f} < {self.min_similarity_threshold})."
            }

        # 5. Default: Safe to auto-handle
        return {
            "decision": "AUTO",
            "reason": f"High intent confidence ({confidence:.2f}) and strong historical resolution grounding ({top_similarity:.2f})."
        }


def evaluate_escalation_system():
    """
    Evaluates Escalation Engine performance against Golden Set ground truth expectations.
    Calculates Precision, Recall, False Auto-Handling Rate, and False Escalation Rate.
    """
    print(f"Loading Golden Set from {GOLDEN_CSV} ...")
    df = pd.read_csv(GOLDEN_CSV)

    import sys
    sys.path.append(str(ROOT))
    from src.intent_classifier import IntentClassifier
    from src.retriever import HistoricalRetriever

    clf = IntentClassifier()
    clf.load()
    retriever = HistoricalRetriever()
    retriever.load()
    engine = EscalationEngine()

    results = []
    for _, row in df.iterrows():
        msg = str(row["message"])
        expected = str(row["expected_action"])

        pred_clf = clf.predict(msg)
        retrieved = retriever.retrieve(msg, top_k=1)
        top_sim = retrieved[0]["similarity_score"] if retrieved else 0.0

        esc_res = engine.evaluate(
            message=msg,
            intent=pred_clf["intent"],
            confidence=pred_clf["confidence"],
            top_similarity=top_sim
        )
        predicted_action = esc_res["decision"]

        results.append({
            "expected": expected,
            "predicted": predicted_action
        })

    eval_df = pd.DataFrame(results)

    # Metrics
    tp = int(((eval_df["predicted"] == "ESCALATE") & (eval_df["expected"] == "ESCALATE")).sum())
    fp = int(((eval_df["predicted"] == "ESCALATE") & (eval_df["expected"] == "AUTO")).sum())
    fn = int(((eval_df["predicted"] == "AUTO") & (eval_df["expected"] == "ESCALATE")).sum())
    tn = int(((eval_df["predicted"] == "AUTO") & (eval_df["expected"] == "AUTO")).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    false_auto_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # Harmful: needed escalation but auto-handled!
    false_esc_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0   # Inefficient: auto-handlable but escalated!

    print("\n" + "=" * 60)
    print("ESCALATION SYSTEM EVALUATION (Golden Set N=200)")
    print("=" * 60)
    print(f"  Total Golden Examples      : {len(eval_df)}")
    print(f"  True Positives (Escalated) : {tp}")
    print(f"  True Negatives (Auto)      : {tn}")
    print(f"  False Escalations (FP)     : {fp}")
    print(f"  False Auto-Handled (FN)    : {fn}")
    print("-" * 60)
    print(f"  Escalation Precision       : {precision:.4f} ({precision*100:.1f}%)")
    print(f"  Escalation Recall          : {recall:.4f} ({recall*100:.1f}%)")
    print(f"  False Auto-Handling Rate   : {false_auto_rate:.4f} ({false_auto_rate*100:.1f}%) [CRITICAL SAFETY METRIC]")
    print(f"  False Escalation Rate      : {false_esc_rate:.4f} ({false_esc_rate*100:.1f}%) [EFFICIENCY METRIC]")
    print("=" * 60)

    return {
        "precision": precision,
        "recall": recall,
        "false_auto_rate": false_auto_rate,
        "false_esc_rate": false_esc_rate
    }


if __name__ == "__main__":
    evaluate_escalation_system()
