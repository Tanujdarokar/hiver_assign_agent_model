"""
Phase 11 — Final AppleSupport AI Customer Support Agent
======================================================
Unified end-to-end Support Agent orchestrating Intent Classification,
Historical Retrieval, Grounded Reply Generation, and Escalation Decisions.
"""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.intent_classifier import IntentClassifier
from src.retriever import HistoricalRetriever
from src.reply_generator import ReplyGenerator
from src.escalation import EscalationEngine


class AppleSupportAgent:
    """End-to-End AI Support Agent for AppleSupport."""

    def __init__(self, top_k: int = 3):
        self.top_k = top_k
        self.classifier = IntentClassifier()
        self.classifier.load()
        self.retriever = HistoricalRetriever(top_k=top_k)
        self.retriever.load()
        self.reply_generator = ReplyGenerator()
        self.escalation_engine = EscalationEngine()

    def handle(self, message: str) -> dict:
        """
        Main entrance point for customer queries.
        Processes incoming message through the complete agent pipeline.
        Returns structured dictionary.
        """
        # 1. Classify Intent & Confidence
        clf_res = self.classifier.predict(message)
        intent = clf_res["intent"]
        confidence = clf_res["confidence"]

        # 2. Retrieve Historical Support Resolutions
        retrieved_examples = self.retriever.retrieve(message, top_k=self.top_k)
        top_similarity = retrieved_examples[0]["similarity_score"] if retrieved_examples else 0.0

        # 3. Evaluate Escalation Decision
        escalation_res = self.escalation_engine.evaluate(
            message=message,
            intent=intent,
            confidence=confidence,
            top_similarity=top_similarity
        )
        decision = escalation_res["decision"]
        reason = escalation_res["reason"]

        # 4. Generate Grounded Reply
        reply_res = self.reply_generator.generate_reply(
            customer_message=message,
            intent=intent,
            retrieved_examples=retrieved_examples
        )
        reply = reply_res["reply"]

        return {
            "intent": intent,
            "confidence": confidence,
            "decision": decision,
            "reason": reason,
            "reply": reply,
            "retrieved_examples": retrieved_examples
        }


if __name__ == "__main__":
    agent = AppleSupportAgent()

    sample_queries = [
        "My battery drops from 80% to 10% in less than an hour on iOS 11",
        "I forgot my Apple ID password and my account is completely locked out",
        "Why was I charged $9.99 on my credit card without my authorization?",
        "Hi"
    ]

    print("\n" + "=" * 70)
    print("APPLE SUPPORT AI AGENT INTERACTIVE DEMONSTRATION")
    print("=" * 70)
    for q in sample_queries:
        print(f"\n[CUSTOMER QUERY]: {q}")
        res = agent.handle(q)
        print(json.dumps(res, indent=2))
        print("-" * 70)
