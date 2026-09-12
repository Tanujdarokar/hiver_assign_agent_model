"""
Phase 20 — Unit and Integration Test Suite
==========================================
Verifies Data Preprocessing, Intent Classifier, Historical Retrieval, Escalation Engine,
and Agent Output Schema under pytest.
"""

import pytest
import sys
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.data_processing import clean_tweet_text, parse_turns
from src.intent_classifier import IntentClassifier
from src.retriever import HistoricalRetriever
from src.escalation import EscalationEngine
from src.agent import AppleSupportAgent


def test_clean_tweet_text():
    raw = "Customer: @AppleSupport my iPhone 6 battery drops quickly! http://t.co/abc ^AB"
    cleaned = clean_tweet_text(raw)
    assert "@AppleSupport" not in cleaned
    assert "^AB" not in cleaned
    assert "[LINK]" in cleaned
    assert "battery drops quickly" in cleaned


def test_parse_turns():
    conv = "Customer: My screen is cracked.\nSupport: We can help with repair options. [LINK]"
    turns = parse_turns(conv)
    assert len(turns) == 2
    assert turns[0]["role"] == "Customer"
    assert turns[1]["role"] == "Support"


def test_intent_classifier():
    clf = IntentClassifier()
    clf.load()
    res = clf.predict("My battery is draining fast on my iPhone 7")
    assert "intent" in res
    assert "confidence" in res
    assert res["intent"] == "battery_and_power"
    assert 0.0 <= res["confidence"] <= 1.0


def test_retriever():
    retriever = HistoricalRetriever(top_k=2)
    retriever.load()
    results = retriever.retrieve("iOS 11 update stuck on Apple logo", top_k=2)
    assert len(results) == 2
    assert "conversation_id" in results[0]
    assert "customer_message" in results[0]
    assert "support_response" in results[0]
    assert "similarity_score" in results[0]


def test_escalation_engine():
    engine = EscalationEngine()

    # 1. High risk intent -> ESCALATE
    esc_passcode = engine.evaluate(
        message="I forgot my passcode and locked my iPhone",
        intent="apple_id_and_security",
        confidence=0.95,
        top_similarity=0.50
    )
    assert esc_passcode["decision"] == "ESCALATE"

    # 2. Short message -> ESCALATE
    esc_short = engine.evaluate(
        message="Help",
        intent="general_inquiry_or_other",
        confidence=0.80,
        top_similarity=0.40
    )
    assert esc_short["decision"] == "ESCALATE"

    # 3. High confidence + safe intent -> AUTO
    auto_res = engine.evaluate(
        message="My battery is draining quickly after iOS update",
        intent="battery_and_power",
        confidence=0.90,
        top_similarity=0.45
    )
    assert auto_res["decision"] == "AUTO"


def test_agent_schema():
    agent = AppleSupportAgent()
    res = agent.handle("How do I fix battery drain on iOS 11?")

    required_keys = ["intent", "confidence", "decision", "reason", "reply", "retrieved_examples"]
    for key in required_keys:
        assert key in res

    assert res["decision"] in ["AUTO", "ESCALATE"]
    assert isinstance(res["confidence"], float)
    assert len(res["retrieved_examples"]) > 0
