"""
Phase 7 — Real Intent Classifier Module
========================================
Lightweight, reproducible, embedding/TF-IDF based intent classifier.
Trained on processed historical AppleSupport data with STRICT Golden Set exclusion.
Returns predicted intent and confidence score.
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_CSV = ROOT / "data" / "processed" / "apple_support_processed.csv"
GOLDEN_CSV = ROOT / "data" / "golden" / "golden_set.csv"
MODEL_PATH = ROOT / "data" / "processed" / "intent_classifier.pkl"


class IntentClassifier:
    """Production Intent Classifier with probability calibration."""

    def __init__(self):
        self.model = None
        self.is_trained = False

    def train(self, sample_size: int = 10000):
        """
        Trains classifier on historical non-Golden data to prevent data leakage.
        """
        print("Training Intent Classifier ...")
        proc_df = pd.read_csv(PROCESSED_CSV)
        golden_df = pd.read_csv(GOLDEN_CSV)

        # 1. Exclude Golden Set conversations to prevent leakage
        golden_ids = set(golden_df["conversation_id"].astype(str))
        train_df = proc_df[~proc_df["conversation_id"].astype(str).isin(golden_ids)].copy()

        # Import expert labeling function
        import sys
        sys.path.append(str(ROOT))
        from create_golden_set import label_message_intent
        labels = train_df["customer_message"].apply(label_message_intent)
        train_df["intent"] = [l[0] for l in labels]

        # Sample for fast, balanced training if dataset is large
        if len(train_df) > sample_size:
            sampled_dfs = []
            for intent_name, group in train_df.groupby("intent"):
                sampled_dfs.append(group.sample(min(len(group), sample_size // 11), random_state=42))
            train_df = pd.concat(sampled_dfs, ignore_index=True)

        X_train = train_df["customer_message"].tolist()
        y_train = train_df["intent"].tolist()

        # Pipeline: TF-IDF n-grams + Logistic Regression
        self.model = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1000, C=2.0, class_weight="balanced"))
        ])

        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Save model
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self.model, f)
        print(f"Classifier trained on {len(X_train)} samples -> Saved to {MODEL_PATH}")

    def load(self):
        """Loads pre-trained model from disk."""
        if MODEL_PATH.exists():
            with open(MODEL_PATH, "rb") as f:
                self.model = pickle.load(f)
            self.is_trained = True
        else:
            self.train()

    def predict(self, text: str) -> dict:
        """
        Predicts intent and confidence score for a customer message.
        Returns:
            {"intent": str, "confidence": float}
        """
        if not self.is_trained or self.model is None:
            self.load()

        probs = self.model.predict_proba([text])[0]
        classes = self.model.classes_
        top_idx = int(np.argmax(probs))

        predicted_intent = str(classes[top_idx])
        confidence = float(probs[top_idx])

        return {
            "intent": predicted_intent,
            "confidence": round(confidence, 4)
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """Predicts intent and confidence for a list of text messages."""
        if not self.is_trained or self.model is None:
            self.load()

        probs_matrix = self.model.predict_proba(texts)
        classes = self.model.classes_

        results = []
        for probs in probs_matrix:
            top_idx = int(np.argmax(probs))
            results.append({
                "intent": str(classes[top_idx]),
                "confidence": round(float(probs[top_idx]), 4)
            })
        return results


if __name__ == "__main__":
    clf = IntentClassifier()
    clf.train()

    # Quick test
    sample_msg = "My iPhone 7 battery is draining super fast after updating to iOS 11"
    res = clf.predict(sample_msg)
    print(f"\nTest Prediction for: '{sample_msg}'")
    print(f"Result: {res}")
