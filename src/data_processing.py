"""
Phase 2 — Data Preprocessing Module
====================================
Filters dataset for AppleSupport, reconstructs clean conversation pairs
(Customer Query -> Support Response), cleans Twitter artifacts (handles, URLs),
removes noisy/orphan/short records, and exports clean CSV.
"""

import re
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data" / "raw" / "twcs_conversations.csv"
PROCESSED_CSV = ROOT / "data" / "processed" / "apple_support_processed.csv"


def clean_tweet_text(text: str) -> str:
    """
    Cleans Twitter artifacts while preserving core semantic meaning:
    - Normalizes URLs (e.g. http://t.co/... -> [LINK])
    - Strips user handles (e.g. @115712, @AppleSupport)
    - Strips agent signature tags (e.g. ^AB, ^JM)
    - Replaces excessive whitespace/newlines with a single space.
    """
    if not isinstance(text, str):
        return ""
    # Strip agent initials at end of tweet (e.g. ^AB, ^JM, ^DM)
    text = re.sub(r"\s*\^[A-Z0-9]{1,3}\b", "", text)
    # Replace URLs
    text = re.sub(r"https?://\S+|www\.\S+", "[LINK]", text)
    # Replace numeric user handles (@115712) or brand handles (@AppleSupport)
    text = re.sub(r"@\w+", "", text)
    # Collapse multiple spaces and newlines
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_turns(conversation_text: str) -> list[dict]:
    """
    Parses 'Customer: ... \nSupport: ...' into structured turns.
    """
    turns = []
    if not isinstance(conversation_text, str):
        return turns
    parts = re.split(r"(?=\b(?:Customer|Support):)", conversation_text.strip())
    for part in parts:
        part = part.strip()
        if part.startswith("Customer:"):
            raw_text = part[len("Customer:"):].strip()
            turns.append({"role": "Customer", "raw": raw_text, "clean": clean_tweet_text(raw_text)})
        elif part.startswith("Support:"):
            raw_text = part[len("Support:"):].strip()
            turns.append({"role": "Support", "raw": raw_text, "clean": clean_tweet_text(raw_text)})
    return turns


def preprocess_brand_data(brand_name: str = "AppleSupport", min_cust_len: int = 15, min_sup_len: int = 20) -> pd.DataFrame:
    """
    Loads raw Twitter dataset, filters to brand_name, pairs Customer initial query
    with Support reply, applies quality filters, and returns a clean DataFrame.
    """
    print(f"Loading raw dataset from {RAW_CSV} ...")
    df = pd.read_csv(RAW_CSV, low_memory=False)

    # 1. Filter by company
    brand_df = df[df["company"] == brand_name].copy()
    print(f"Total conversations for '{brand_name}': {len(brand_df):,}")

    records = []
    for _, row in brand_df.iterrows():
        conv_id = str(row["conversation_id"])
        raw_conv = str(row["conversation"])
        turns = parse_turns(raw_conv)

        # Find first customer turn and first support reply
        cust_turns = [t for t in turns if t["role"] == "Customer"]
        sup_turns = [t for t in turns if t["role"] == "Support"]

        if not cust_turns or not sup_turns:
            continue

        first_cust = cust_turns[0]["clean"]
        first_sup = sup_turns[0]["clean"]

        # Quality filters: length checks
        if len(first_cust) < min_cust_len:
            continue
        if len(first_sup) < min_sup_len:
            continue

        # Ignore boilerplate generic replies like "Please DM us" with no substance
        if first_sup.lower().strip() in ["please dm us.", "dm us for help.", "send us a dm."]:
            continue

        records.append({
            "conversation_id": conv_id,
            "customer_message": first_cust,
            "support_response": first_sup,
            "customer_raw": cust_turns[0]["raw"],
            "support_raw": sup_turns[0]["raw"],
            "n_turns": len(turns)
        })

    clean_df = pd.DataFrame(records)
    print(f"Preprocessed & filtered records for '{brand_name}': {len(clean_df):,}")

    # Save output
    PROCESSED_CSV.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(PROCESSED_CSV, index=False, encoding="utf-8")
    print(f"Saved processed dataset -> {PROCESSED_CSV}")
    return clean_df


if __name__ == "__main__":
    df = preprocess_brand_data("AppleSupport")
    print("\nPreview of preprocessed AppleSupport conversations:")
    print(df[["conversation_id", "customer_message", "support_response"]].head(5).to_string())
