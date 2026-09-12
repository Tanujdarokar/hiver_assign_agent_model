"""
Phase 1 — Dataset Exploration
==============================
Uses: TNE-AI/customer-support-on-twitter-conversation (HuggingFace)
      794,335 pre-reconstructed conversations with company labels.

Columns:
  conversation_id  — unique thread hash
  company          — brand handle (e.g. AmazonHelp, SpotifyCares)
  conversation     — full thread as text "Customer: ... \nSupport: ..."
  summary          — mostly empty; ignored

Run:
    python explore_dataset.py

Output files (written to data/processed/):
    brand_summary.csv          — per-brand statistics
    sample_conversations.txt   — 3 sample threads for the top brands
"""

import re
import sys
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
import numpy as np

# ── 0. Bootstrap output dirs ──────────────────────────────────────────────────
ROOT = Path(__file__).parent
RAW_DIR  = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

CACHE_CSV = RAW_DIR / "twcs_conversations.csv"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load the dataset
# ─────────────────────────────────────────────────────────────────────────────
def load_dataset() -> pd.DataFrame:
    """
    Loads TNE-AI/customer-support-on-twitter-conversation from HuggingFace.
    Caches to data/raw/twcs_conversations.csv on first run.
    Columns after load: conversation_id, company, conversation
    """
    if CACHE_CSV.exists():
        print(f"Loading cached CSV from {CACHE_CSV} …")
        df = pd.read_csv(CACHE_CSV, low_memory=False)
        print(f"  Loaded {len(df):,} rows from cache.")
        return df

    print("Downloading TNE-AI/customer-support-on-twitter-conversation from HuggingFace …")
    try:
        from datasets import load_dataset as hf_load
        ds = hf_load("TNE-AI/customer-support-on-twitter-conversation", split="train")
        df = ds.to_pandas()
        # Keep only the columns we need
        df = df[["conversation_id", "company", "conversation"]].copy()
        df.to_csv(CACHE_CSV, index=False, encoding="utf-8")
        print(f"  Downloaded {len(df):,} conversations -> cached at {CACHE_CSV}")
        return df
    except Exception as e:
        sys.exit(f"\nERROR loading dataset: {e}\n"
                 "Please ensure you have an internet connection and `datasets` installed.\n")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Parse conversation text into structured turns
# ─────────────────────────────────────────────────────────────────────────────
def parse_turns(conversation_text: str) -> list[dict]:
    """
    The 'conversation' field looks like:
        Customer: <text>\nSupport: <text>\nCustomer: <text>\n...
    Returns a list of {'role': 'Customer'|'Support', 'text': str}
    """
    turns = []
    if not isinstance(conversation_text, str):
        return turns
    # Split on role prefix
    parts = re.split(r"(?=\b(?:Customer|Support):)", conversation_text.strip())
    for part in parts:
        part = part.strip()
        if part.startswith("Customer:"):
            text = part[len("Customer:"):].strip()
            turns.append({"role": "Customer", "text": text})
        elif part.startswith("Support:"):
            text = part[len("Support:"):].strip()
            turns.append({"role": "Support", "text": text})
    return turns


def add_turn_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Add useful per-conversation statistics."""
    df = df.copy()
    parsed = df["conversation"].apply(parse_turns)
    df["turns"]            = parsed
    df["n_turns"]          = parsed.apply(len)
    df["n_customer_turns"] = parsed.apply(lambda t: sum(1 for x in t if x["role"] == "Customer"))
    df["n_support_turns"]  = parsed.apply(lambda t: sum(1 for x in t if x["role"] == "Support"))
    df["has_support_reply"]= df["n_support_turns"] > 0

    # Average support-reply length (chars) — key quality signal
    def avg_support_len(turns):
        lens = [len(x["text"]) for x in turns if x["role"] == "Support"]
        return np.mean(lens) if lens else 0.0

    df["avg_support_len"] = parsed.apply(avg_support_len)

    # Noise flag: support reply is < 20 chars after stripping
    def has_noisy_reply(turns):
        replies = [x["text"].strip() for x in turns if x["role"] == "Support"]
        if not replies:
            return False
        return all(len(r) < 20 for r in replies)

    df["is_noisy"] = parsed.apply(has_noisy_reply)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Basic inspection
# ─────────────────────────────────────────────────────────────────────────────
def basic_inspection(df: pd.DataFrame) -> None:
    print("\n" + "═" * 70)
    print("BASIC INSPECTION")
    print("═" * 70)
    print(f"\nShape          : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print("\nColumns & dtypes:")
    print(df.dtypes.to_string())
    print("\nFirst 2 rows:")
    print(df[["conversation_id", "company", "conversation"]].head(2).to_string())
    print("\nMissing values:")
    miss = df[["conversation_id", "company", "conversation"]].isnull().sum()
    print(miss.to_string())
    dupes = df.duplicated(subset="conversation_id").sum()
    print(f"\nDuplicate conversation_ids: {dupes:,}")
    print(f"Unique companies          : {df['company'].nunique():,}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Per-brand statistics
# ─────────────────────────────────────────────────────────────────────────────
def compute_brand_stats(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "─" * 50)
    print("Computing per-brand statistics …")

    records = []
    for company, grp in df.groupby("company"):
        n_convs          = len(grp)
        has_reply        = grp["has_support_reply"].sum()
        pct_replied      = has_reply / max(n_convs, 1)
        avg_turns        = grp["n_turns"].mean()
        avg_sup_len      = grp["avg_support_len"].mean()

        noisy_count      = grp["is_noisy"].sum()
        noise_rate       = noisy_count / max(has_reply, 1)
        usable           = int(has_reply - noisy_count)

        # Average customer message length (proxy for complexity of issues)
        def avg_cust_len(turns):
            lens = [len(x["text"]) for x in turns if x["role"] == "Customer"]
            return np.mean(lens) if lens else 0.0

        avg_cust = grp["turns"].apply(avg_cust_len).mean()

        records.append({
            "company"               : company,
            "n_conversations"       : n_convs,
            "pct_with_support_reply": round(pct_replied * 100, 1),
            "avg_turns"             : round(avg_turns, 2),
            "avg_support_len_chars" : round(avg_sup_len, 1),
            "avg_customer_len_chars": round(avg_cust, 1),
            "noise_rate"            : round(noise_rate, 3),
            "usable_conversations"  : usable,
        })

    stats = pd.DataFrame(records)
    stats = stats.sort_values("n_conversations", ascending=False)
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Score and rank brands
# ─────────────────────────────────────────────────────────────────────────────
def score_and_recommend(stats: pd.DataFrame) -> pd.DataFrame:
    """
    Composite score (0–100):
      40% usable_conversations   — raw data volume
      25% avg_turns              — richer conversation context
      20% avg_support_len_chars  — more detailed replies = better retrieval
      15% (1 − noise_rate)       — cleaner data
    """
    def norm(col):
        mn, mx = col.min(), col.max()
        if mx == mn:
            return pd.Series([50.0] * len(col), index=col.index)
        return (col - mn) / (mx - mn) * 100

    s = stats.copy()
    s["score"] = (
        0.40 * norm(s["usable_conversations"])
        + 0.25 * norm(s["avg_turns"])
        + 0.20 * norm(s["avg_support_len_chars"])
        + 0.15 * norm(1 - s["noise_rate"])
    ).round(1)

    return s.sort_values("score", ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Sample conversations for qualitative inspection
# ─────────────────────────────────────────────────────────────────────────────
def sample_conversations(df: pd.DataFrame, companies: list, n_per_brand: int = 2) -> str:
    lines = []
    for company in companies:
        brand_df = df[df["company"] == company].head(n_per_brand)
        lines.append(f"\n{'═'*60}")
        lines.append(f"Brand: {company}")
        lines.append(f"{'═'*60}")
        for _, row in brand_df.iterrows():
            lines.append(f"\n  ── Conversation {row['conversation_id'][:12]}… ({row['n_turns']} turns) ──")
            for turn in row["turns"]:
                prefix = "  [CUSTOMER]" if turn["role"] == "Customer" else "  [SUPPORT] "
                lines.append(f"{prefix} {turn['text'][:200]}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── Load ──────────────────────────────────────────────────────────────────
    df = load_dataset()

    # ── Add turn statistics ───────────────────────────────────────────────────
    print("Parsing conversation turns …")
    df = add_turn_stats(df)

    # ── Inspect ───────────────────────────────────────────────────────────────
    basic_inspection(df)

    # ── Global stats ──────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("GLOBAL CONVERSATION STATISTICS")
    print("═" * 70)
    print(f"  Total conversations       : {len(df):,}")
    print(f"  With ≥1 support reply     : {df['has_support_reply'].sum():,} "
          f"({df['has_support_reply'].mean()*100:.1f}%)")
    print(f"  Unique brands             : {df['company'].nunique():,}")
    cs = df["n_turns"]
    print(f"\n  Turns per conversation:")
    print(f"    min={cs.min()}  median={cs.median():.0f}  mean={cs.mean():.1f}  max={cs.max()}")
    print(f"    1-turn (orphan): {(cs==1).sum():,} ({(cs==1).mean()*100:.1f}%)")

    # ── Per-brand stats ───────────────────────────────────────────────────────
    stats = compute_brand_stats(df)
    ranked = score_and_recommend(stats)

    # ── Display top 15 ────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("TOP 15 BRAND ACCOUNTS BY COMPOSITE SCORE")
    print("═" * 70)
    display_cols = [
        "company", "n_conversations", "usable_conversations",
        "avg_turns", "avg_support_len_chars", "noise_rate", "score",
    ]
    print(ranked.head(15)[display_cols].to_string(index=False))

    # ── Top-3 recommendations ─────────────────────────────────────────────────
    top3 = ranked.head(3)
    print("\n" + "═" * 70)
    print("RECOMMENDED TOP-3 CANDIDATE BRANDS")
    print("═" * 70)
    for rank, (_, row) in enumerate(top3.iterrows(), 1):
        print(f"\n  #{rank}  {row['company']}")
        print(f"       Conversations         : {row['n_conversations']:,}")
        print(f"       Usable conversations  : {row['usable_conversations']:,}")
        print(f"       Avg turns/conv        : {row['avg_turns']}")
        print(f"       Avg support reply len : {row['avg_support_len_chars']} chars")
        print(f"       Noise rate            : {row['noise_rate']:.1%}")
        print(f"       Composite score       : {row['score']}")

    # ── Sample conversations ───────────────────────────────────────────────────
    top3_companies = top3["company"].tolist()
    sample_text = sample_conversations(df, top3_companies, n_per_brand=2)
    sample_path = PROC_DIR / "sample_conversations.txt"
    sample_path.write_text(sample_text, encoding="utf-8")
    print(f"\nSample conversations written -> {sample_path}")
    print("\nPREVIEW (first 3000 chars):")
    print(sample_text[:3000])

    # ── Save outputs ──────────────────────────────────────────────────────────
    brand_summary_path = PROC_DIR / "brand_summary.csv"
    ranked.drop(columns=["turns"], errors="ignore").to_csv(brand_summary_path, index=False, encoding="utf-8")
    print(f"\nFull brand summary -> {brand_summary_path}")

    print("\n" + "═" * 70)
    print("PHASE 1 COMPLETE")
    print("─" * 70)
    print("Review the top-3 above and tell me which brand you'd like to use.")
    print("Also check data/processed/sample_conversations.txt for qualitative review.")
    print("═" * 70)


if __name__ == "__main__":
    main()
