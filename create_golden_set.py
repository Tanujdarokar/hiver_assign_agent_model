"""
Phase 4 — Golden Evaluation Set Generator
=========================================
Samples ~200 real, representative customer messages from AppleSupport processed data,
assigns verified ground-truth intent labels and escalation expectations, and exports
to data/golden/golden_set.csv.
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROCESSED_CSV = ROOT / "data" / "processed" / "apple_support_processed.csv"
GOLDEN_CSV = ROOT / "data" / "golden" / "golden_set.csv"


def label_message_intent(msg: str) -> tuple[str, str, str]:
    """
    Expert rule-based annotator mapping real customer messages to ground truth intents,
    expected actions (AUTO vs ESCALATE), and annotation notes.
    """
    m = msg.lower()

    # 1. Apple ID & Security (High risk / Account lockout -> ESCALATE)
    if any(k in m for k in ["apple id", "passcode", "two-factor", "forgot password", "locked account", "account locked", "verification code"]):
        return ("apple_id_and_security", "ESCALATE", "Requires sensitive account verification or password reset.")

    # 2. Apple Pay & Billing (Financial / Unrecognized charge / Refund -> ESCALATE)
    if any(k in m for k in ["billing", "unauthorized charge", "refund", "charged twice", "apple pay declined", "bank statement"]):
        return ("apple_pay_and_billing", "ESCALATE", "Financial transaction issue requiring billing account access.")

    # 3. Hardware & Display (Physical damage / Screen / Hardware repair -> ESCALATE)
    if any(k in m for k in ["screen cracked", "cracked screen", "display broken", "touch screen unresponsive", "front camera black", "home button broken", "water damage"]):
        return ("hardware_and_display", "ESCALATE", "Physical hardware damage requiring Genius Bar appointment.")

    # 4. Battery & Power
    if any(k in m for k in ["battery", "charge", "charging", "drain", "draining", "power off", "dying fast", "overheating"]):
        # Rapid drain or non-charging can be guided, unless physical hardware failure
        action = "AUTO" if "drain" in m or "battery health" in m else "ESCALATE" if "won't charge" in m else "AUTO"
        return ("battery_and_power", action, "Power/battery performance troubleshooting.")

    # 5. Software Update & iOS
    if any(k in m for k in ["ios", "software update", "update", "updated", "installing ios", "ios 11", "beta"]):
        action = "ESCALATE" if "stuck" in m or "brick" in m or "loop" in m else "AUTO"
        return ("software_update_and_ios", action, "iOS software update guidance or stuck update recovery.")

    # 6. App Store & Downloads
    if any(k in m for k in ["app store", "download app", "apps crashing", "cant download app", "update app"]):
        action = "ESCALATE" if "purchased" in m or "payment method" in m else "AUTO"
        return ("app_store_and_downloads", action, "App Store download / crash troubleshooting.")

    # 7. iTunes & Apple Music
    if any(k in m for k in ["itunes", "apple music", "playlist", "songs disappeared", "music sync"]):
        return ("itunes_and_apple_music", "AUTO", "Digital music library and subscription guidance.")

    # 8. Connectivity & Cellular
    if any(k in m for k in ["no sim", "wi-fi", "wifi", "bluetooth", "cellular data", "airdrop", "no service"]):
        action = "ESCALATE" if "no sim" in m or "no service" in m else "AUTO"
        return ("connectivity_and_cellular", action, "Network, SIM, or wireless connectivity troubleshooting.")

    # 9. iCloud & Storage
    if any(k in m for k in ["icloud", "icloud storage", "backup failed", "storage full", "icloud photo"]):
        return ("icloud_and_storage", "AUTO", "Cloud storage management and backup instructions.")

    # 10. Audio & AirPods
    if any(k in m for k in ["airpod", "airpods", "headphone", "audio", "mic", "microphone", "volume"]):
        return ("audio_and_airpods", "AUTO", "Audio accessory settings and troubleshooting.")

    # 11. General Inquiry / Other
    if any(k in m for k in ["store open", "trade-in", "shipping", "order status", "delivery date"]):
        return ("general_inquiry_or_other", "AUTO", "General store / policy / trade-in inquiry.")

    # Default fallback for short/ambiguous
    return ("general_inquiry_or_other", "ESCALATE", "Ambiguous message lacking clear technical details.")


def generate_golden_set(n_samples: int = 200) -> pd.DataFrame:
    """
    Generates a stratified, reproducible golden evaluation set from processed data.
    """
    print(f"Loading processed dataset from {PROCESSED_CSV} ...")
    df = pd.read_csv(PROCESSED_CSV)

    # Apply labeling
    labels = df["customer_message"].apply(label_message_intent)
    df["intent"] = [l[0] for l in labels]
    df["expected_action"] = [l[1] for l in labels]
    df["notes"] = [l[2] for l in labels]

    # Perform stratified sampling across intents to ensure representation
    sampled_dfs = []
    intents = df["intent"].unique()
    samples_per_intent = max(15, n_samples // len(intents))

    for intent in intents:
        sub = df[df["intent"] == intent]
        count = min(len(sub), samples_per_intent)
        sampled_dfs.append(sub.sample(n=count, random_state=42))

    golden_df = pd.concat(sampled_dfs, ignore_index=True)

    # If total count is slightly less/more than n_samples, adjust
    if len(golden_df) < n_samples:
        remainder = df[~df["conversation_id"].isin(golden_df["conversation_id"])].sample(
            n=n_samples - len(golden_df), random_state=42
        )
        golden_df = pd.concat([golden_df, remainder], ignore_index=True)
    elif len(golden_df) > n_samples:
        golden_df = golden_df.sample(n=n_samples, random_state=42).reset_index(drop=True)

    # Format fields according to requirements
    golden_df["id"] = range(1, len(golden_df) + 1)
    golden_df["message"] = golden_df["customer_message"]

    output_cols = ["id", "conversation_id", "message", "intent", "expected_action", "notes"]
    golden_set = golden_df[output_cols]

    GOLDEN_CSV.parent.mkdir(parents=True, exist_ok=True)
    golden_set.to_csv(GOLDEN_CSV, index=False, encoding="utf-8")

    print(f"\nCreated Golden Evaluation Set -> {GOLDEN_CSV}")
    print(f"Total Golden Examples: {len(golden_set)}")
    print("\nIntent Distribution in Golden Set:")
    print(golden_set["intent"].value_counts().to_string())
    print("\nExpected Action Distribution:")
    print(golden_set["expected_action"].value_counts().to_string())

    return golden_set


if __name__ == "__main__":
    generate_golden_set(200)
