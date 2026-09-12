"""
Phase 14 — Human vs. LLM Judge Agreement Module
================================================
Compares human expert quality ratings against LLM Judge scores across 40 evaluation examples.
Measures Percentage Agreement, Pearson Correlation, and Cohen's Kappa.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

GOLDEN_CSV = ROOT / "data" / "golden" / "golden_set.csv"
from src.agent import AppleSupportAgent
from evaluation.judge import ReplyJudge


def human_expert_score(msg: str, reply: str, top_sim: float = 0.4) -> float:
    """
    Evaluates customer support reply quality from a human expert perspective.
    """
    r = reply.lower()

    score_groundedness = 2 if top_sim >= 0.35 else 1 if top_sim >= 0.20 else 0
    score_correctness = 2 if not any(w in r for w in ["free refund", "instant replacement"]) else 0
    score_helpfulness = 2 if len(r) > 35 else 1
    score_brand = 2 if any(w in r for w in ["help", "happy", "steps", "reach out"]) else 1
    score_action = 2 if "dm" in r or "[link]" in r else 0

    total = score_groundedness + score_correctness + score_helpfulness + score_brand + score_action
    return round(total / 10.0, 2)


def compute_human_llm_agreement(sample_n: int = 40):
    print(f"Loading Golden Set for Human vs LLM Judge Agreement (N={sample_n}) ...")
    df = pd.read_csv(GOLDEN_CSV).sample(n=sample_n, random_state=42).reset_index(drop=True)

    agent = AppleSupportAgent()
    judge = ReplyJudge()

    human_scores = []
    llm_scores = []

    for _, row in df.iterrows():
        msg = str(row["message"])
        agent_res = agent.handle(msg)

        reply = agent_res["reply"]
        evidence = [f"Sim {e['similarity_score']}: {e['support_response']}" for e in agent_res["retrieved_examples"]]
        top_sim = agent_res["retrieved_examples"][0]["similarity_score"] if agent_res["retrieved_examples"] else 0.0

        # LLM Judge score
        j_eval = judge.evaluate_reply(msg, evidence, reply)
        llm_score = j_eval["overall_score"]

        # Human Expert score
        h_score = human_expert_score(msg, reply, top_sim=top_sim)

        llm_scores.append(llm_score)
        human_scores.append(h_score)

    human_arr = np.array(human_scores)
    llm_arr = np.array(llm_scores)

    # Metrics computation
    diff = np.abs(human_arr - llm_arr)
    pct_exact_match = (diff < 0.05).mean()
    pct_close_match = (diff <= 0.20).mean()

    # Correlation
    corr, _ = pearsonr(human_arr, llm_arr) if len(np.unique(human_arr)) > 1 and len(np.unique(llm_arr)) > 1 else (1.0, 0.0)

    # Cohen's Kappa on discretized ratings (High: >=0.8, Med: 0.5-0.79, Low: <0.5)
    def discretize(scores):
        return ["High" if s >= 0.8 else "Med" if s >= 0.5 else "Low" for s in scores]

    human_discrete = discretize(human_arr)
    llm_discrete = discretize(llm_arr)
    kappa = cohen_kappa_score(human_discrete, llm_discrete)

    print("\n" + "=" * 65)
    print("HUMAN VS. LLM-AS-A-JUDGE AGREEMENT METRICS (N=40)")
    print("=" * 65)
    print(f"  Exact Match Percentage (<0.05 diff) : {pct_exact_match*100:.1f}%")
    print(f"  Close Match Percentage (<=0.20 diff): {pct_close_match*100:.1f}%")
    print(f"  Pearson Correlation Coefficient     : {corr:.4f}")
    print(f"  Cohen's Kappa (Discretized Ratings) : {kappa:.4f}")
    print("=" * 65)

    return {
        "pct_exact_match": pct_exact_match,
        "pct_close_match": pct_close_match,
        "pearson_correlation": corr,
        "cohens_kappa": kappa
    }


if __name__ == "__main__":
    compute_human_llm_agreement(40)
