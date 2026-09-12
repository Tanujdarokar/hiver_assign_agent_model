"""
Phase 13 — LLM-as-a-Judge Evaluation Module
============================================
Automated LLM-as-a-Judge system that evaluates generated replies across 5 dimensions
(0-2 scale): Groundedness, Correctness, Helpfulness, Brand/Tone Consistency, Actionability.
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class ReplyJudge:
    """LLM-as-a-Judge for evaluating customer support reply quality."""

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")

    def evaluate_reply(self, customer_message: str, retrieved_evidence: list[str], generated_reply: str) -> dict:
        """
        Evaluates generated reply on a 0-2 scale across 5 core dimensions.
        Returns JSON containing sub-scores and overall quality score (0.0 - 1.0).
        """
        evidence_str = "\n".join(retrieved_evidence) if retrieved_evidence else "None provided."

        # 1. If LLM API available
        if self.api_key and len(self.api_key) > 5:
            try:
                import openai
                client = openai.OpenAI(api_key=self.api_key)
                prompt = f"""You are an expert AI Support Quality Auditor evaluating an automated AppleSupport response.

Customer Message: "{customer_message}"
Retrieved Historical Evidence:
{evidence_str}

Generated Reply: "{generated_reply}"

EVALUATION RUBRIC (Score 0, 1, or 2 for each):
1. groundedness: Is the reply strictly based on the retrieved evidence without inventing policies or false promises?
2. correctness: Is the technical advice accurate and safe for Apple devices?
3. helpfulness: Does the reply directly address the customer's issue?
4. brand_tone: Does it maintain a professional, empathetic, official AppleSupport brand tone?
5. actionability: Does it provide clear, actionable next steps (e.g., check settings, DM details, link)?

Output ONLY a JSON object with this structure:
{{
  "groundedness": 0-2,
  "correctness": 0-2,
  "helpfulness": 0-2,
  "brand_tone": 0-2,
  "actionability": 0-2,
  "reasoning": "short explanation"
}}"""

                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    max_tokens=200,
                    temperature=0.0
                )
                res = json.loads(response.choices[0].message.content)
                total = res["groundedness"] + res["correctness"] + res["helpfulness"] + res["brand_tone"] + res["actionability"]
                res["overall_score"] = round(total / 10.0, 2)
                return res
            except Exception as e:
                print(f"LLM Judge call failed/skipped: {e}. Using rule-based judge fallback.")

        # 2. Rule-Based Quality Heuristic Judge Fallback
        top_sim = 0.0
        if retrieved_evidence:
            try:
                import re
                m = re.search(r"Sim\s*([0-9\.]+)", str(retrieved_evidence[0]))
                if m:
                    top_sim = float(m.group(1))
            except Exception:
                top_sim = 0.35

        score_groundedness = 2 if top_sim >= 0.35 else 1 if top_sim >= 0.20 else 0
        score_correctness = 2 if not any(w in generated_reply.lower() for w in ["free refund", "instant replacement", "guaranteed delivery"]) else 0
        score_helpfulness = 2 if len(generated_reply) > 40 else 1
        score_brand = 2 if any(w in generated_reply.lower() for w in ["help", "happy", "steps", "reach out"]) else 1
        score_action = 2 if "dm" in generated_reply.lower() or "[link]" in generated_reply.lower() else 0

        total = score_groundedness + score_correctness + score_helpfulness + score_brand + score_action
        return {
            "groundedness": score_groundedness,
            "correctness": score_correctness,
            "helpfulness": score_helpfulness,
            "brand_tone": score_brand,
            "actionability": score_action,
            "overall_score": round(total / 10.0, 2),
            "reasoning": f"Evaluated via rule-based quality heuristics (top similarity: {top_sim:.2f})."
        }


if __name__ == "__main__":
    judge = ReplyJudge()
    msg = "My battery is dying fast on iOS 11"
    evidence = ["Historical Case: We know how important battery life is. Follow these tips: [LINK]"]
    reply = "We'd love to help troubleshoot your battery performance. Check your battery health settings or review these steps: [LINK] DM us if the issue persists!"

    eval_res = judge.evaluate_reply(msg, evidence, reply)
    print("\nLLM-as-a-Judge Evaluation Output:")
    print(json.dumps(eval_res, indent=2))
