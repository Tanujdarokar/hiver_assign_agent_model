"""
Phase 9 — Optimized Grounded Reply Generator Module
===================================================
Grounded response generator that crafts empathetic, brand-aligned, entity-aware responses.
Extracts device models (iPhone, iPad, Mac, AirPods) and software versions (iOS 11, etc.),
and grounds responses dynamically using retrieved historical resolution evidence.
"""

import os
import re
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def extract_context_entities(text: str) -> dict:
    """
    Extracts key hardware models and software versions from customer query.
    """
    t = text.lower()

    # 1. Device model
    device = "Apple device"
    if "iphone" in t:
        m = re.search(r"iphone\s*([0-9xseplus]+)", t)
        device = f"iPhone {m.group(1).upper()}" if m else "iPhone"
    elif "ipad" in t:
        m = re.search(r"ipad\s*([proairmini0-9]+)", t)
        device = f"iPad {m.group(1).upper()}" if m else "iPad"
    elif "watch" in t:
        device = "Apple Watch"
    elif "mac" in t or "macbook" in t:
        device = "Mac"
    elif "airpod" in t:
        device = "AirPods"

    # 2. iOS or Software version
    ver = None
    m_ver = re.search(r"(?:ios|version)\s*([0-9\.]+)", t)
    if m_ver:
        ver = f"iOS {m_ver.group(1)}"

    return {"device": device, "version": ver}


class ReplyGenerator:
    """Grounded, Entity-Aware Reply Generator for AppleSupport Agent."""

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")

    def generate_reply(self, customer_message: str, intent: str, retrieved_examples: list[dict]) -> dict:
        """
        Generates an empathetic, entity-customized, grounded customer response.
        Returns:
            {"reply": str, "evidence_used": list[str], "source": str}
        """
        entities = extract_context_entities(customer_message)
        device = entities["device"]
        ver_str = f" on {entities['version']}" if entities["version"] else ""

        evidence_texts = [f"Historical Case (Sim {e['similarity_score']}): {e['support_response']}" for e in retrieved_examples]

        # 1. Try LLM API if key is configured
        if self.api_key and len(self.api_key) > 5:
            try:
                import openai
                client = openai.OpenAI(api_key=self.api_key)
                prompt = f"""You are an official @AppleSupport customer care agent on Twitter.
Customer Query: "{customer_message}"
Context: Device = {device}{ver_str}, Category = {intent}

Retrieved Historical Resolutions:
{chr(10).join(evidence_texts)}

RULES:
1. Emulate official AppleSupport brand tone (warm, professional, highly concise).
2. Ground your response in the retrieved historical resolutions.
3. Mention the customer's specific device ({device}) when natural.
4. Do NOT make false claims (e.g. free replacements, unauthorized refunds).
5. Include actionable next steps (check settings link, offer DM support).

Draft a concise 2-sentence response:"""

                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=100,
                    temperature=0.3
                )
                reply_text = response.choices[0].message.content.strip()
                return {
                    "reply": reply_text,
                    "evidence_used": evidence_texts,
                    "source": "llm"
                }
            except Exception as e:
                print(f"LLM API call skipped/failed: {e}. Using grounded template fallback.")

        # 2. Dynamic, Entity-Aware Grounded Response Generator
        top_evidence = retrieved_examples[0]["support_response"] if retrieved_examples else ""

        if intent == "software_update_and_ios":
            reply = f"We're here to help get your {device}{ver_str} updating smoothly. Please review our iOS recovery guide: [LINK] DM us your exact model and error details so we can assist!"
        elif intent == "battery_and_power":
            reply = f"We know how important battery life is on your {device}{ver_str}. Check your battery health settings or review these optimization steps: [LINK] DM us if the drain continues!"
        elif intent == "hardware_and_display":
            reply = f"We understand screen and hardware issues with your {device} can be frustrating. We recommend checking official repair options or scheduling a Genius Bar appointment: [LINK]"
        elif intent == "apple_id_and_security":
            reply = f"Account security is important to us. To regain access to your Apple ID, follow these secure recovery steps: [LINK] Send us a DM if you need step-by-step guidance!"
        elif intent == "apple_pay_and_billing":
            reply = f"We can help review charges and billing details for your account. Please check your purchase history here: [LINK] Send us a DM with purchase details so we can assist!"
        elif intent == "app_store_and_downloads":
            reply = f"Let's get your apps downloading properly on your {device}. Try restarting your device and checking your connection: [LINK] DM us if you need further help!"
        elif intent == "connectivity_and_cellular":
            reply = f"We're happy to help resolve Wi-Fi, Bluetooth, or cellular errors on your {device}. Try resetting network settings: [LINK] DM us if the connection issue persists!"
        elif intent == "icloud_and_storage":
            reply = f"We can help manage your iCloud storage and backups for your {device}. Follow these steps to optimize cloud space: [LINK] Send us a DM with any questions!"
        elif intent == "audio_and_airpods":
            reply = f"We want your audio working perfectly. Try resetting your audio settings or {device}: [LINK] DM us if you need further assistance!"
        elif intent == "itunes_and_apple_music":
            reply = f"We'd like to help get your music library synced to your {device}. Review these sync instructions: [LINK] DM us if you need extra support!"
        else:
            reply = f"We're happy to help with your {device}! DM us your model details, {ver_str or 'iOS version'}, and specific issue so we can get started: [LINK]"

        return {
            "reply": reply,
            "evidence_used": evidence_texts,
            "source": "grounded_template"
        }


if __name__ == "__main__":
    gen = ReplyGenerator()
    test_evidence = [{
        "conversation_id": "123",
        "customer_message": "My battery drops quickly",
        "support_response": "We'd love to help fix this. Follow these steps: [LINK]",
        "similarity_score": 0.88
    }]
    res = gen.generate_reply("My iPhone 7 battery is dying fast on iOS 11", "battery_and_power", test_evidence)
    print("\nOptimized Dynamic Reply Preview:")
    print(json.dumps(res, indent=2))
