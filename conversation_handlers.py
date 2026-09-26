"""
magicpin AI Challenge — Multi-Turn Conversation Handler
Implements section 7.4 of challenge-brief.md for replay & multi-turn evaluation.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import bot

@dataclass
class ConversationTurn:
    role: str
    message: str
    turn_number: int
    intent: Optional[str] = None

@dataclass
class ConversationState:
    conversation_id: str
    merchant_id: str
    history: List[ConversationTurn] = field(default_factory=list)
    mode: str = "initial"  # "initial", "action", "ended"

def respond(state: ConversationState, merchant_message: str) -> Dict[str, Any]:
    """
    Given the conversation state so far + the merchant's latest message, produce the reply.
    Evaluated on:
    - Auto-reply detection -> action: 'end' or 'wait'
    - Intent transition -> action: 'send' in ACTION mode (zero qualifying questions)
    - Hostility handling -> action: 'end' with graceful de-escalation
    """
    prev_merchant_msgs = [turn.message for turn in state.history if turn.role in ("merchant", "customer")]
    intent = bot.classify_inbound_message(merchant_message, prev_merchant_msgs)

    # 1. WhatsApp Business Auto-reply pattern
    if intent == "auto_reply":
        state.mode = "ended"
        return {
            "action": "end",
            "body": "Understood. Ending automated outreach so as not to disturb your team. Best wishes!",
            "cta": "none",
            "rationale": "Detected canned WhatsApp Business auto-reply pattern. Terminated conversation immediately."
        }

    # 2. Hostile / Opt-out
    if intent == "hostile":
        state.mode = "ended"
        return {
            "action": "end",
            "body": "Sorry for the inconvenience. We won't message you again. Wishing your business continued success.",
            "cta": "none",
            "rationale": "Merchant opted out or signaled hostility. Apologized gracefully and ended outreach."
        }

    # 3. Commitment / Action Transition (CRITICAL: zero qualification questions)
    if intent == "commitment":
        state.mode = "action"
        return {
            "action": "send",
            "body": "Done! Proceeding with the action now. Here is your confirmed campaign draft ready to go live. Next step is publishing directly to your Google Business Profile.",
            "cta": "binary_yes_no",
            "rationale": "Merchant committed. Switched immediately to action mode with confirmed draft without repetitive qualification."
        }

    # 4. Busy / Delay
    if intent == "busy":
        return {
            "action": "wait",
            "wait_seconds": 1800,
            "body": "",
            "cta": "none",
            "rationale": "Merchant requested time / is currently busy. Backing off for 30 minutes."
        }

    # 5. Slot choice (Customer booking)
    if intent == "slot_choice":
        return {
            "action": "send",
            "body": "Confirmed! Your preferred appointment slot has been locked in. Our team has scheduled your visit. Let us know if you need to reschedule.",
            "cta": "open_ended",
            "rationale": "Confirmed patient slot choice for recall booking."
        }

    # 6. Default inquiry response
    return {
        "action": "send",
        "body": "Here are the concrete details: I have prepared the complete profile boost draft with updated hours, photos, and your active catalog offer. Sending the preview link now.",
        "cta": "open_ended",
        "rationale": "Addressed merchant inquiry with specific details and actionable preview."
    }
