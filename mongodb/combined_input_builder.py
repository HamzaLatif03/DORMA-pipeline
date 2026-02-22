"""
Builds a complete input package combining MongoDB context and audio/text input
for ElevenLabs or other AI services.
"""
from datetime import datetime
from typing import Optional, Dict, Any
from mongodb.agent_story import build_user_story, get_latest_session
from mongodb.db import get_client

DB2 = "user_history_db"


class CombinedInputBuilder:
    """Builds rich input combining user context, history, and current interaction."""

    def __init__(self, user_id: str):
        self.user_id = user_id

    def build_complete_input(
        self,
        current_input: str,
        include_metrics: bool = True,
        include_recent_events: bool = True,
        max_events: int = 5
    ) -> Dict[str, Any]:
        """
        Build a complete input package for AI processing.

        Returns:
            {
                "user_id": str,
                "timestamp": datetime,
                "user_context": {
                    "name": str,
                    "tags": list,
                    "preferences": dict
                },
                "recent_history": list,
                "latest_metrics": dict (optional),
                "current_input": str,
                "formatted_prompt": str  # Ready for LLM
            }
        """
        db = get_client()[DB2]

        # 1. Get user profile
        profile = db["profiles"].find_one({"_id": self.user_id}) or {}
        user_context = {
            "name": profile.get("name", "Unknown"),
            "tags": profile.get("tags", []),
            "preferences": profile.get("preferences", {})
        }

        # 2. Get recent events
        recent_history = []
        if include_recent_events:
            events = list(
                db["events"]
                .find({"userId": self.user_id})
                .sort("ts", -1)
                .limit(max_events)
            )
            events.reverse()

            for event in events:
                recent_history.append({
                    "type": event.get("type"),
                    "timestamp": event.get("ts"),
                    "meta": event.get("meta", {})
                })

        # 3. Get latest metrics (if available)
        latest_metrics = None
        if include_metrics:
            session = get_latest_session(self.user_id)
            if session:
                metrics = session.get("metrics", {})
                # Compute simple statistics
                hr = metrics.get("hr", {})
                rr = metrics.get("rr", {})

                if hr:
                    hr_vals = list(hr.values())
                    avg_hr = sum(hr_vals) / len(hr_vals) if hr_vals else 0
                else:
                    avg_hr = 0

                if rr:
                    rr_vals = list(rr.values())
                    avg_rr = sum(rr_vals) / len(rr_vals) if rr_vals else 0
                else:
                    avg_rr = 0

                latest_metrics = {
                    "session_start": session.get("startedAt"),
                    "duration_sec": session.get("durationSec"),
                    "avg_heart_rate": round(avg_hr, 1) if avg_hr else None,
                    "avg_respiratory_rate": round(avg_rr, 1) if avg_rr else None,
                    "apnea_detected": any(metrics.get("apnea", {}).values()) if metrics.get("apnea") else False
                }

        # 4. Build formatted prompt
        formatted_prompt = self._format_prompt(
            user_context, recent_history, latest_metrics, current_input
        )

        return {
            "user_id": self.user_id,
            "timestamp": datetime.utcnow(),
            "user_context": user_context,
            "recent_history": recent_history,
            "latest_metrics": latest_metrics,
            "current_input": current_input,
            "formatted_prompt": formatted_prompt
        }

    def _format_prompt(
        self,
        user_context: dict,
        recent_history: list,
        latest_metrics: Optional[dict],
        current_input: str
    ) -> str:
        """Format all data into a single prompt string."""
        sections = []

        # User identity
        sections.append("=== USER IDENTITY ===")
        sections.append(f"Name: {user_context['name']}")
        if user_context['tags']:
            sections.append(f"Tags: {', '.join(user_context['tags'])}")
        if user_context['preferences']:
            sections.append(f"Preferences: {user_context['preferences']}")

        # Recent history
        sections.append("\n=== RECENT HISTORY ===")
        if not recent_history:
            sections.append("(No recent history)")
        else:
            for event in recent_history:
                etype = event['type']
                meta = event.get('meta', {})
                if etype == "seen":
                    sections.append(
                        f"- Seen at: {meta.get('location', 'unknown')}")
                elif etype == "conversation":
                    sections.append(
                        f"- Previous conversation: {meta.get('summary', 'N/A')}")
                elif etype == "note":
                    sections.append(f"- Note: {meta.get('text', 'N/A')}")

        # Latest biometric metrics
        if latest_metrics:
            sections.append("\n=== LATEST BIOMETRICS ===")
            if latest_metrics.get("avg_heart_rate"):
                sections.append(
                    f"Heart Rate: {latest_metrics['avg_heart_rate']} bpm")
            if latest_metrics.get("avg_respiratory_rate"):
                sections.append(
                    f"Respiratory Rate: {latest_metrics['avg_respiratory_rate']} breaths/min")
            if latest_metrics.get("apnea_detected"):
                sections.append("⚠️ Apnea detected in recent session")

        # Current input
        sections.append("\n=== CURRENT INPUT ===")
        sections.append(f"User says: {current_input}")

        sections.append("\n=== INSTRUCTIONS ===")
        sections.append("Respond naturally based on all the context above.")
        sections.append(
            "Be concise and personalized to the user's preferences.")

        return "\n".join(sections)


def demo_combined_input():
    """Demo the combined input builder."""
    builder = CombinedInputBuilder(user_id="user_123")

    result = builder.build_complete_input(
        current_input="How am I doing today?",
        include_metrics=True
    )

    print("\n" + "="*60)
    print("COMPLETE INPUT PACKAGE")
    print("="*60)
    print(f"\nUser ID: {result['user_id']}")
    print(f"Timestamp: {result['timestamp']}")
    print(f"\nUser Context: {result['user_context']}")
    print(f"\nRecent History: {len(result['recent_history'])} events")
    print(f"\nLatest Metrics: {result['latest_metrics']}")
    print(f"\nCurrent Input: {result['current_input']}")
    print(f"\n{'='*60}")
    print("FORMATTED PROMPT FOR AI")
    print("="*60)
    print(result['formatted_prompt'])


if __name__ == "__main__":
    demo_combined_input()
