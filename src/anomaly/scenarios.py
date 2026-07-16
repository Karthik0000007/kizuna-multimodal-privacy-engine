from enum import Enum
from typing import Any, Dict, Optional


class JapanScenario(Enum):
    FALL_RISK = "fall_risk"
    WANDERING = "wandering"
    CONGESTION = "congestion_alert"
    UNUSUAL_SOUND = "unusual_sound"
    ENVIRONMENTAL = "environmental_anomaly"


JAPANESE_ALERTS = {
    JapanScenario.FALL_RISK.value: "「転倒リスク検出」",
    JapanScenario.WANDERING.value: "「徘徊の可能性検出」",
    JapanScenario.CONGESTION.value: "「混雑警報：異常な人口密度」",
    JapanScenario.UNUSUAL_SOUND.value: "「異常音検出」",
    JapanScenario.ENVIRONMENTAL.value: "「環境異常検出」",
}


def get_japanese_alert(event_type: str) -> str:
    """
    Returns the localized Japanese alert string for a given event_type.
    If the event_type is not a recognized scenario, returns a generic anomaly string.
    """
    if event_type in JAPANESE_ALERTS:
        return JAPANESE_ALERTS[event_type]
    elif event_type == "normal":
        return "正常"
    else:
        return f"「異常検出: {event_type}」"


def build_alert_payload(event_type: str, confidence: float, source_node: str) -> Dict[str, Any]:
    """
    Constructs a localized alert payload suitable for dashboard display or push notifications.
    """
    localized_message = get_japanese_alert(event_type)

    return {
        "alert_message_jp": localized_message,
        "event_type": event_type,
        "confidence": confidence,
        "source_node": source_node,
        "severity": "high" if confidence > 0.8 else "medium",
    }
