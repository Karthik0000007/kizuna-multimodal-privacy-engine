import numpy as np
import pytest

from src.anomaly.scenarios import JapanScenario, build_alert_payload, get_japanese_alert


def test_japan_scenario_alerts():
    """Verify that all Japanese anomaly scenarios map to the correct localized strings."""
    assert get_japanese_alert(JapanScenario.FALL_RISK.value) == "「転倒リスク検出」"
    assert get_japanese_alert(JapanScenario.WANDERING.value) == "「徘徊の可能性検出」"
    assert get_japanese_alert(JapanScenario.CONGESTION.value) == "「混雑警報：異常な人口密度」"
    assert get_japanese_alert(JapanScenario.UNUSUAL_SOUND.value) == "「異常音検出」"
    assert get_japanese_alert(JapanScenario.ENVIRONMENTAL.value) == "「環境異常検出」"
    assert get_japanese_alert("normal") == "正常"
    assert get_japanese_alert("random_event") == "「異常検出: random_event」"


def test_build_alert_payload():
    """Verify that the alert payload structure is correct for the dashboard."""
    payload = build_alert_payload(JapanScenario.FALL_RISK.value, 0.95, "edge-node-2")

    assert payload["alert_message_jp"] == "「転倒リスク検出」"
    assert payload["event_type"] == "fall_risk"
    assert payload["confidence"] == 0.95
    assert payload["source_node"] == "edge-node-2"
    assert payload["severity"] == "high"

    payload_med = build_alert_payload(JapanScenario.WANDERING.value, 0.6, "edge-node-1")
    assert payload_med["severity"] == "medium"


def test_enrollment_pre_enroll_scenarios():
    """Verify that pre-enrollment adds exemplars for Japan scenarios to the DB."""
    from src.anomaly.enrollment import pre_enroll_japan_scenarios
    from src.database.base import VectorStore

    class DummyStore(VectorStore):
        def __init__(self):
            self.inserted = []

        def insert(self, vector, metadata):
            self.inserted.append(metadata)
            return f"id-{len(self.inserted)}"

        def search(self, query, top_k, filters=None):
            return []

        def create_collection(self, dimension, distance="Cosine"):
            pass

        def delete_collection(self):
            pass

        def get_collection_info(self):
            return {}

    store = DummyStore()
    pre_enroll_japan_scenarios(store)

    # Check that at least the 4 core scenarios were enrolled
    enrolled_types = [m["event_type"] for m in store.inserted]

    assert JapanScenario.FALL_RISK.value in enrolled_types
    assert JapanScenario.WANDERING.value in enrolled_types
    assert JapanScenario.CONGESTION.value in enrolled_types
    assert JapanScenario.UNUSUAL_SOUND.value in enrolled_types
