import unittest

from engine.refinement.reviewer import _normalize_review_payload


class TestReviewerNormalization(unittest.TestCase):
    def test_normalize_review_payload_accepts_drifted_fields(self):
        parsed = {
            "issues": [{"text": "证据分级不够细"}],
            "refinement_experts": [
                {
                    "name": "证据分级核查员",
                    "domain": "证据强弱与争议边界",
                    "task": "补证据分级",
                    "temperature": 0.2,
                }
            ],
            "expert_guidance": [
                {"name": "证据分级核查员", "guidance": ["先做分级", "再补证据"]},
            ],
            "approved": False,
            "approval_reason": ["还不够稳"],
        }

        normalized = _normalize_review_payload(parsed)

        self.assertEqual(normalized["issues"], ["证据分级不够细"])
        self.assertEqual(
            normalized["refinement_experts"][0]["role"], "证据分级核查员"
        )
        self.assertEqual(
            normalized["expert_guidance"]["证据分级核查员"],
            "先做分级\n再补证据",
        )
        self.assertEqual(normalized["approval_reason"], "还不够稳")
