import unittest

from engine.manager import _normalize_review_payload


class TestManagerReviewNormalization(unittest.TestCase):
    def test_normalize_review_payload_accepts_fallback_field_drift(self):
        raw = {
            "satisfied": False,
            "review_critique": [{"text": "总体不错"}],
            "overall_rejection_reason": "",
            "critique": "还需要补细节",
            "next_round_strategy": ["补证据", "补逐句覆盖"],
            "refined_experts": [
                {
                    "name": "证据分级核查员",
                    "domain": "证据强弱与争议边界",
                    "task": "补证据分级",
                    "temperature": 0.2,
                    "expert_model": "gpt-5.4-high",
                }
            ],
            "expert_actions": [
                {
                    "target_expert_role": "粤语顾问",
                    "action": "iterate",
                    "reason": "需要更细",
                    "improvement_suggestions": ["补语感", "补双关"],
                    "iterated_expert": {
                        "name": "粤语歌词校订员",
                        "domain": "粤语词义与语感",
                        "prompt": "重做粤语部分",
                        "temperature": 0.1,
                        "expert_model": "gemini-3.1-pro-preview",
                    },
                }
            ],
        }

        normalized = _normalize_review_payload(raw)

        self.assertEqual(normalized["review_critique"], "总体不错")
        self.assertEqual(normalized["next_round_strategy"], "补证据\n补逐句覆盖")
        self.assertEqual(normalized["refined_experts"][0]["role"], "证据分级核查员")
        self.assertEqual(
            normalized["refined_experts"][0]["description"],
            "证据强弱与争议边界",
        )
        self.assertEqual(
            normalized["refined_experts"][0]["expert_model"],
            "gpt-5.4-high",
        )
        self.assertEqual(
            normalized["expert_actions"][0]["improvement_suggestions"],
            "补语感\n补双关",
        )
        self.assertEqual(
            normalized["expert_actions"][0]["iterated_expert"]["role"],
            "粤语歌词校订员",
        )
        self.assertEqual(
            normalized["expert_actions"][0]["iterated_expert"]["expert_model"],
            "gemini-3.1-pro-preview",
        )
