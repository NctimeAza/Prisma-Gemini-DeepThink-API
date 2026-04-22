import unittest

from config import resolve_model, split_provider_model_prefix


class TestProviderResolution(unittest.TestCase):
    def test_split_provider_model_prefix(self):
        self.assertEqual(
            split_provider_model_prefix("openai_responses/gpt-5.4"),
            ("openai_responses", "gpt-5.4"),
        )
        self.assertIsNone(split_provider_model_prefix("gpt-5.4"))
        self.assertIsNone(split_provider_model_prefix("missing/gpt-5.4"))

    def test_resolve_direct_provider_model(self):
        (
            real_model,
            manager_model,
            synthesis_model,
            _planning_level,
            _expert_level,
            _synthesis_level,
            _max_rounds,
            provider,
            *_rest,
            stage_providers,
        ) = resolve_model("openai_responses/gpt-5.4")

        self.assertEqual(real_model, "gpt-5.4")
        self.assertEqual(manager_model, "gpt-5.4")
        self.assertEqual(synthesis_model, "gpt-5.4")
        self.assertEqual(provider, "openai_responses")
        self.assertEqual(stage_providers.manager, "openai_responses")
        self.assertEqual(stage_providers.expert, "openai_responses")
        self.assertEqual(stage_providers.synthesis, "openai_responses")

    def test_resolve_builtin_mixed_model(self):
        (
            real_model,
            manager_model,
            synthesis_model,
            _planning_level,
            _expert_level,
            _synthesis_level,
            max_rounds,
            provider,
            _planning_temp,
            _expert_temp,
            _review_temp,
            _synthesis_temp,
            mode,
            _json_via_prompt,
            stage_providers,
        ) = resolve_model("gpt-5.4-deepthink-minimal")

        self.assertEqual(real_model, "gpt-5.4")
        self.assertEqual(manager_model, "gemini-3.1-pro-preview")
        self.assertEqual(synthesis_model, "gemini-3.1-pro-preview")
        self.assertEqual(max_rounds, 1)
        self.assertEqual(provider, "gemini")
        self.assertEqual(mode, "classic")
        self.assertEqual(stage_providers.manager, "gemini")
        self.assertEqual(stage_providers.expert, "openai_responses")
        self.assertEqual(stage_providers.synthesis, "gemini")

    def test_forced_suffix_and_provider_model_can_coexist(self):
        (
            real_model,
            manager_model,
            synthesis_model,
            *_head,
            stage_providers,
        ) = resolve_model("openai_responses/gpt-5.4-forced")

        self.assertEqual(real_model, "gpt-5.4")
        self.assertEqual(manager_model, "gpt-5.4")
        self.assertEqual(synthesis_model, "gpt-5.4")
        self.assertEqual(stage_providers.expert, "openai_responses")
