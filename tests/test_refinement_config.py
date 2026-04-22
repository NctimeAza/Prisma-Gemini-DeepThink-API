import unittest
from unittest.mock import patch

import config
from config import (
    StageProviders,
    VirtualModel,
    resolve_expert_model_selection,
    resolve_expert_routing_config,
    resolve_model,
    resolve_no_cliches_config,
    resolve_refinement_config,
)


class TestRefinementConfig(unittest.TestCase):
    def test_resolve_mixed_refinement_high_config(self):
        (
            real_model,
            manager_model,
            synthesis_model,
            _planning_level,
            _expert_level,
            _synthesis_level,
            _max_rounds,
            _provider,
            _planning_temp,
            _expert_temp,
            _review_temp,
            _synthesis_temp,
            mode,
            _json_via_prompt,
            stage_providers,
        ) = resolve_model("gemini-gpt-deepthink-refinement-high")

        self.assertEqual(mode, "refinement")
        self.assertEqual(real_model, "gemini-3.1-pro-preview")
        self.assertEqual(manager_model, "gpt-5.4-high")
        self.assertEqual(synthesis_model, "gemini-3.1-pro-preview")

        ref_cfg = resolve_refinement_config(
            "gemini-gpt-deepthink-refinement-high",
            real_model,
            manager_model,
            synthesis_model,
            stage_providers,
        )

        self.assertEqual(ref_cfg.refinement_planner_model, "gpt-5.4-high")
        self.assertEqual(ref_cfg.refinement_planner_provider, "openai_responses")
        self.assertEqual(ref_cfg.pre_draft_expert_model, "gemini-3.1-pro-preview")
        self.assertEqual(ref_cfg.pre_draft_expert_provider, "gemini")
        self.assertEqual(ref_cfg.pre_draft_review_model, "gpt-5.4-high")
        self.assertEqual(ref_cfg.pre_draft_review_provider, "openai_responses")
        self.assertEqual(ref_cfg.draft_model, "claude-opus-4-6-thinking")
        self.assertEqual(ref_cfg.draft_provider, "gemini")
        self.assertEqual(ref_cfg.review_model, "gpt-5.4-high")
        self.assertEqual(ref_cfg.review_provider, "openai_responses")
        self.assertEqual(ref_cfg.improver_model, "gemini-3.1-pro-preview")
        self.assertEqual(ref_cfg.improver_provider, "gemini")
        self.assertEqual(ref_cfg.text_cleaner_model, "gemini-3.1-pro-preview")
        self.assertEqual(ref_cfg.text_cleaner_provider, "gemini")

    def test_resolve_no_cliches_defaults(self):
        cfg = resolve_no_cliches_config(
            "unit-nonexistent-model",
            "gemini-3.1-pro-preview",
            StageProviders.from_single("gemini"),
        )

        self.assertEqual(
            cfg.enable_no_cliches,
            config.REFINEMENT_NO_CLICHES_ENABLED,
        )
        self.assertEqual(
            cfg.no_cliches_model,
            config.REFINEMENT_NO_CLICHES_MODEL,
        )
        self.assertEqual(
            cfg.no_cliches_provider,
            config.REFINEMENT_NO_CLICHES_PROVIDER,
        )

    def test_resolve_no_cliches_virtual_model_override(self):
        vm = VirtualModel(
            id="unit-no-cliches",
            real_model="gemini-3.1-pro-preview",
            planning_level="high",
            expert_level="high",
            synthesis_level="high",
            desc="unit",
            enable_no_cliches=True,
            no_cliches_model="gemini-3.1-pro-preview",
            no_cliches_provider="gemini",
        )

        with patch.dict(config._VIRTUAL_MODEL_MAP, {vm.id: vm}, clear=False):
            cfg = resolve_no_cliches_config(
                vm.id,
                vm.real_model,
                StageProviders.from_single("openai"),
            )

        self.assertTrue(cfg.enable_no_cliches)
        self.assertEqual(cfg.no_cliches_model, "gemini-3.1-pro-preview")
        self.assertEqual(cfg.no_cliches_provider, "gemini")

    def test_resolve_expert_routing_config_for_code_xhigh(self):
        cfg = resolve_expert_routing_config(
            "gemini-gpt-deepthink-refinement-code-xhigh"
        )

        self.assertTrue(cfg.enable_manager_expert_model_selection)
        self.assertTrue(cfg.enable_review_expert_model_selection)
        self.assertEqual(
            [item.id for item in cfg.expert_model_pool],
            ["gpt-5.4-high", "gemini-3.1-pro-preview"],
        )

    def test_resolve_expert_model_selection_uses_pool_override(self):
        routing_cfg = resolve_expert_routing_config(
            "gemini-gpt-deepthink-refinement-code-xhigh"
        )

        selected, model, provider = resolve_expert_model_selection(
            "gpt-5.4-high",
            default_model="gemini-3.1-pro-preview",
            default_provider="gemini",
            expert_model_pool=routing_cfg.expert_model_pool,
        )

        self.assertEqual(selected, "gpt-5.4-high")
        self.assertEqual(model, "gpt-5.4-high")
        self.assertEqual(provider, "openai_responses")
