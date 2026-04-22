import unittest

from prompts import (
    EXPERT_HIDDEN_OUTPUT_EN_INSTRUCTION,
    get_expert_system_instruction,
)


class TestPrompts(unittest.TestCase):
    def test_expert_system_instruction_contains_hidden_output_rule(self):
        prompt = get_expert_system_instruction(
            role="测试专家",
            description="负责测试",
            context="上下文",
            all_expert_roles=["测试专家"],
            user_system_prompt="",
        )

        self.assertIn(EXPERT_HIDDEN_OUTPUT_EN_INSTRUCTION, prompt)
