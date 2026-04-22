import unittest

from engine.refinement.no_cliches import apply_no_cliches_operations
from models import DiffOperation


class TestNoCliches(unittest.TestCase):
    def test_apply_no_cliches_operations(self):
        text = "第一行\n第二行\n第三行"
        operations = [
            DiffOperation(action="modify", line=2, content="第二行-改"),
            DiffOperation(action="remove", line=3, reason="删除套话"),
        ]

        cleaned = apply_no_cliches_operations(text, operations)
        self.assertEqual(cleaned, "第一行\n第二行-改")
