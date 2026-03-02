import unittest

from engine.refinement.cleaner import parse_cleaner_result


class TestRefinementCleaner(unittest.TestCase):
    def test_parse_cleaner_result_filters_invalid_ops(self):
        parsed = {
            "analysis": "  ok  ",
            "operations": [
                {"action": "add", "line": 1, "content": "X"},  # not allowed
                {"action": "modify", "line": 0, "content": "X"},  # out of range
                {"action": "modify", "line": 2},  # missing content
                {"action": "modify", "line": 2, "content": "B2"},
                {"action": "remove", "line": 99},  # out of range
                {"action": "remove", "line": 3, "reason": "dup"},
                {"action": "modify", "line": 1, "content": "bad\nline"},  # newline
            ],
        }

        analysis, ops = parse_cleaner_result(parsed, max_line=3)
        self.assertEqual(analysis, "ok")
        self.assertEqual(
            [(op.action, op.line, op.content) for op in ops],
            [("modify", 2, "B2"), ("remove", 3, "")],
        )

    def test_parse_cleaner_result_dedup_remove_wins(self):
        parsed = {
            "analysis": "",
            "operations": [
                {"action": "modify", "line": 2, "content": "B2"},
                {"action": "remove", "line": 2, "reason": "dup"},
            ],
        }

        _, ops = parse_cleaner_result(parsed, max_line=3)
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0].action, "remove")
        self.assertEqual(ops[0].line, 2)

