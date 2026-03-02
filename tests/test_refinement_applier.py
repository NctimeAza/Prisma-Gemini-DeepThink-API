import unittest

from engine.refinement.applier import apply_refinements
from models import DiffOperation, MergeDecision


class TestRefinementApplier(unittest.TestCase):
    def test_apply_refinements_modify_remove(self):
        draft = "A\nB\nC"

        operations = [
            DiffOperation(op_id=0, expert_role="t", action="modify", line=2, content="B2"),
            DiffOperation(op_id=1, expert_role="t", action="remove", line=3),
        ]
        decisions = [
            MergeDecision(op_id=0, decision="accept"),
            MergeDecision(op_id=1, decision="accept"),
        ]

        out = apply_refinements(draft, operations, decisions)
        self.assertEqual(out, "A\nB2")

