"""精修应用模块.

将综合助手 accept/modify 后的 diff 操作按行号应用到初稿文本.
处理 add (行号+1 处插入)、modify (替换行)、remove (删除行) 的冲突.
"""

import logging
from models import DiffOperation, MergeDecision

logger = logging.getLogger(__name__)


def apply_refinements(
    draft_text: str,
    operations: list[DiffOperation],
    decisions: list[MergeDecision],
) -> str:
    """将精修操作应用到初稿文本.

    Args:
        draft_text: 初稿原文.
        operations: 所有改进专家提交的 diff 操作.
        decisions: 综合助手对每个操作的决策.

    Returns:
        精修后的文本.
    """
    # 构建 op_id -> decision 映射
    decision_map: dict[int, MergeDecision] = {d.op_id: d for d in decisions}

    # 筛选出被 accept 或 modify 的操作
    accepted_ops: list[DiffOperation] = []
    for op in operations:
        dec = decision_map.get(op.op_id)
        if not dec:
            continue

        if dec.decision == "reject":
            continue

        if dec.decision == "modify":
            # 用综合助手修改后的值覆盖
            effective_op = op.model_copy()
            if dec.modified_line is not None:
                effective_op.line = dec.modified_line
            if dec.modified_content is not None:
                effective_op.content = dec.modified_content
            accepted_ops.append(effective_op)
        else:
            # accept
            accepted_ops.append(op)

    if not accepted_ops:
        return draft_text

    lines = draft_text.split("\n")

    # 分类操作
    removes: set[int] = set()
    modifies: dict[int, str] = {}
    adds: dict[int, list[str]] = {}  # line -> [contents to add after]

    for op in accepted_ops:
        if op.action == "remove":
            removes.add(op.line)
        elif op.action == "modify":
            modifies[op.line] = op.content
        elif op.action == "add":
            adds.setdefault(op.line, []).append(op.content)
        else:
            logger.warning(
                "[Applier] unknown action %r for op_id=%d", op.action, op.op_id,
            )

    # 逐行构建结果
    result: list[str] = []
    for idx, line in enumerate(lines):
        line_no = idx + 1  # 1-indexed

        if line_no in removes:
            logger.debug("[Applier] removed line %d", line_no)
            # 但仍需处理 add (在被删除行之后添加)
            if line_no in adds:
                result.extend(adds[line_no])
            continue

        if line_no in modifies:
            result.append(modifies[line_no])
            logger.debug("[Applier] modified line %d", line_no)
        else:
            result.append(line)

        # 在当前行之后添加新内容
        if line_no in adds:
            result.extend(adds[line_no])
            logger.debug(
                "[Applier] added %d lines after line %d",
                len(adds[line_no]), line_no,
            )

    return "\n".join(result)
