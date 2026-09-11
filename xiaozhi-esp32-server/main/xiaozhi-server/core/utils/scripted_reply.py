"""扣子固定话术控制标记解析：分段播报 / 停顿 / 出药。"""

from __future__ import annotations

import re
from typing import Any, Dict, List

# <<<PAUSE:3>>> 或 <<<DISPENSE>>>
_MARKER_RE = re.compile(
    r"<<<\s*(?:PAUSE\s*:\s*(\d+)|DISPENSE)\s*>>>",
    re.IGNORECASE,
)


def strip_markers(text: str) -> str:
    """去掉控制标记，保留可读正文（用于对话历史）。"""
    if not text:
        return ""
    cleaned = _MARKER_RE.sub("", text)
    # 压缩因删标记产生的多余空行
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def parse_scripted_reply(text: str) -> List[Dict[str, Any]]:
    """
    将带标记的回复解析为有序步骤：
      {"type": "speak", "text": "..."}
      {"type": "pause", "seconds": 3}
      {"type": "dispense"}
    """
    raw = text or ""
    if not raw.strip():
        return []

    steps: List[Dict[str, Any]] = []
    last = 0
    for m in _MARKER_RE.finditer(raw):
        before = raw[last : m.start()]
        speak = before.strip()
        if speak:
            steps.append({"type": "speak", "text": speak})

        if m.group(1) is not None:
            try:
                seconds = int(m.group(1))
            except (TypeError, ValueError):
                seconds = 0
            if seconds > 0:
                steps.append({"type": "pause", "seconds": seconds})
        else:
            steps.append({"type": "dispense"})
        last = m.end()

    tail = raw[last:].strip()
    if tail:
        steps.append({"type": "speak", "text": tail})

    if not steps and raw.strip():
        steps.append({"type": "speak", "text": raw.strip()})

    return steps


def scripted_reply_needs_orchestration(steps: List[Dict[str, Any]]) -> bool:
    """是否需要走分段播报+停顿/出药编排（相对普通一次 TTS）。"""
    return any(s.get("type") in ("pause", "dispense") for s in steps)


def maybe_inject_round7_fallback(text: str) -> str:
    """
    标记丢失时的演示兜底：若同时含「请稍等」与「医生已经确认」，
    且尚无控制标记，则在「请稍等」后插入 PAUSE+DISPENSE。
    """
    t = text or ""
    if not t.strip():
        return t
    if _MARKER_RE.search(t):
        return t
    if "请稍等" not in t or "医生已经确认" not in t:
        return t

    # 在首次「请稍等」及其后标点处切开
    m = re.search(r"请稍等[。.!！？]?", t)
    if not m:
        return t
    head = t[: m.end()].rstrip()
    tail = t[m.end() :].lstrip()
    if not tail:
        return t
    return f"{head}\n<<<PAUSE:3>>>\n<<<DISPENSE>>>\n{tail}"
