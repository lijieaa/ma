"""scripted_reply 标记解析单测（无第三方依赖）。"""

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.utils.scripted_reply import (  # noqa: E402
    maybe_inject_round7_fallback,
    parse_scripted_reply,
    scripted_reply_needs_orchestration,
    strip_markers,
)


ROUND7 = """这是轻微表皮擦伤，符合自助外用药品领取条件。已将你的情况发送到智能医务室管理系统由医生确认，请稍等。
<<<PAUSE:3>>>
<<<DISPENSE>>>
好的，医生已经确认，现在给你发放碘伏和敷料贴，请在窗口取用碘伏和敷料贴，用碘伏由伤口中心向外画圈消毒两遍，再用敷料贴贴上。如果伤口继续疼痛红肿，请去学校医务室找医生处置，不可拖延。"""


class ScriptedReplyTest(unittest.TestCase):
    def test_round7_parse(self):
        steps = parse_scripted_reply(ROUND7)
        self.assertTrue(scripted_reply_needs_orchestration(steps))
        types = [s["type"] for s in steps]
        self.assertEqual(types, ["speak", "pause", "dispense", "speak"])
        self.assertEqual(steps[1]["seconds"], 3)
        self.assertIn("请稍等", steps[0]["text"])
        self.assertIn("医生已经确认", steps[3]["text"])
        self.assertNotIn("<<<", strip_markers(ROUND7))

    def test_no_markers(self):
        text = "你好呀，我是小智。"
        steps = parse_scripted_reply(text)
        self.assertFalse(scripted_reply_needs_orchestration(steps))
        self.assertEqual(steps, [{"type": "speak", "text": text}])

    def test_fallback_inject(self):
        text = (
            "符合条件。已发送到系统由医生确认，请稍等。"
            "好的，医生已经确认，现在给你发放碘伏和敷料贴。"
        )
        injected = maybe_inject_round7_fallback(text)
        self.assertIn("<<<PAUSE:3>>>", injected)
        self.assertIn("<<<DISPENSE>>>", injected)
        self.assertTrue(
            scripted_reply_needs_orchestration(parse_scripted_reply(injected))
        )


if __name__ == "__main__":
    unittest.main()
