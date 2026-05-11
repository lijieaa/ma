import re
from typing import TYPE_CHECKING

from config.logger import setup_logging
from plugins_func.register import Action, ActionResponse, ToolType, register_function

if TYPE_CHECKING:
    from core.connection import ConnectionHandler

TAG = __name__
logger = setup_logging()


EMERGENCY_KEYWORDS = [
    "想死",
    "不想活",
    "结束一切",
    "告别",
    "自残",
    "安眠药",
    "轻生",
    "割腕",
]

TRAUMA_KEYWORDS = ["破皮", "流血", "碰撞", "摔跤", "擦伤", "划伤", "扭伤", "出血", "割到", "扎到"]
INTERNAL_KEYWORDS = [
    "头疼",
    "头痛",
    "头晕",
    "头昏",
    "恶心",
    "肚子疼",
    "肚子痛",
    "发烧",
    "发热",
    "腹泻",
    "痛经",
]
MOOD_KEYWORDS = ["心情不好", "心里烦闷", "不开心", "压抑", "焦虑", "烦躁", "难受"]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").lower())


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(k in text for k in keywords)


def _wound_has_reported_debris(text: str) -> bool:
    """
    用户是否表示伤口内仍有泥沙/异物。
    注意：「没有脏东西」仍包含子串「脏东西」，不能仅用 in 判断。
    """
    t = (text or "").strip()
    if not t:
        return False

    # 明确否定或「干净」类回答
    neg_markers = [
        "没有脏东西",
        "没脏东西",
        "没有泥",
        "没泥",
        "没有沙",
        "没沙",
        "没有沙子",
        "没沙子",
        "没有石子",
        "没石子",
        "没有异物",
        "没异物",
        "无异物",
        "无泥沙",
        "没有灰",
        "没灰",
        "挺干净",
        "很干净",
        "挺干净的",
        "干净的",
        "没啥",
        "什么都没有",
        "不是脏东西",
        "不像脏东西",
    ]
    if any(m in t for m in neg_markers):
        return False
    if re.search(r"(没有|无|不含|没看到|没看见|没见).{0,8}(泥|沙|石子|沙子|异物|脏|灰)", t):
        return False

    positives = ["泥", "沙", "石子", "沙子", "脏东西", "异物", "泥土", "灰尘", "灰", "草屑", "渣子"]
    return any(k in t for k in positives)


def _set_state(conn: "ConnectionHandler", state: str):
    conn.medical_triage_state = state
    logger.bind(tag=TAG).info(f"更新医疗分诊状态: {state}")


def should_run_deterministic_triage(conn: "ConnectionHandler", raw_text: str) -> bool:
    """
    是否应由确定性分诊接管（function_call 模式下在进主 LLM 之前执行）。
    已进入分诊状态机时继续接管；新会话且命中关键词时也接管。
    """
    text = _normalize(raw_text)
    st = getattr(conn, "medical_triage_state", None) or ""
    if st and st not in ("done", "emergency_stop"):
        return True
    if _contains_any(text, EMERGENCY_KEYWORDS):
        return True
    if _contains_any(text, TRAUMA_KEYWORDS + INTERNAL_KEYWORDS + MOOD_KEYWORDS):
        return True
    return False


def _emergency_response(conn: "ConnectionHandler") -> ActionResponse:
    _set_state(conn, "emergency_stop")
    return ActionResponse(
        action=Action.RESPONSE,
        response=(
            "【AI紧急提醒】同学，现在请先停下对话，听我说。"
            "如果你此刻处在难以忍受的痛苦边缘，请立刻联系现实中的成年人帮你。"
            "1）如果你在宿舍或教室：请马上对最近的同学说“陪我去找老师”。"
            "2）如果你独自一人：请立刻拨打12355，或联系学校心理中心值班电话。"
            "你也可以立刻去找班主任、心理老师或家长。"
        ),
        result='{"ui_action":"emergency_contact","options":["call_teacher","call_12355"]}',
    )


campus_medical_triage_desc = {
    "type": "function",
    "function": {
        "name": "campus_medical_triage",
        "description": (
            "校园医疗分诊入口。用户提到外伤(破皮/流血/碰撞/摔跤)、"
            "身体不适(头疼/恶心/肚子疼)、情绪问题时必须调用。"
            "若出现自伤自杀意图词，必须立即输出紧急提醒并终止常规问答。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "user_text": {
                    "type": "string",
                    "description": "用户当前这句话原文",
                },
                "temperature_c": {
                    "type": "number",
                    "description": "可选，红外测温结果，单位摄氏度",
                },
            },
            "required": ["user_text"],
        },
    },
}


@register_function("campus_medical_triage", campus_medical_triage_desc, ToolType.SYSTEM_CTL)
def campus_medical_triage(
    conn: "ConnectionHandler", user_text: str, temperature_c: float | None = None
):
    text = _normalize(user_text)
    state = getattr(conn, "medical_triage_state", "") or ""
    # 上一轮已结束分诊时，允许根据新主诉重新进入流程
    if state in ("done",):
        _set_state(conn, "")
        state = ""

    if _contains_any(text, EMERGENCY_KEYWORDS):
        return _emergency_response(conn)

    if not state:
        if _contains_any(text, TRAUMA_KEYWORDS):
            _set_state(conn, "trauma_injury_type")
            return ActionResponse(
                action=Action.RESPONSE,
                response=(
                    "先确认怎么弄伤的：是摔倒蹭破皮、桌椅角碰撞淤青，还是猫抓狗咬/生锈铁钉扎伤？"
                ),
            )
        if _contains_any(text, INTERNAL_KEYWORDS):
            _set_state(conn, "internal_temp_check")
            return ActionResponse(
                action=Action.RESPONSE,
                response=(
                    "有没有量过体温？体感是发冷还是发热？请靠近体温探测窗口，我先帮你测温。"
                ),
                result='{"device_action":"read_infrared_temperature"}',
            )
        if _contains_any(text, MOOD_KEYWORDS):
            _set_state(conn, "mood_stage_1")
            return ActionResponse(
                action=Action.RESPONSE,
                response=(
                    "我是校园心理树洞。这里会尽量保护你的隐私；"
                    "但如果涉及你或他人的严重人身安全，老师必须介入。"
                    "你愿意先说说是心里闷，还是身体很累吗？"
                ),
            )
        return ActionResponse(
            action=Action.RESPONSE,
            response="我先帮你分诊：你现在是外伤、身体不适，还是心情状态不太好？",
        )

    # 外伤流程
    if state == "trauma_injury_type":
        if any(k in text for k in ["猫抓", "狗咬", "生锈", "铁钉"]):
            _set_state(conn, "done")
            return ActionResponse(
                action=Action.RESPONSE,
                response=(
                    "请务必马上去医务室找老师，需要判断是否要打破伤风针或狂犬疫苗，我这里不能处理。"
                ),
            )
        if "扭伤" in text or "脚踝肿" in text:
            _set_state(conn, "sprain_time")
            return ActionResponse(
                action=Action.RESPONSE,
                response="是刚扭伤48小时内，还是好几天了还有淤血？",
            )
        _set_state(conn, "trauma_cleanliness")
        return ActionResponse(
            action=Action.RESPONSE,
            response="伤口里有没有沙子、小石子或泥土？现在还在渗血吗？",
        )

    if state == "trauma_cleanliness":
        has_debris = _wound_has_reported_debris(text)
        if has_debris:
            _set_state(conn, "done")
            return ActionResponse(
                action=Action.RESPONSE,
                response=(
                    "伤口有异物，自己处理容易感染留疤。"
                    "请去医务室找老师用无菌生理盐水彻底冲洗清创，我不能直接发创可贴。"
                ),
            )
        _set_state(conn, "done")
        return ActionResponse(
            action=Action.RESPONSE,
            response=(
                "这是轻微表皮擦伤，符合自助外用药品领取条件。"
                "先用碘伏棉签由伤口中心向外画圈消毒两遍；"
                "若摩擦疼痛可贴无菌敷料。"
            ),
            result='{"device_actions":[{"servo":1,"item":"iodophor_swab"},{"servo":2,"item":"bandage"}]}',
        )

    if state == "sprain_time":
        if "48" in text or "刚" in text or "今天" in text or "24" in text:
            _set_state(conn, "done")
            return ActionResponse(
                action=Action.RESPONSE,
                response=(
                    "扭伤48小时内属于急性期，不能用活血化瘀药膏。"
                    "请去医务室拿冰袋冰敷15-20分钟，现在揉药酒会肿得更厉害。"
                ),
            )
        _set_state(conn, "sprain_allergy_check")
        return ActionResponse(
            action=Action.RESPONSE,
            response="超过48小时恢复期了。你对胶布过敏吗？不过敏可做弹性固定。",
        )

    if state == "sprain_allergy_check":
        _set_state(conn, "done")
        if "过敏" in text:
            return ActionResponse(
                action=Action.RESPONSE,
                response="既然有胶布过敏史，请直接去医务室由老师做替代固定方案。",
            )
        return ActionResponse(
            action=Action.RESPONSE,
            response="可以领取弹性绷带用于外用固定，注意避免再次负重扭转。",
            result='{"device_actions":[{"servo":2,"item":"elastic_bandage"}]}',
        )

    # 内科流程
    if state == "internal_temp_check":
        if temperature_c is not None and temperature_c > 38.5:
            _set_state(conn, "done")
            return ActionResponse(
                action=Action.RESPONSE,
                response=(
                    "你发热超过38.5度，AI无法处理，必须立即去医务室由老师判断是否需要去医院。"
                    "请戴好口罩，现在就去隔壁找老师。"
                ),
            )
        if "浑身发烫" in text or "骨头疼" in text:
            _set_state(conn, "done")
            return ActionResponse(
                action=Action.RESPONSE,
                response="你有高热伴全身酸痛风险，请立刻去医务室，不要继续等待。",
            )
        _set_state(conn, "digestive_check")
        return ActionResponse(
            action=Action.RESPONSE,
            response="肚子是阵发绞痛想拉肚子，还是一直隐隐作痛不舒服？",
        )

    if state == "digestive_check":
        if "拉肚子" in text or "腹泻" in text or "水样便" in text:
            _set_state(conn, "done")
            return ActionResponse(
                action=Action.RESPONSE,
                response=(
                    "先暂停冰饮和油炸外卖，可热敷腹部并喝温盐水。"
                    "如果水样便超过3次，请直接去医务室。"
                ),
            )
        if "痛经" in text:
            _set_state(conn, "done")
            return ActionResponse(
                action=Action.RESPONSE,
                response=(
                    "可以先做外用热敷。口服止痛药需要本人去医务室登记签字领取。"
                ),
                result='{"device_actions":[{"servo":3,"item":"warm_patch"}]}',
            )
        _set_state(conn, "done")
        return ActionResponse(
            action=Action.RESPONSE,
            response=(
                "这更像轻度身体应激反应。建议先休息补水。"
                "若出现心慌手抖明显加重，请立即去医务室做进一步检查。"
            ),
        )

    # 心理流程（简化分层）
    if state == "mood_stage_1":
        _set_state(conn, "mood_stage_2")
        return ActionResponse(
            action=Action.RESPONSE,
            response=(
                "如果用天气形容现在状态，更像毛毛雨天（闷）、大雾天（迷茫）、"
                "雷暴天（烦躁）还是大阴天（压抑）？"
            ),
        )

    if state == "mood_stage_2":
        _set_state(conn, "mood_stage_3")
        return ActionResponse(
            action=Action.RESPONSE,
            response=(
                "最近一周睡眠和食欲怎么样？有没有早醒、吃不下、对以前喜欢的事提不起劲？"
            ),
        )

    if state == "mood_stage_3":
        if any(k in text for k in ["两周", "2周", "早醒", "吃不下", "没意思"]):
            _set_state(conn, "done")
            return ActionResponse(
                action=Action.RESPONSE,
                response=(
                    "这听起来不只是短期情绪波动，建议去心理辅导室做一次心理体检。"
                    "这和看校医一样正常，不代表你有问题。"
                ),
            )
        _set_state(conn, "done")
        return ActionResponse(
            action=Action.RESPONSE,
            response=(
                "先做三个减压小动作：两次生理叹气、课间五分钟离开座位看远处、"
                "把烦心事写纸上揉成团扔掉。下周还难受可以继续来找我。"
            ),
        )

    _set_state(conn, "")
    return ActionResponse(action=Action.RESPONSE, response="我们重新开始分诊，你现在最不舒服的是哪一类？")
