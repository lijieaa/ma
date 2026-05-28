import os
import sys
import copy
import json
import re
import uuid
import time
import queue
import asyncio
import threading
import traceback
import subprocess
import websockets

from core.utils.util import (
    extract_json_from_string,
    check_vad_update,
    check_asr_update,
    filter_sensitive_info,
    sanitize_tool_name,
)
from typing import Any, Dict, Optional
from collections import deque
from core.utils.modules_initialize import (
    initialize_modules,
    initialize_tts,
    initialize_asr,
)
from core.handle.reportHandle import report, enqueue_tool_report
from core.providers.tts.default import DefaultTTS
from concurrent.futures import ThreadPoolExecutor
from core.utils.dialogue import Message, Dialogue
from core.providers.asr.dto.dto import InterfaceType
from core.handle.textHandle import handleTextMessage
from core.providers.tools.unified_tool_handler import UnifiedToolHandler
from plugins_func.loadplugins import auto_import_modules
from plugins_func.register import Action, ActionResponse
from core.auth import AuthenticationError
from config.config_loader import get_private_config_from_api
from core.providers.tts.dto.dto import ContentType, TTSMessageDTO, SentenceType
from config.logger import setup_logging, build_module_string, create_connection_logger
from config.manage_api_client import DeviceNotFoundException, DeviceBindException
from core.utils.prompt_manager import PromptManager
from core.utils.voiceprint_provider import VoiceprintProvider
from core.utils.util import get_system_error_response
from core.utils import textUtils


TAG = __name__

# 工具调用规则 - 用于动态注入提醒
TOOL_CALLING_RULES = """
<tool_calling>
【核心原则】你是拥有工具能力的智能助手。当用户请求需要实时信息或执行操作时，调用相应工具获取数据，禁止凭空编造答案。

- **何时必须调用工具：**
  1. 实时信息查询（新闻、非本地天气、股价、汇率等）
  2. 执行操作（播放音乐、控制设备、拍照、设置闹钟等）
  3. 知识库检索（当工具列表包含 search_from_ragflow 时，结合用户意图判断是否需要调用）
  4. 查询非今天的农历信息（明天农历、某日宜忌、节气等）
  5. 用户说"拍照"时调用 self_camera_take_photo，默认 question 参数为"描述一下看到的物品"

- **何时无需调用工具：**
  1. `<context>` 中已提供的信息（当前时间、今天日期、今天农历、本地天气等）
  2. 普通对话、问候、闲聊、情感交流、讲故事
  3. 通用知识问答（非实时信息）

- **调用规范：**
  1. 每次请求独立判断，不复用历史工具结果，需重新获取最新数据
  2. 多任务时依次调用所有需要的工具，并依次总结每个工具的结果，不得遗漏
  3. 严格遵循工具的参数要求，提供所有必要参数
  4. 不确定时引导用户澄清或告知能力限制，切勿猜测或编造
  5. 不调用未提供的工具，对话中提及的旧工具若不可用则忽略或说明

- **反偷懒机制（最高优先级）：**
  1. **每次独立判断：** 无论对话历史中是否调用过工具，当前请求必须根据当前需求独立判断是否需要调用
  2. **禁止模式模仿：** 即使之前的回复没有调用工具，也不代表本次可以不调用
  3. **自我检查：** 回复前必须自问："这个请求是否涉及实时信息或执行操作？如果是，我调用工具了吗？"
  4. **历史不等于现在：** 对话历史中的行为模式不影响当前判断，每个用户请求都是全新的开始
</tool_calling>
"""


def _main_llm_is_coze_workflow_run(config) -> bool:
    """主对话 LLM 是否为扣子工作流 /run（纯对话，不走 MCP 工具注入）。"""
    llm_name = (config.get("selected_module") or {}).get("LLM", "")
    llm_cfg = (config.get("LLM") or {}).get(llm_name, {})
    return llm_cfg.get("type") == "coze_workflow_run"


def _extract_plain_query_for_tool_routing(query) -> str:
    """从 ASR/上游可能传入的 JSON 信封中取出用户原句，供关键词与工具路由使用。"""
    if query is None:
        return ""
    if isinstance(query, dict):
        c = query.get("content")
        return c.strip() if isinstance(c, str) else ""
    q = str(query).strip()
    if q.startswith("{") and q.endswith("}"):
        try:
            data = json.loads(q)
            if isinstance(data, dict):
                c = data.get("content")
                if isinstance(c, str) and c.strip():
                    return c.strip()
        except json.JSONDecodeError:
            pass
    return q


def _user_intent_body_temperature_measure(query: str) -> bool:
    """与设备端 MLX90614 工具描述中的触发语对齐（用于无 OpenAI tool_calls 的 LLM 后端）。"""
    if not query:
        return False
    if _user_voice_command_start_mlx_module(query):
        return True
    if _user_voice_command_open_mlx_temp_feature(query):
        return True
    if "mlx90614" in query.lower():
        return True
    needles = (
        "测体温",
        "测量体温",
        "量体温",
        "测一下体温",
        "测量一下体温",
        "量一下体温",
        "额温",
        "体温度数",
        "红外测温",
        "帮我测",
        "给我测",
    )
    return any(n in query for n in needles)


def _normalize_voice_command_text(text: str) -> str:
    """纠正常见 ASR 同音误识别，便于命中固定 MCP 指令。"""
    t = (text or "").strip().replace(" ", "")
    for src, dst in (
        ("舵肌", "舵机"),
        ("多机", "舵机"),
        ("簸箕", "舵机"),
        ("分发", "发放"),
        ("发一个药", "发放一个药品"),
        ("出药", "发放一个药品"),
        ("赞一个药", "发放一个药品"),
        ("换一个药", "发放一个药品"),
        ("设体温", "测体温"),
        ("侧温", "测温"),
        ("直接分放", "直接发放"),
    ):
        t = t.replace(src, dst)
    return t


def _func_handler_ready_for_device_mcp(conn) -> bool:
    fh = getattr(conn, "func_handler", None)
    if not fh:
        return False
    if fh.finish_init:
        return True
    mc = getattr(conn, "mcp_client", None)
    return bool(mc and getattr(mc, "tools", None))


def _user_voice_command_start_mlx_module(query: str) -> bool:
    """直连 MCP 指令一：启动测温模块，报出体温数据。"""
    t = _normalize_voice_command_text(query)
    if not t:
        return False
    if "启动测温模块，报出体温数据" in t:
        return True
    return "启动测温模块" in t and ("报出体温" in t or "体温数据" in t)


def _user_voice_command_open_mlx_temp_feature(query: str) -> bool:
    """直连 MCP：打开测温功能（MLX90614）。"""
    t = _normalize_voice_command_text(query)
    if not t:
        return False
    if "打开测温功能" in t:
        return True
    return "打开" in t and "测温" in t and "功能" in t


def _user_voice_command_dispense_medicine_360(query: str) -> bool:
    """直连 MCP 指令二：舵机旋转360度，发放一个药品（兼容 ASR 误识别）。"""
    t = _normalize_voice_command_text(query)
    if not t:
        return False
    if "舵机旋转360度，发放一个药品" in t:
        return True
    if "360度发" in t and ("舵机" in t or "舵" in t):
        return True
    has_rotate = "旋转360度" in t or (
        "360度" in t and ("旋转" in t or "转" in t)
    )
    has_dispense = "发放" in t and ("药品" in t or "药" in t)
    has_servo = "舵机" in t
    return (has_rotate and has_dispense) or (has_servo and has_dispense and "360" in t)


def _user_voice_command_direct_dispense_medicine(query: str) -> bool:
    """直连 MCP：直接发放药品（舵机旋转 360 度出药）。"""
    t = _normalize_voice_command_text(query)
    if not t:
        return False
    if "直接发放药品" in t:
        return True
    return "直接" in t and "发放" in t and ("药品" in t or "药" in t)


def _user_voice_command_adjust_volume(query: str) -> bool:
    """调节设备扬声器音量（设备 MCP self.audio_speaker.set_volume）。"""
    t = _normalize_voice_command_text(query)
    if "音量" not in t and "声音" not in t:
        return False
    return any(
        k in t
        for k in (
            "调",
            "高",
            "低",
            "大",
            "小",
            "响",
            "轻",
            "设置",
            "一点",
            "百分之",
            "%",
        )
    )


def _volume_arguments_from_voice_command(query: str) -> dict:
    t = _normalize_voice_command_text(query)
    m = re.search(r"音量?\s*到?\s*(\d{1,3})", t)
    if not m:
        m = re.search(r"(\d{1,3})\s*%", t)
    if m:
        return {"volume": min(100, max(0, int(m.group(1))))}
    if any(k in t for k in ("调高", "调大", "大一点", "大声", "响一点", "加大", "高一些")):
        return {"volume": 85}
    if any(k in t for k in ("调低", "调小", "小声", "轻一点", "减小", "低一些")):
        return {"volume": 35}
    return {"volume": 75}


def _resolve_set_volume_tool_name(conn, functions=None) -> Optional[str]:
    return _resolve_device_mcp_tool_name(
        conn,
        "set_volume",
        "audio_speaker",
        canonical_dot_name="self.audio_speaker.set_volume",
    )


def _is_set_volume_tool(tool_name: str) -> bool:
    low = (tool_name or "").lower()
    return "set_volume" in low and "audio" in low


def _speak_direct_mcp_tool_result(conn, tool_name: str, result) -> None:
    if result.action == Action.ERROR:
        text = result.response or result.result or "操作失败"
        conn.tts.tts_one_sentence(conn, ContentType.TEXT, content_detail=text)
        conn.dialogue.put(Message(role="assistant", content=text))
        return
    say = _extract_tool_json_say(result.result)
    if not say and result.result:
        raw = (result.result or "").strip()
        try:
            inner = json.loads(raw)
            if isinstance(inner, dict):
                say = inner.get("message") or inner.get("say")
                if isinstance(say, str):
                    say = say.strip()
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    if not say and result.response:
        say = str(result.response).strip()
    if say:
        conn.tts.tts_one_sentence(conn, ContentType.TEXT, content_detail=say)
        conn.dialogue.put(Message(role="assistant", content=say))


def _build_direct_mcp_voice_command_calls(conn, plain_query: str) -> list:
    """解析固定语音指令，返回待执行的 MCP tool_calls 列表。"""
    if not (plain_query or "").strip():
        return []
    if not _func_handler_ready_for_device_mcp(conn):
        return []

    fh = conn.func_handler
    _wait_device_mcp_ready_sync(conn, timeout=8.0)
    fh.tool_manager.refresh_tools()
    functions = fh.get_functions()
    norm = _normalize_voice_command_text(plain_query)

    if _user_voice_command_start_mlx_module(norm):
        mlx_tool = _resolve_mlx90614_tool_name(conn, functions)
        if mlx_tool and fh.has_tool(mlx_tool):
            conn.logger.bind(tag=TAG).info(
                "命中语音指令「启动测温模块，报出体温数据」"
            )
            return [
                {
                    "id": uuid.uuid4().hex,
                    "name": mlx_tool,
                    "arguments": "{}",
                }
            ]
        conn.logger.bind(tag=TAG).warning(
            "指令「启动测温模块，报出体温数据」命中，但未找到 MLX90614 测温工具"
        )
        return []

    if _user_voice_command_open_mlx_temp_feature(norm):
        mlx_tool = _resolve_mlx90614_tool_name(conn, functions)
        if mlx_tool and fh.has_tool(mlx_tool):
            conn.logger.bind(tag=TAG).info(
                "命中语音指令「打开测温功能」(ASR原文: %s)" % plain_query
            )
            return [
                {
                    "id": uuid.uuid4().hex,
                    "name": mlx_tool,
                    "arguments": "{}",
                }
            ]
        conn.logger.bind(tag=TAG).warning(
            "指令「打开测温功能」命中，但未找到 MLX90614 测温工具"
        )
        return []

    if _user_voice_command_dispense_medicine_360(norm):
        disp_tool = _resolve_dispense_medicine_tool_name(conn, functions)
        if disp_tool and fh.has_tool(disp_tool):
            conn.logger.bind(tag=TAG).info(
                "命中语音指令「舵机旋转360度，发放一个药品」(ASR原文: %s)"
                % plain_query
            )
            return [
                {
                    "id": uuid.uuid4().hex,
                    "name": disp_tool,
                    "arguments": "{}",
                }
            ]
        _log_serial_servo_tools_missing(
            conn,
            "指令「舵机旋转360度，发放一个药品」命中，但未找到 dispense_medicine 工具",
        )
        return []

    if _user_voice_command_direct_dispense_medicine(norm):
        disp_tool = _resolve_dispense_medicine_tool_name(conn, functions)
        if disp_tool and fh.has_tool(disp_tool):
            conn.logger.bind(tag=TAG).info(
                "命中语音指令「直接发放药品」(ASR原文: %s)" % plain_query
            )
            return [
                {
                    "id": uuid.uuid4().hex,
                    "name": disp_tool,
                    "arguments": "{}",
                }
            ]
        _log_serial_servo_tools_missing(
            conn,
            "指令「直接发放药品」命中，但未找到 dispense_medicine 工具",
        )
        return []

    if _user_voice_command_adjust_volume(norm):
        vol_tool = _resolve_set_volume_tool_name(conn, functions)
        if vol_tool and fh.has_tool(vol_tool):
            vol_args = _volume_arguments_from_voice_command(norm)
            conn.logger.bind(tag=TAG).info(
                "命中语音指令调节音量: %s -> %s" % (plain_query, vol_args)
            )
            return [
                {
                    "id": uuid.uuid4().hex,
                    "name": vol_tool,
                    "arguments": json.dumps(vol_args, ensure_ascii=False),
                }
            ]
        conn.logger.bind(tag=TAG).warning(
            "音量调节指令命中，但未找到 self.audio_speaker.set_volume"
        )
        return []

    return []


def _find_mlx90614_body_temperature_tool_name(functions) -> Optional[str]:
    """从当前轮 LLM 可用 functions 中找出 MLX90614 体表测温工具名（sanitize 后）。"""
    if not functions:
        return None
    for item in functions:
        fn = item.get("function")
        if not isinstance(fn, dict):
            continue
        name = fn.get("name") or ""
        low = name.lower()
        if "mlx90614" in low and "measure" in low:
            return name
        if "measure_body_temperature" in low:
            return name
    return None


def _coze_workflow_reply_needs_mlx_measure(ai_reply: str) -> bool:
    """扣子工作流话术：引导用户靠近测温区后触发 MLX90614。"""
    t = (ai_reply or "").strip()
    if not t:
        return False
    return "靠近我的测温区" in t or "请把你的手或头部靠近" in t


def _coze_workflow_reply_needs_dispense_medicine(ai_reply: str) -> bool:
    """扣子工作流话术：确认发放碘伏敷料后触发舵机出药。"""
    t = (ai_reply or "").strip()
    if not t:
        return False
    return ("碘伏和敷料贴" in t or "自助外用药品领取" in t) and "发放" in t


def _wait_device_mcp_ready_sync(conn, timeout: float = 5.0) -> bool:
    mc = getattr(conn, "mcp_client", None)
    loop = getattr(conn, "loop", None)
    if not mc or not loop:
        return False

    async def _poll():
        for _ in range(max(1, int(timeout * 10))):
            if await mc.is_ready():
                return True
            await asyncio.sleep(0.1)
        return False

    try:
        return asyncio.run_coroutine_threadsafe(_poll(), loop).result(
            timeout=timeout + 1.0
        )
    except Exception:
        return False


def _iter_registered_tool_names(conn):
    seen = set()
    fh = getattr(conn, "func_handler", None)
    if fh:
        try:
            fh.tool_manager.refresh_tools()
            for key in fh.tool_manager.get_all_tools().keys():
                if key not in seen:
                    seen.add(key)
                    yield key
        except Exception:
            pass
    mc = getattr(conn, "mcp_client", None)
    if mc and getattr(mc, "tools", None):
        for key in mc.tools.keys():
            if key not in seen:
                seen.add(key)
                yield key


def _resolve_device_mcp_tool_name(
    conn, *needles: str, canonical_dot_name: Optional[str] = None
) -> Optional[str]:
    """
    在 tool_manager / mcp_client 中解析设备 MCP 工具名（多为 sanitize 后的下划线名）。
    """
    needles_l = [n.lower() for n in needles if n]
    if not needles_l:
        return None

    def _match(name: str) -> bool:
        low = name.lower()
        return all(n in low for n in needles_l)

    fh = getattr(conn, "func_handler", None)
    for name in _iter_registered_tool_names(conn):
        if _match(name) and (not fh or fh.has_tool(name)):
            return name

    if canonical_dot_name:
        for candidate in (
            canonical_dot_name,
            sanitize_tool_name(canonical_dot_name),
        ):
            mc = getattr(conn, "mcp_client", None)
            if mc and mc.has_tool(candidate):
                if not fh or fh.has_tool(candidate):
                    return candidate
                # 缓存未同步时仍返回 mcp_client 已知的名称
                return candidate
    return None


def _resolve_dispense_medicine_tool_name(conn, functions=None) -> Optional[str]:
    return _resolve_device_mcp_tool_name(
        conn,
        "dispense_medicine",
        canonical_dot_name="self.serial_servo.dispense_medicine",
    )


def _log_serial_servo_tools_missing(conn, context: str):
    names = [
        n
        for n in _iter_registered_tool_names(conn)
        if "serial" in n.lower() or "servo" in n.lower() or "dispense" in n.lower()
    ]
    conn.logger.bind(tag=TAG).warning(
        "%s；当前设备 serial/servo 相关工具: %s"
        % (context, names if names else "(无，请确认固件已烧录且 UART 舵机已连接)")
    )


def _coze_workflow_reply_side_effect_tool_calls(conn, ai_reply: str) -> list:
    """扣子 /run 固定回复触发的设备 MCP（不经过 LLM function_call）。"""
    calls = []
    if not (ai_reply or "").strip():
        return calls
    fh = getattr(conn, "func_handler", None)
    if not fh or not fh.finish_init:
        return calls

    _wait_device_mcp_ready_sync(conn)
    fh.tool_manager.refresh_tools()
    functions = fh.get_functions()

    if _coze_workflow_reply_needs_mlx_measure(ai_reply):
        mlx_tool = _resolve_mlx90614_tool_name(conn, functions)
        if not mlx_tool:
            mlx_tool = _resolve_device_mcp_tool_name(
                conn,
                "mlx90614",
                "measure",
                canonical_dot_name="self.env.mlx90614_measure_body_temperature",
            )
        if mlx_tool and fh.has_tool(mlx_tool):
            conn.logger.bind(tag=TAG).info(
                "扣子工作流话术触发设备 MCP 测温: %s" % mlx_tool
            )
            calls.append(
                {
                    "id": uuid.uuid4().hex,
                    "name": mlx_tool,
                    "arguments": "{}",
                }
            )
        else:
            conn.logger.bind(tag=TAG).warning(
                "扣子测温话术命中，但未找到 self.env.mlx90614_measure_body_temperature"
            )

    if _coze_workflow_reply_needs_dispense_medicine(ai_reply):
        disp_tool = _resolve_dispense_medicine_tool_name(conn, functions)
        if disp_tool and fh.has_tool(disp_tool):
            conn.logger.bind(tag=TAG).info(
                "扣子工作流话术触发设备 MCP 出药: %s" % disp_tool
            )
            calls.append(
                {
                    "id": uuid.uuid4().hex,
                    "name": disp_tool,
                    "arguments": "{}",
                }
            )
        else:
            _log_serial_servo_tools_missing(
                conn,
                "扣子出药话术命中，但未找到 dispense_medicine（需设备 MCP 暴露该工具）",
            )

    return calls


def _extract_tool_json_say(text: str) -> Optional[str]:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        inner = json.loads(raw)
        if isinstance(inner, dict):
            say = inner.get("say")
            if isinstance(say, str) and say.strip():
                return say.strip()
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return None


def _resolve_mlx90614_tool_name(conn, functions) -> Optional[str]:
    """结合 functions 列表与 tool_manager 全量表解析 MLX 工具名（缓解缓存/MCP 竞态）。"""
    name = _find_mlx90614_body_temperature_tool_name(functions)
    if name and getattr(conn, "func_handler", None) and conn.func_handler.has_tool(name):
        return name
    resolved = _resolve_device_mcp_tool_name(
        conn,
        "mlx90614",
        "measure",
        canonical_dot_name="self.env.mlx90614_measure_body_temperature",
    )
    if resolved:
        return resolved
    fh = getattr(conn, "func_handler", None)
    if not fh:
        return None
    try:
        for key in fh.tool_manager.get_all_tools().keys():
            low = key.lower()
            if "mlx90614" in low and "measure" in low:
                return key
            if "measure_body_temperature" in low:
                return key
    except Exception:
        pass
    return None


def _parse_coze_marker_function_call(content: str) -> Optional[Dict[str, str]]:
    """
    解析扣子/部分模型以文本形式输出的工具调用：
    <|FunctionCallBegin|>...<|FunctionCallEnd|>
    """
    if not content or "<|FunctionCallBegin|>" not in content:
        return None
    if "<|FunctionCallEnd|>" not in content:
        return None
    try:
        inner = content.split("<|FunctionCallBegin|>", 1)[1].split(
            "<|FunctionCallEnd|>", 1
        )[0]
        m = re.search(r'"name"\s*:\s*"([^"]+)"', inner)
        if not m:
            return None
        name = m.group(1).strip()
        mp = re.search(r'"parameters"\s*:\s*(\{[^}]*\})', inner, re.DOTALL)
        args_str = mp.group(1) if mp else "{}"
        json.loads(args_str)
        return {"name": name, "arguments": args_str}
    except Exception:
        return None


def _is_mlx90614_body_temperature_tool(tool_name: str) -> bool:
    low = (tool_name or "").lower()
    return "measure_body_temperature" in low and "mlx90614" in low


def _is_dispense_medicine_tool(tool_name: str) -> bool:
    return "dispense_medicine" in (tool_name or "").lower()


def _format_mlx90614_tool_result_for_llm(tool_name: str, text: str) -> str:
    """
    将 MLX90614 测温 JSON 整理成易读块并附加播报约束，避免二次 LLM 只说「正常」不报具体摄氏度。
    """
    if not _is_mlx90614_body_temperature_tool(tool_name):
        return text
    hint = (
        "\n\n【播报硬性要求】你必须用口语向用户说出具体数字："
        "物体/体表温度与环境温度须各报至少一位小数的摄氏度读数；"
        "可直接复述下面「建议播报」全文；禁止仅用「正常」「没事」「体温正常」等概括而不报数值。"
    )
    raw = (text or "").strip()
    if not raw:
        return hint.strip()
    try:
        inner = json.loads(raw)
        if isinstance(inner, dict):
            lines = []
            say = inner.get("say")
            if isinstance(say, str) and say.strip():
                lines.append("建议播报（含具体度数，请尽量完整复述给用户）：" + say.strip())
            oc = inner.get("object_celsius")
            ac = inner.get("ambient_celsius")
            if oc is not None:
                lines.append(f"物体/体表温度（数值）: {oc} °C")
            if ac is not None:
                lines.append(f"环境温度（数值）: {ac} °C")
            if lines:
                return "\n".join(lines) + hint
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return raw + hint


auto_import_modules("plugins_func.functions")


class TTSException(RuntimeError):
    pass


class ConnectionHandler:
    def __init__(
            self,
            config: Dict[str, Any],
            _vad,
            _asr,
            _llm,
            _memory,
            _intent,
            server=None,
    ):
        self.common_config = config
        self.config = copy.deepcopy(config)
        self.session_id = str(uuid.uuid4())
        self.logger = setup_logging()
        self.server = server  # 保存server实例的引用

        self.need_bind = False  # 是否需要绑定设备
        self.bind_completed_event = asyncio.Event()
        self.bind_code = None  # 绑定设备的验证码
        self.last_bind_prompt_time = 0  # 上次播放绑定提示的时间戳(秒)
        self.bind_prompt_interval = 60  # 绑定提示播放间隔(秒)

        self.read_config_from_api = self.config.get("read_config_from_api", False)

        self.websocket: websockets.ServerConnection | None = None
        self.headers = None
        self.device_id = None
        self.client_ip = None
        self.prompt = None
        self.welcome_msg = None
        self.max_output_size = 0
        self.chat_history_conf = 0
        self.audio_format = "opus"
        self.sample_rate = 24000  # 默认采样率，从客户端 hello 消息中动态更新

        # 客户端状态相关
        self.client_abort = False
        self.client_is_speaking = False
        self.client_listen_mode = "auto"

        # 线程任务相关
        self.loop = None  # 在 handle_connection 中获取运行中的事件循环
        self.stop_event = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=5)

        # 添加上报线程池
        self.report_queue = queue.Queue()
        self.report_thread = None
        # 未来可以通过修改此处，调节asr的上报和tts的上报，目前默认都开启
        self.report_asr_enable = self.read_config_from_api
        self.report_tts_enable = self.read_config_from_api

        # 依赖的组件
        self.vad = None
        self.asr = None
        self.tts = None
        self._asr = _asr
        self._vad = _vad
        self.llm = _llm
        self.memory = _memory
        self.intent = _intent

        self.is_exiting = False  # 标记是否正在执行退出流程

        # 为每个连接单独管理声纹识别
        self.voiceprint_provider = None

        # vad相关变量
        self.client_audio_buffer = bytearray()
        self.client_have_voice = False
        self.client_voice_window = deque(maxlen=5)
        self.first_activity_time = 0.0  # 记录首次活动的时间（毫秒）
        self.last_activity_time = 0.0  # 统一的活动时间戳（毫秒）
        self.client_voice_stop = False
        self.last_is_voice = False

        # asr相关变量
        # 因为实际部署时可能会用到公共的本地ASR，不能把变量暴露给公共ASR
        # 所以涉及到ASR的变量，需要在这里定义，属于connection的私有变量
        self.asr_audio = []
        self.asr_audio_queue = queue.Queue()
        self.current_speaker = None  # 存储当前说话人

        # llm相关变量
        self.dialogue = Dialogue()

        # 工具调用统计（用于监控和自动恢复）
        self.tool_call_stats = {
            'last_call_turn': -1,  # 上次调用工具的轮数
            'consecutive_no_call': 0,  # 连续未调用次数
        }

        # tts相关变量
        self.sentence_id = None
        # 处理TTS响应没有文本返回
        self.tts_MessageText = ""

        # iot相关变量
        self.iot_descriptors = {}
        self.func_handler = None

        self.cmd_exit = self.config["exit_commands"]

        # 是否在聊天结束后关闭连接
        self.close_after_chat = False
        self.load_function_plugin = False
        self.intent_type = "nointent"

        self.timeout_seconds = (
                int(self.config.get("close_connection_no_voice_time", 120)) + 60
        )  # 在原来第一道关闭的基础上加60秒，进行二道关闭
        self.timeout_task = None

        # {"mcp":true} 表示启用MCP功能
        self.features = None

        # 标记连接是否来自MQTT
        self.conn_from_mqtt_gateway = False

        # 初始化提示词管理器
        self.prompt_manager = PromptManager(self.config, self.logger)

    async def handle_connection(self, ws: websockets.ServerConnection):
        try:
            # 获取运行中的事件循环（必须在异步上下文中）
            self.loop = asyncio.get_running_loop()

            # 获取并验证headers
            self.headers = dict(ws.request.headers)
            real_ip = self.headers.get("x-real-ip") or self.headers.get(
                "x-forwarded-for"
            )
            if real_ip:
                self.client_ip = real_ip.split(",")[0].strip()
            else:
                self.client_ip = ws.remote_address[0]
            self.logger.bind(tag=TAG).info(
                f"{self.client_ip} conn - Headers: {self.headers}"
            )

            self.device_id = self.headers.get("device-id", None)

            # 认证通过,继续处理
            self.websocket = ws

            # 检查是否来自MQTT连接
            request_path = ws.request.path
            self.conn_from_mqtt_gateway = request_path.endswith("?from=mqtt_gateway")
            if self.conn_from_mqtt_gateway:
                self.logger.bind(tag=TAG).info("连接来自:MQTT网关")

            # 初始化活动时间戳
            self.first_activity_time = time.time() * 1000
            self.last_activity_time = time.time() * 1000

            # 启动超时检查任务
            self.timeout_task = asyncio.create_task(self._check_timeout())

            self.welcome_msg = self.config["xiaozhi"]
            self.welcome_msg["session_id"] = self.session_id

            # 从配置中读取采样率
            self.sample_rate = self.welcome_msg["audio_params"]["sample_rate"]
            self.logger.bind(tag=TAG).info(f"配置输出音频采样率为: {self.sample_rate}")

            # 在后台初始化配置和组件（完全不阻塞主循环）
            asyncio.create_task(self._background_initialize())

            try:
                async for message in self.websocket:
                    await self._route_message(message)
            except websockets.exceptions.ConnectionClosed:
                self.logger.bind(tag=TAG).info("客户端断开连接")

        except AuthenticationError as e:
            self.logger.bind(tag=TAG).error(f"Authentication failed: {str(e)}")
            return
        except Exception as e:
            stack_trace = traceback.format_exc()
            self.logger.bind(tag=TAG).error(f"Connection error: {str(e)}-{stack_trace}")
            return
        finally:
            try:
                await self._save_and_close(ws)
            except Exception as final_error:
                self.logger.bind(tag=TAG).error(f"最终清理时出错: {final_error}")
                # 确保即使保存记忆失败，也要关闭连接
                try:
                    await self.close(ws)
                except Exception as close_error:
                    self.logger.bind(tag=TAG).error(
                        f"强制关闭连接时出错: {close_error}"
                    )

    async def _save_and_close(self, ws):
        """保存记忆并关闭连接"""
        try:
            if self.memory:
                # 使用线程池异步保存记忆
                def save_memory_task():
                    try:
                        # 创建新事件循环（避免与主循环冲突）
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(
                            self.memory.save_memory(
                                self.dialogue.dialogue, self.session_id
                            )
                        )
                    except Exception as e:
                        self.logger.bind(tag=TAG).error(f"保存记忆失败: {e}")
                    finally:
                        try:
                            loop.close()
                        except Exception:
                            pass

                # 启动线程保存记忆，不等待完成
                threading.Thread(target=save_memory_task, daemon=True).start()
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"保存记忆失败: {e}")
        finally:
            # 立即关闭连接，不等待记忆保存完成
            try:
                await self.close(ws)
            except Exception as close_error:
                self.logger.bind(tag=TAG).error(
                    f"保存记忆后关闭连接失败: {close_error}"
                )

    async def _discard_message_with_bind_prompt(self):
        """丢弃消息并检查是否需要播放绑定提示"""
        current_time = time.time()
        # 检查是否需要播放绑定提示
        if current_time - self.last_bind_prompt_time >= self.bind_prompt_interval:
            self.last_bind_prompt_time = current_time
            # 复用现有的绑定提示逻辑
            from core.handle.receiveAudioHandle import check_bind_device

            asyncio.create_task(check_bind_device(self))

    async def _route_message(self, message):
        """消息路由"""
        # 退出状态丢弃所有消息
        if self.is_exiting:
           return

        # 检查是否已经获取到真实的绑定状态
        if not self.bind_completed_event.is_set():
            # 还没有获取到真实状态，等待直到获取到真实状态或超时
            try:
                await asyncio.wait_for(self.bind_completed_event.wait(), timeout=1)
            except asyncio.TimeoutError:
                # 超时仍未获取到真实状态，丢弃消息
                await self._discard_message_with_bind_prompt()
                return

        # 已经获取到真实状态，检查是否需要绑定
        if self.need_bind:
            # 需要绑定，丢弃消息
            await self._discard_message_with_bind_prompt()
            return

        # 不需要绑定，继续处理消息

        if isinstance(message, str):
            await handleTextMessage(self, message)
        elif isinstance(message, bytes):
            if self.vad is None or self.asr is None:
                return

            # 处理来自MQTT网关的音频包
            if self.conn_from_mqtt_gateway and len(message) >= 16:
                handled = await self._process_mqtt_audio_message(message)
                if handled:
                    return

            # 不需要头部处理或没有头部时，直接处理原始消息
            self.asr_audio_queue.put(message)

    async def _process_mqtt_audio_message(self, message):
        """
        处理来自MQTT网关的音频消息，解析16字节头部并提取音频数据

        Args:
            message: 包含头部的音频消息

        Returns:
            bool: 是否成功处理了消息
        """
        try:
            # 提取头部信息
            timestamp = int.from_bytes(message[8:12], "big")
            audio_length = int.from_bytes(message[12:16], "big")

            # 提取音频数据
            if audio_length > 0 and len(message) >= 16 + audio_length:
                # 有指定长度，提取精确的音频数据
                audio_data = message[16 : 16 + audio_length]
                # 基于时间戳进行排序处理
                self._process_websocket_audio(audio_data, timestamp)
                return True
            elif len(message) > 16:
                # 没有指定长度或长度无效，去掉头部后处理剩余数据
                audio_data = message[16:]
                self.asr_audio_queue.put(audio_data)
                return True
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"解析WebSocket音频包失败: {e}")

        # 处理失败，返回False表示需要继续处理
        return False

    def _process_websocket_audio(self, audio_data, timestamp):
        """处理WebSocket格式的音频包"""
        # 初始化时间戳序列管理
        if not hasattr(self, "audio_timestamp_buffer"):
            self.audio_timestamp_buffer = {}
            self.last_processed_timestamp = 0
            self.max_timestamp_buffer_size = 20

        # 如果时间戳是递增的，直接处理
        if timestamp >= self.last_processed_timestamp:
            self.asr_audio_queue.put(audio_data)
            self.last_processed_timestamp = timestamp

            # 处理缓冲区中的后续包
            processed_any = True
            while processed_any:
                processed_any = False
                for ts in sorted(self.audio_timestamp_buffer.keys()):
                    if ts > self.last_processed_timestamp:
                        buffered_audio = self.audio_timestamp_buffer.pop(ts)
                        self.asr_audio_queue.put(buffered_audio)
                        self.last_processed_timestamp = ts
                        processed_any = True
                        break
        else:
            # 乱序包，暂存
            if len(self.audio_timestamp_buffer) < self.max_timestamp_buffer_size:
                self.audio_timestamp_buffer[timestamp] = audio_data
            else:
                self.asr_audio_queue.put(audio_data)

    async def handle_restart(self, message):
        """处理服务器重启请求"""
        try:

            self.logger.bind(tag=TAG).info("收到服务器重启指令，准备执行...")

            # 发送确认响应
            await self.websocket.send(
                json.dumps(
                    {
                        "type": "server",
                        "status": "success",
                        "message": "服务器重启中...",
                        "content": {"action": "restart"},
                    }
                )
            )

            # 异步执行重启操作
            def restart_server():
                """实际执行重启的方法"""
                time.sleep(1)
                self.logger.bind(tag=TAG).info("执行服务器重启...")
                subprocess.Popen(
                    [sys.executable, "app.py"],
                    stdin=sys.stdin,
                    stdout=sys.stdout,
                    stderr=sys.stderr,
                    start_new_session=True,
                )
                os._exit(0)

            # 使用线程执行重启避免阻塞事件循环
            threading.Thread(target=restart_server, daemon=True).start()

        except Exception as e:
            self.logger.bind(tag=TAG).error(f"重启失败: {str(e)}")
            await self.websocket.send(
                json.dumps(
                    {
                        "type": "server",
                        "status": "error",
                        "message": f"Restart failed: {str(e)}",
                        "content": {"action": "restart"},
                    }
                )
            )

    def _initialize_components(self):
        try:
            if self.tts is None:
                self.tts = self._initialize_tts()
            # 打开语音合成通道
            asyncio.run_coroutine_threadsafe(
                self.tts.open_audio_channels(self), self.loop
            )
            if self.need_bind:
                self.bind_completed_event.set()
                return
            self.selected_module_str = build_module_string(
                self.config.get("selected_module", {})
            )
            self.logger = create_connection_logger(self.selected_module_str)

            """初始化组件"""
            if self.config.get("prompt") is not None:
                user_prompt = self.config["prompt"]
                # 使用快速提示词进行初始化
                prompt = self.prompt_manager.get_quick_prompt(user_prompt)
                self.change_system_prompt(prompt)
                self.logger.bind(tag=TAG).info(
                    f"快速初始化组件: prompt成功 {prompt[:50]}..."
                )

            """初始化本地组件"""
            if self.vad is None:
                self.vad = self._vad
            if self.asr is None:
                self.asr = self._initialize_asr()

            # 初始化声纹识别
            self._initialize_voiceprint()
            # 打开语音识别通道
            asyncio.run_coroutine_threadsafe(
                self.asr.open_audio_channels(self), self.loop
            )

            """加载记忆"""
            self._initialize_memory()
            """加载意图识别"""
            self._initialize_intent()
            """初始化上报线程"""
            self._init_report_threads()
            """更新系统提示词"""
            self._init_prompt_enhancement()

        except Exception as e:
            self.logger.bind(tag=TAG).error(f"实例化组件失败: {e}")

    def _init_prompt_enhancement(self):

        # 更新上下文信息
        self.prompt_manager.update_context_info(self, self.client_ip)
        enhanced_prompt = self.prompt_manager.build_enhanced_prompt(
            self.config["prompt"], self.device_id, self.client_ip
        )
        if enhanced_prompt:
            self.change_system_prompt(enhanced_prompt)
            self.logger.bind(tag=TAG).debug("系统提示词已增强更新")

    def _init_report_threads(self):
        """初始化ASR和TTS上报线程"""
        if not self.read_config_from_api or self.need_bind:
            return
        if self.chat_history_conf == 0:
            return
        if self.report_thread is None or not self.report_thread.is_alive():
            self.report_thread = threading.Thread(
                target=self._report_worker, daemon=True
            )
            self.report_thread.start()
            self.logger.bind(tag=TAG).info("TTS上报线程已启动")

    def _initialize_tts(self):
        """初始化TTS"""
        tts = None
        if not self.need_bind:
            tts = initialize_tts(self.config)

        if tts is None:
            tts = DefaultTTS(self.config, delete_audio_file=True)

        return tts

    def _initialize_asr(self):
        """初始化ASR"""
        if (
                self._asr is not None
                and hasattr(self._asr, "interface_type")
                and self._asr.interface_type == InterfaceType.LOCAL
        ):
            # 如果公共ASR是本地服务，则直接返回
            # 因为本地一个实例ASR，可以被多个连接共享
            asr = self._asr
        else:
            # 如果公共ASR是远程服务，则初始化一个新实例
            # 因为远程ASR，涉及到websocket连接和接收线程，需要每个连接一个实例
            asr = initialize_asr(self.config)

        return asr

    def _initialize_voiceprint(self):
        """为当前连接初始化声纹识别"""
        try:
            voiceprint_config = self.config.get("voiceprint", {})
            if voiceprint_config:
                voiceprint_provider = VoiceprintProvider(voiceprint_config)
                if voiceprint_provider is not None and voiceprint_provider.enabled:
                    self.voiceprint_provider = voiceprint_provider
                    self.logger.bind(tag=TAG).info("声纹识别功能已在连接时动态启用")
                else:
                    self.logger.bind(tag=TAG).warning("声纹识别功能启用但配置不完整")
            else:
                self.logger.bind(tag=TAG).info("声纹识别功能未启用")
        except Exception as e:
            self.logger.bind(tag=TAG).warning(f"声纹识别初始化失败: {str(e)}")

    async def _background_initialize(self):
        """在后台初始化配置和组件（完全不阻塞主循环）"""
        try:
            # 异步获取差异化配置
            await self._initialize_private_config_async()
            # 在线程池中初始化组件
            self.executor.submit(self._initialize_components)
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"后台初始化失败: {e}")

    async def _initialize_private_config_async(self):
        """从接口异步获取差异化配置（异步版本，不阻塞主循环）"""
        if not self.read_config_from_api:
            self.need_bind = False
            self.bind_completed_event.set()
            return
        try:
            begin_time = time.time()
            private_config = await get_private_config_from_api(
                self.config,
                self.headers.get("device-id"),
                self.headers.get("client-id", self.headers.get("device-id")),
            )
            private_config["delete_audio"] = bool(self.config.get("delete_audio", True))
            self.logger.bind(tag=TAG).info(
                f"{time.time() - begin_time} 秒，异步获取差异化配置成功: {json.dumps(filter_sensitive_info(private_config), ensure_ascii=False)}"
            )
            self.need_bind = False
            self.bind_completed_event.set()
        except DeviceNotFoundException as e:
            self.need_bind = True
            private_config = {}
        except DeviceBindException as e:
            self.need_bind = True
            self.bind_code = e.bind_code
            private_config = {}
        except Exception as e:
            self.need_bind = True
            self.logger.bind(tag=TAG).error(f"异步获取差异化配置失败: {e}")
            private_config = {}

        init_llm, init_tts, init_memory, init_intent = (
            False,
            False,
            False,
            False,
        )

        init_vad = check_vad_update(self.common_config, private_config)
        init_asr = check_asr_update(self.common_config, private_config)

        if init_vad:
            self.config["VAD"] = private_config["VAD"]
            self.config["selected_module"]["VAD"] = private_config["selected_module"][
                "VAD"
            ]
        if init_asr:
            self.config["ASR"] = private_config["ASR"]
            self.config["selected_module"]["ASR"] = private_config["selected_module"][
                "ASR"
            ]
        if private_config.get("TTS", None) is not None:
            init_tts = True
            self.config["TTS"] = private_config["TTS"]
            self.config["selected_module"]["TTS"] = private_config["selected_module"][
                "TTS"
            ]
        if private_config.get("LLM", None) is not None:
            init_llm = True
            self.config["LLM"] = private_config["LLM"]
            self.config["selected_module"]["LLM"] = private_config["selected_module"][
                "LLM"
            ]
        if private_config.get("VLLM", None) is not None:
            self.config["VLLM"] = private_config["VLLM"]
            self.config["selected_module"]["VLLM"] = private_config["selected_module"][
                "VLLM"
            ]
        if private_config.get("Memory", None) is not None:
            init_memory = True
            self.config["Memory"] = private_config["Memory"]
            self.config["selected_module"]["Memory"] = private_config[
                "selected_module"
            ]["Memory"]
        if private_config.get("Intent", None) is not None:
            init_intent = True
            self.config["Intent"] = private_config["Intent"]
            model_intent = private_config.get("selected_module", {}).get("Intent", {})
            self.config["selected_module"]["Intent"] = model_intent
            # 加载插件配置
            if model_intent != "Intent_nointent":
                plugin_from_server = private_config.get("plugins", {})
                for plugin, config_str in plugin_from_server.items():
                    plugin_from_server[plugin] = json.loads(config_str)
                self.config["plugins"] = plugin_from_server
                self.config["Intent"][self.config["selected_module"]["Intent"]][
                    "functions"
                ] = plugin_from_server.keys()
        if private_config.get("prompt", None) is not None:
            self.config["prompt"] = private_config["prompt"]
        # 获取声纹信息
        if private_config.get("voiceprint", None) is not None:
            self.config["voiceprint"] = private_config["voiceprint"]
        if private_config.get("summaryMemory", None) is not None:
            self.config["summaryMemory"] = private_config["summaryMemory"]
        if private_config.get("device_max_output_size", None) is not None:
            self.max_output_size = int(private_config["device_max_output_size"])
        if private_config.get("chat_history_conf", None) is not None:
            self.chat_history_conf = int(private_config["chat_history_conf"])
        if private_config.get("mcp_endpoint", None) is not None:
            self.config["mcp_endpoint"] = private_config["mcp_endpoint"]
        if private_config.get("context_providers", None) is not None:
            self.config["context_providers"] = private_config["context_providers"]

        # 使用 run_in_executor 在线程池中执行 initialize_modules，避免阻塞主循环
        try:
            modules = await self.loop.run_in_executor(
                None,  # 使用默认线程池
                initialize_modules,
                self.logger,
                private_config,
                init_vad,
                init_asr,
                init_llm,
                init_tts,
                init_memory,
                init_intent,
            )
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"初始化组件失败: {e}")
            modules = {}
        if modules.get("tts", None) is not None:
            self.tts = modules["tts"]
        if modules.get("vad", None) is not None:
            self.vad = modules["vad"]
        if modules.get("asr", None) is not None:
            self.asr = modules["asr"]
        if modules.get("llm", None) is not None:
            self.llm = modules["llm"]
        if modules.get("intent", None) is not None:
            self.intent = modules["intent"]
        if modules.get("memory", None) is not None:
            self.memory = modules["memory"]

    def _initialize_memory(self):
        if self.memory is None:
            return
        """初始化记忆模块"""
        self.memory.init_memory(
            role_id=self.device_id,
            llm=self.llm,
            summary_memory=self.config.get("summaryMemory", None),
            save_to_file=not self.read_config_from_api,
        )

        # 获取记忆总结配置
        memory_config = self.config["Memory"]
        memory_type = self.config["Memory"][self.config["selected_module"]["Memory"]][
            "type"
        ]
        # 如果使用 nomen 或 mem_report_only，直接返回
        if memory_type == "nomem" or memory_type == "mem_report_only":
            return
        # 使用 mem_local_short 模式
        elif memory_type == "mem_local_short":
            memory_llm_name = memory_config[self.config["selected_module"]["Memory"]][
                "llm"
            ]
            if memory_llm_name and memory_llm_name in self.config["LLM"]:
                # 如果配置了专用LLM，则创建独立的LLM实例
                from core.utils import llm as llm_utils

                memory_llm_config = self.config["LLM"][memory_llm_name]
                memory_llm_type = memory_llm_config.get("type", memory_llm_name)
                memory_llm = llm_utils.create_instance(
                    memory_llm_type, memory_llm_config
                )
                self.logger.bind(tag=TAG).info(
                    f"为记忆总结创建了专用LLM: {memory_llm_name}, 类型: {memory_llm_type}"
                )
                self.memory.set_llm(memory_llm)
            else:
                # 否则使用主LLM
                self.memory.set_llm(self.llm)
                self.logger.bind(tag=TAG).info("使用主LLM作为意图识别模型")

    def _initialize_intent(self):
        if self.intent is None:
            return
        self.intent_type = self.config["Intent"][
            self.config["selected_module"]["Intent"]
        ]["type"]
        if self.intent_type == "function_call" or self.intent_type == "intent_llm":
            self.load_function_plugin = True
        """初始化意图识别模块"""
        # 获取意图识别配置
        intent_config = self.config["Intent"]
        intent_type = self.config["Intent"][self.config["selected_module"]["Intent"]][
            "type"
        ]

        # 如果使用 nointent，直接返回
        if intent_type == "nointent":
            return
        # 使用 intent_llm 模式
        elif intent_type == "intent_llm":
            intent_llm_name = intent_config[self.config["selected_module"]["Intent"]][
                "llm"
            ]

            if intent_llm_name and intent_llm_name in self.config["LLM"]:
                # 如果配置了专用LLM，则创建独立的LLM实例
                from core.utils import llm as llm_utils

                intent_llm_config = self.config["LLM"][intent_llm_name]
                intent_llm_type = intent_llm_config.get("type", intent_llm_name)
                intent_llm = llm_utils.create_instance(
                    intent_llm_type, intent_llm_config
                )
                self.logger.bind(tag=TAG).info(
                    f"为意图识别创建了专用LLM: {intent_llm_name}, 类型: {intent_llm_type}"
                )
                self.intent.set_llm(intent_llm)
            else:
                # 否则使用主LLM
                self.intent.set_llm(self.llm)
                self.logger.bind(tag=TAG).info("使用主LLM作为意图识别模型")

        """加载统一工具处理器"""
        self.func_handler = UnifiedToolHandler(self)

        # 异步初始化工具处理器
        if hasattr(self, "loop") and self.loop:
            asyncio.run_coroutine_threadsafe(self.func_handler._initialize(), self.loop)

    def change_system_prompt(self, prompt):
        self.prompt = prompt
        # 更新系统prompt至上下文
        self.dialogue.update_system_message(self.prompt)

    def chat(self, query, depth=0):
        if query is not None:
            self.logger.bind(tag=TAG).info(f"大模型收到用户消息: {query}")

        # 为最顶层时新建会话ID和发送FIRST请求
        if depth == 0:
            self.sentence_id = str(uuid.uuid4().hex)
            self.dialogue.put(Message(role="user", content=query))
            self.tts.tts_text_queue.put(
                TTSMessageDTO(
                    sentence_id=self.sentence_id,
                    sentence_type=SentenceType.FIRST,
                    content_type=ContentType.ACTION,
                )
            )
            plain_query = _extract_plain_query_for_tool_routing(query or "")
            if self._try_run_direct_mcp_voice_command(plain_query, depth=depth):
                return True

        # 设置最大递归深度，避免无限循环，可根据实际需求调整
        MAX_DEPTH = 5
        force_final_answer = False  # 标记是否强制最终回答

        if depth >= MAX_DEPTH:
            self.logger.bind(tag=TAG).debug(
                f"已达到最大工具调用深度 {MAX_DEPTH}，将强制基于现有信息回答"
            )
            force_final_answer = True
            # 添加系统指令，要求 LLM 基于现有信息回答
            self.dialogue.put(
                Message(
                    role="user",
                    content="[系统提示] 已达到最大工具调用次数限制，请你基于目前已经获取的所有信息，直接给出最终答案。不要再尝试调用任何工具。",
                )
            )

        # 长对话工具调用提醒：当对话轮数较多时，提醒模型正确使用工具
        force_reminder = False  # 是否强制提醒

        if depth == 0 and query is not None:
            dialogue_length = len(self.dialogue.dialogue)
            current_turn = dialogue_length // 2

            # 检测距离上一次连续未调用工具的情况
            if self.tool_call_stats['last_call_turn'] >= 0:
                turns_since_last = current_turn - self.tool_call_stats['last_call_turn']
                if turns_since_last > 3:  # 超过3轮未调用
                    self.logger.bind(tag=TAG).warning(
                        f"检测到{turns_since_last}轮未调用工具，可能进入偷懒模式，将强制注入提醒"
                    )
                    force_reminder = True

            # 对话历史截断：防止历史过长导致模型"偷懒模式"扩散
            # 当对话历史超过阈值时，保留最近的 10 轮对话
            # max_dialogue_turns = 10
            # if dialogue_length > max_dialogue_turns * 2:
            #     removed = self.dialogue.trim_history(max_turns=max_dialogue_turns)
            #     if removed > 0:
            #         self.logger.bind(tag=TAG).info(
            #             f"对话历史过长({dialogue_length}条)，已智能截断保留最近{max_dialogue_turns}轮，移除{removed}条消息"
            #         )

        # Define intent functions
        functions = None
        # 达到最大深度时，禁用工具调用，强制 LLM 直接回答
        if (
                self.intent_type == "function_call"
                and hasattr(self, "func_handler")
                and not force_final_answer
        ):
            functions = self.func_handler.get_functions()

        # 扣子 /run 由工作流维护对话，不注入 MCP 工具提醒
        coze_workflow_only = _main_llm_is_coze_workflow_run(self.config)

        # 长对话工具调用规则强化：动态生成基于当前可用工具的提醒
        tool_call_reminder = None
        if (
            depth == 0
            and query is not None
            and functions is not None
            and not coze_workflow_only
        ):
            dialogue_length = len(self.dialogue.dialogue)
            # 当对话历史超过4条消息时，注入规则强化
            if dialogue_length > 4:
                tool_summary = self._get_tool_summary(functions)
                if tool_summary:
                    # 根据对话长度和偷懒检测，使用不同强度的提醒
                    if force_reminder:
                        # 强提醒 - 包含完整规则前缀
                        tool_call_reminder = (
                            TOOL_CALLING_RULES +
                            f"[重要提醒] 多轮未使用工具，检查回复是否遗漏了必要的工具调用！上一轮未使用工具，本轮必须重新判断是否需要工具。"
                            f"当前可用工具: {tool_summary}。"
                        )
                        reminder_level = "强"
                    else:
                        # 中等提醒 - 包含规则前缀
                        tool_call_reminder = (
                            TOOL_CALLING_RULES +
                            f"当前可用工具: {tool_summary}。"
                            f"仅当用户请求涉及实时信息查询或执行操作时调用，日常对话无需调用。"
                        )
                        reminder_level = "中"
                    self.logger.bind(tag=TAG).debug(
                        f"对话历史较长({dialogue_length}条)，已注入{reminder_level}等级工具调用规则强化，当前可用工具：{tool_summary}"
                    )

        response_message = []

        # 如果有工具调用提醒，临时添加到对话中（标记为临时消息）
        if tool_call_reminder:
            self.dialogue.put(Message(role="user", content=tool_call_reminder, is_temporary=True))

        tool_call_flag = False
        tool_calls_list = []
        content_arguments = ""
        self.client_abort = False
        emotion_flag = True

        try:
            memory_str = None
            if self.memory is not None and query:
                future = asyncio.run_coroutine_threadsafe(
                    self.memory.query_memory(query), self.loop
                )
                memory_str = future.result()

            if coze_workflow_only:
                llm_responses = self.llm.response(
                    self.session_id,
                    self.dialogue.get_llm_dialogue_with_memory(
                        memory_str, self.config.get("voiceprint", {})
                    ),
                    plain_query=_extract_plain_query_for_tool_routing(query or ""),
                )
            elif self.intent_type == "function_call" and functions is not None:
                from plugins_func.functions.campus_medical_triage import (
                    should_run_deterministic_triage,
                )

                plain_query = _extract_plain_query_for_tool_routing(query)
                mlx_injected = False
                functions_for_coze = functions

                if (
                    depth == 0
                    and plain_query
                    and _user_intent_body_temperature_measure(plain_query)
                ):
                    self.func_handler.tool_manager.refresh_tools()
                    functions_for_coze = self.func_handler.get_functions()
                    mlx_tool = _resolve_mlx90614_tool_name(self, functions_for_coze)
                    if mlx_tool and self.func_handler.has_tool(mlx_tool):
                        self.logger.bind(tag=TAG).info(
                            "用户使用测体温相关说法且设备提供 MLX90614 工具；"
                            "主 LLM 无标准 function_call 输出时直接下发设备 MCP 调用"
                        )
                        tool_call_flag = True
                        tool_calls_list = [
                            {
                                "id": uuid.uuid4().hex,
                                "name": mlx_tool,
                                "arguments": "{}",
                            }
                        ]
                        llm_responses = iter(())
                        mlx_injected = True
                    else:
                        self.logger.bind(tag=TAG).warning(
                            "测体温意图命中但未找到设备 MCP 中的 MLX90614 工具，回退主模型。"
                        )

                if not mlx_injected and (
                    depth == 0
                    and plain_query
                    and self.func_handler.has_tool("campus_medical_triage")
                    and should_run_deterministic_triage(self, plain_query)
                ):
                    self.logger.bind(tag=TAG).info(
                        "命中校园医疗分诊关键词，跳过主 LLM 工具路由，直接调用 campus_medical_triage"
                    )
                    tool_call_flag = True
                    tool_calls_list = [
                        {
                            "id": uuid.uuid4().hex,
                            "name": "campus_medical_triage",
                            "arguments": json.dumps(
                                {"user_text": plain_query}, ensure_ascii=False
                            ),
                        }
                    ]
                    llm_responses = iter(())
                elif not mlx_injected:
                    llm_responses = self.llm.response_with_functions(
                        self.session_id,
                        self.dialogue.get_llm_dialogue_with_memory(
                            memory_str, self.config.get("voiceprint", {})
                        ),
                        functions=functions_for_coze,
                    )
            else:
                llm_responses = self.llm.response(
                    self.session_id,
                    self.dialogue.get_llm_dialogue_with_memory(
                        memory_str, self.config.get("voiceprint", {})
                    ),
                )
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"LLM 处理出错 {query}: {e}")
            return None

        # 处理流式响应
        try:
            for response in llm_responses:
                if self.client_abort:
                    break
                if (
                    self.intent_type == "function_call"
                    and functions is not None
                    and not coze_workflow_only
                ):
                    content, tools_call = response
                    if "content" in response:
                        content = response["content"]
                        tools_call = None
                    if content is not None and len(content) > 0:
                        if isinstance(content, bytes):
                            content = content.decode("utf-8", errors="replace")
                        elif not isinstance(content, str):
                            content = str(content)
                        content_arguments += content

                    if not tool_call_flag and (
                        content_arguments.startswith("<tool_call>")
                        or "<|FunctionCallBegin|>" in content_arguments
                    ):
                        tool_call_flag = True

                    if tools_call is not None and len(tools_call) > 0:
                        tool_call_flag = True
                        self._merge_tool_calls(tool_calls_list, tools_call)
                else:
                    content = response
                    if isinstance(content, bytes):
                        content = content.decode("utf-8", errors="replace")
                    elif content is not None and not isinstance(content, str):
                        content = str(content)

                # 在llm回复中获取情绪表情，一轮对话只在开头获取一次
                if emotion_flag and content is not None and content.strip():
                    asyncio.run_coroutine_threadsafe(
                        textUtils.get_emotion(self, content),
                        self.loop,
                    )
                    emotion_flag = False

                if content is not None and len(content) > 0:
                    if not tool_call_flag:
                        response_message.append(content)
                        self.tts.tts_text_queue.put(
                            TTSMessageDTO(
                                sentence_id=self.sentence_id,
                                sentence_type=SentenceType.MIDDLE,
                                content_type=ContentType.TEXT,
                                content_detail=content,
                            )
                        )
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"LLM stream processing error: {e}")
            self.tts.tts_text_queue.put(
                TTSMessageDTO(
                    sentence_id=self.sentence_id,
                    sentence_type=SentenceType.MIDDLE,
                    content_type=ContentType.TEXT,
                    content_detail=get_system_error_response(self.config),
                )
            )
            if depth == 0:
                self.tts.tts_text_queue.put(
                    TTSMessageDTO(
                        sentence_id=self.sentence_id,
                        sentence_type=SentenceType.LAST,
                        content_type=ContentType.ACTION,
                    )
                )
            return

        # 扣子 /run：按工作流固定话术触发设备 MCP（不测用户意图、不注入 tool_calling）
        if (
            coze_workflow_only
            and depth == 0
            and not tool_call_flag
            and len(response_message) > 0
        ):
            side_calls = _coze_workflow_reply_side_effect_tool_calls(
                self, "".join(response_message)
            )
            if side_calls:
                tool_call_flag = True
                tool_calls_list = side_calls

        # 处理function call
        if tool_call_flag:
            bHasError = False
            # 处理基于文本的工具调用格式
            if len(tool_calls_list) == 0 and content_arguments:
                a = extract_json_from_string(content_arguments)
                coze_fc = _parse_coze_marker_function_call(content_arguments)
                if a is not None:
                    try:
                        content_arguments_json = json.loads(a)
                        tool_calls_list.append(
                            {
                                "id": str(uuid.uuid4().hex),
                                "name": content_arguments_json["name"],
                                "arguments": json.dumps(
                                    content_arguments_json["arguments"],
                                    ensure_ascii=False,
                                ),
                            }
                        )
                    except Exception as e:
                        bHasError = True
                        response_message.append(a)
                elif coze_fc is not None and self.func_handler.has_tool(
                    coze_fc["name"]
                ):
                    tool_calls_list.append(
                        {
                            "id": str(uuid.uuid4().hex),
                            "name": coze_fc["name"],
                            "arguments": coze_fc["arguments"],
                        }
                    )
                    self.logger.bind(tag=TAG).info(
                        "从模型文本中解析到 Coze 风格工具调用: %s"
                        % coze_fc["name"]
                    )
                elif tool_call_flag:
                    bHasError = True
                    response_message.append(content_arguments)
                if bHasError:
                    self.logger.bind(tag=TAG).error(
                        f"function call error: {content_arguments}"
                    )

            if not bHasError and len(tool_calls_list) > 0:
                self.logger.bind(tag=TAG).debug(
                    f"检测到 {len(tool_calls_list)} 个工具调用"
                )

                # 更新工具调用统计
                if depth == 0:
                    current_turn = len(self.dialogue.dialogue) // 2
                    self.tool_call_stats['last_call_turn'] = current_turn
                    self.tool_call_stats['consecutive_no_call'] = 0
                    self.logger.bind(tag=TAG).debug(
                        f"工具调用统计更新: 当前轮次={current_turn}"
                    )

                # 如需要大模型先处理一轮，添加相关处理后的日志情况
                if len(response_message) > 0:
                    text_buff = "".join(response_message)
                    self.tts_MessageText = text_buff
                    self.dialogue.put(Message(role="assistant", content=text_buff))
                response_message.clear()

                # 收集所有工具调用的 Future
                futures_with_data = []
                for tool_call_data in tool_calls_list:
                    self.logger.bind(tag=TAG).debug(
                        f"function_name={tool_call_data['name']}, function_id={tool_call_data['id']}, function_arguments={tool_call_data['arguments']}"
                    )

                    # 使用公共方法上报工具调用
                    tool_input = json.loads(tool_call_data.get("arguments") or "{}")
                    enqueue_tool_report(self, tool_call_data['name'], tool_input)

                    future = asyncio.run_coroutine_threadsafe(
                        self.func_handler.handle_llm_function_call(
                            self, tool_call_data
                        ),
                        self.loop,
                    )
                    futures_with_data.append((future, tool_call_data, tool_input))

                # 工具调用超时时间，可配置，默认30秒
                tool_call_timeout = int(self.config.get("tool_call_timeout", 30))
                # 等待协程结束（实际等待时长为最慢的那个）
                tool_results = []

                for future, tool_call_data, tool_input in futures_with_data:
                    try:
                        result = future.result(timeout=tool_call_timeout)
                        tool_results.append((result, tool_call_data))
                        # 使用公共方法上报工具调用结果
                        enqueue_tool_report(self, tool_call_data['name'], tool_input, str(result.result) if result.result else None, report_tool_call=False)

                    except Exception as e:
                        self.logger.bind(tag=TAG).error(
                            f"工具调用超时或异常: {tool_call_data['name']}, 错误: {e}"
                        )
                        # 超时时返回错误响应，避免整个流程卡死
                        tool_results.append((
                            ActionResponse(action=Action.ERROR, result="哎呀，网络遇到点问题，请稍后再试下！"),
                            tool_call_data
                        ))
                        # 上报工具调用错误
                        enqueue_tool_report(self, tool_call_data['name'], tool_input, str(e), report_tool_call=False)

                # 统一处理工具调用结果
                if tool_results:
                    self._handle_function_result(tool_results, depth=depth)

        # 存储对话内容
        if len(response_message) > 0:
            text_buff = "".join(response_message)
            self.tts_MessageText = text_buff
            self.dialogue.put(Message(role="assistant", content=text_buff))

            # 更新工具调用统计：如果没有调用工具，增加计数
            if depth == 0 and not tool_call_flag:
                self.tool_call_stats['consecutive_no_call'] += 1

        if depth == 0:
            self.tts.tts_text_queue.put(
                TTSMessageDTO(
                    sentence_id=self.sentence_id,
                    sentence_type=SentenceType.LAST,
                    content_type=ContentType.ACTION,
                )
            )
            # 使用lambda延迟计算，只有在DEBUG级别时才执行get_llm_dialogue()
            self.logger.bind(tag=TAG).debug(
                lambda: json.dumps(
                    self.dialogue.get_llm_dialogue(), indent=4, ensure_ascii=False
                )
            )

            # 清理临时插入的工具调用提醒消息（使用标记清理）
            if tool_call_reminder and len(self.dialogue.dialogue) > 0:
                original_length = len(self.dialogue.dialogue)
                self.dialogue.dialogue = [
                    msg for msg in self.dialogue.dialogue
                    if not getattr(msg, 'is_temporary', False)
                ]
                if len(self.dialogue.dialogue) < original_length:
                    self.logger.bind(tag=TAG).debug("已清理临时的工具调用提醒消息")

        return True

    def _try_run_direct_mcp_voice_command(self, plain_query: str, depth: int = 0) -> bool:
        """固定语音指令直连设备 MCP，跳过 LLM/扣子。"""
        tool_calls_list = _build_direct_mcp_voice_command_calls(self, plain_query)
        if not tool_calls_list:
            return False
        self._run_mcp_tool_calls(tool_calls_list, depth=depth)
        return True

    def _run_mcp_tool_calls(self, tool_calls_list: list, depth: int = 0):
        if not tool_calls_list or not getattr(self, "func_handler", None):
            return

        futures_with_data = []
        for tool_call_data in tool_calls_list:
            tool_input = json.loads(tool_call_data.get("arguments") or "{}")
            enqueue_tool_report(self, tool_call_data["name"], tool_input)
            future = asyncio.run_coroutine_threadsafe(
                self.func_handler.handle_llm_function_call(self, tool_call_data),
                self.loop,
            )
            futures_with_data.append((future, tool_call_data, tool_input))

        tool_call_timeout = int(self.config.get("tool_call_timeout", 30))
        tool_results = []
        for future, tool_call_data, tool_input in futures_with_data:
            try:
                result = future.result(timeout=tool_call_timeout)
                tool_results.append((result, tool_call_data))
                enqueue_tool_report(
                    self,
                    tool_call_data["name"],
                    tool_input,
                    str(result.result) if result.result else None,
                    report_tool_call=False,
                )
            except Exception as e:
                self.logger.bind(tag=TAG).error(
                    "工具调用超时或异常: %s, 错误: %s"
                    % (tool_call_data.get("name"), e)
                )
                tool_results.append(
                    (
                        ActionResponse(
                            action=Action.ERROR,
                            result="哎呀，网络遇到点问题，请稍后再试下！",
                        ),
                        tool_call_data,
                    )
                )
                enqueue_tool_report(
                    self,
                    tool_call_data["name"],
                    tool_input,
                    str(e),
                    report_tool_call=False,
                )

        if tool_results:
            self._handle_function_result(tool_results, depth=depth)
            for result, tool_call_data in tool_results:
                tool_name = tool_call_data.get("name") or ""
                if _is_mlx90614_body_temperature_tool(tool_name):
                    continue
                if _is_dispense_medicine_tool(tool_name) or _is_set_volume_tool(
                    tool_name
                ):
                    _speak_direct_mcp_tool_result(self, tool_name, result)

        if depth == 0:
            self.tts.tts_text_queue.put(
                TTSMessageDTO(
                    sentence_id=self.sentence_id,
                    sentence_type=SentenceType.LAST,
                    content_type=ContentType.ACTION,
                )
            )

    def _get_tool_summary(self, functions: list) -> str:
        """
        从工具定义中提取摘要，用于规则强化注入

        Args:
            functions: 工具列表

        Returns:
            str: 工具名称字符串
        """
        if not functions:
            return ""

        datas = []
        for func in functions:
            func_info = func.get("function", {})
            name = func_info.get("name", "")
            datas.append(name)
        result = "、".join(datas)
        return result

    def _handle_function_result(self, tool_results, depth):
        need_llm_tools = []

        for result, tool_call_data in tool_results:
            if result.action in [
                Action.RESPONSE,
                Action.NOTFOUND,
                Action.ERROR,
            ]:  # 直接回复前端
                text = result.response if result.response else result.result
                self.tts.tts_one_sentence(self, ContentType.TEXT, content_detail=text)
                self.dialogue.put(Message(role="assistant", content=text))
            elif result.action == Action.REQLLM:
                # 收集需要 LLM 处理的工具
                need_llm_tools.append((result, tool_call_data))
            else:
                pass

        if need_llm_tools and _main_llm_is_coze_workflow_run(self.config):
            remaining = []
            for result, tool_call_data in need_llm_tools:
                tool_name = tool_call_data.get("name") or ""
                if _is_mlx90614_body_temperature_tool(tool_name):
                    say = _extract_tool_json_say(result.result)
                    if not say and result.result:
                        try:
                            inner = json.loads((result.result or "").strip())
                            if isinstance(inner, dict):
                                oc = inner.get("object_celsius")
                                ac = inner.get("ambient_celsius")
                                if oc is not None and ac is not None:
                                    say = (
                                        f"红外测温约{oc}摄氏度，环境温度约{ac}摄氏度。"
                                        "如需精确体温请用医用体温计复核。"
                                    )
                        except (json.JSONDecodeError, TypeError, ValueError):
                            pass
                    if say:
                        self.tts.tts_one_sentence(
                            self, ContentType.TEXT, content_detail=say
                        )
                        self.dialogue.put(
                            Message(role="assistant", content=say)
                        )
                    continue
                if _is_dispense_medicine_tool(tool_name):
                    if result.action == Action.ERROR:
                        text = result.response or result.result or "出药失败"
                        self.tts.tts_one_sentence(
                            self, ContentType.TEXT, content_detail=text
                        )
                    continue
                if _is_set_volume_tool(tool_name):
                    _speak_direct_mcp_tool_result(self, tool_name, result)
                    continue
                remaining.append((result, tool_call_data))
            need_llm_tools = remaining

        if need_llm_tools:
            all_tool_calls = [
                {
                    "id": tool_call_data["id"],
                    "function": {
                        "arguments": (
                            "{}"
                            if tool_call_data["arguments"] == ""
                            else tool_call_data["arguments"]
                        ),
                        "name": tool_call_data["name"],
                    },
                    "type": "function",
                    "index": idx,
                }
                for idx, (_, tool_call_data) in enumerate(need_llm_tools)
            ]
            self.dialogue.put(Message(role="assistant", tool_calls=all_tool_calls))

            for result, tool_call_data in need_llm_tools:
                text = result.result
                if text is not None and len(text) > 0:
                    tool_name = tool_call_data.get("name") or ""
                    content_for_llm = _format_mlx90614_tool_result_for_llm(
                        tool_name, text
                    )
                    self.dialogue.put(
                        Message(
                            role="tool",
                            tool_call_id=(
                                str(uuid.uuid4())
                                if tool_call_data["id"] is None
                                else tool_call_data["id"]
                            ),
                            content=content_for_llm,
                        )
                    )

            self.chat(None, depth=depth + 1)

    def _report_worker(self):
        """聊天记录上报工作线程"""
        while not self.stop_event.is_set():
            try:
                # 从队列获取数据，设置超时以便定期检查停止事件
                item = self.report_queue.get(timeout=1)
                if item is None:  # 检测毒丸对象
                    break
                try:
                    # 检查线程池状态
                    if self.executor is None:
                        continue
                    # 提交任务到线程池
                    self.executor.submit(self._process_report, *item)
                except Exception as e:
                    self.logger.bind(tag=TAG).error(f"聊天记录上报线程异常: {e}")
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.bind(tag=TAG).error(f"聊天记录上报工作线程异常: {e}")

        self.logger.bind(tag=TAG).info("聊天记录上报线程已退出")

    def _process_report(self, type, text, audio_data, report_time):
        """处理上报任务"""
        try:
            # 执行异步上报（在事件循环中运行）
            asyncio.run(report(self, type, text, audio_data, report_time))
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"上报处理异常: {e}")
        finally:
            # 标记任务完成
            self.report_queue.task_done()

    def clearSpeakStatus(self):
        self.client_is_speaking = False
        self.logger.bind(tag=TAG).debug(f"清除服务端讲话状态")

    async def close(self, ws=None):
        """资源清理方法"""
        try:
            # 清理 VAD 连接资源
            if (
                    hasattr(self, "vad")
                    and self.vad
                    and hasattr(self.vad, "release_conn_resources")
            ):
                self.vad.release_conn_resources(self)

            # 清理音频缓冲区
            if hasattr(self, "audio_buffer"):
                self.audio_buffer.clear()

            # 取消超时任务
            if self.timeout_task and not self.timeout_task.done():
                self.timeout_task.cancel()
                try:
                    await self.timeout_task
                except asyncio.CancelledError:
                    pass
                self.timeout_task = None

            # 清理工具处理器资源
            if hasattr(self, "func_handler") and self.func_handler:
                try:
                    await self.func_handler.cleanup()
                except Exception as cleanup_error:
                    self.logger.bind(tag=TAG).error(
                        f"清理工具处理器时出错: {cleanup_error}"
                    )

            # 触发停止事件
            if self.stop_event:
                self.stop_event.set()

            # 清空任务队列
            self.clear_queues()

            # 关闭WebSocket连接
            try:
                if ws:
                    # 安全地检查WebSocket状态并关闭
                    try:
                        if hasattr(ws, "closed") and not ws.closed:
                            await ws.close()
                        elif hasattr(ws, "state") and ws.state.name != "CLOSED":
                            await ws.close()
                        else:
                            # 如果没有closed属性，直接尝试关闭
                            await ws.close()
                    except Exception:
                        # 如果关闭失败，忽略错误
                        pass
                elif self.websocket:
                    try:
                        if (
                                hasattr(self.websocket, "closed")
                                and not self.websocket.closed
                        ):
                            await self.websocket.close()
                        elif (
                                hasattr(self.websocket, "state")
                                and self.websocket.state.name != "CLOSED"
                        ):
                            await self.websocket.close()
                        else:
                            # 如果没有closed属性，直接尝试关闭
                            await self.websocket.close()
                    except Exception:
                        # 如果关闭失败，忽略错误
                        pass
            except Exception as ws_error:
                self.logger.bind(tag=TAG).error(f"关闭WebSocket连接时出错: {ws_error}")

            if self.tts:
                await self.tts.close()
            if self.asr:
                await self.asr.close()

            # 最后关闭线程池（避免阻塞）
            if self.executor:
                try:
                    self.executor.shutdown(wait=False)
                except Exception as executor_error:
                    self.logger.bind(tag=TAG).error(
                        f"关闭线程池时出错: {executor_error}"
                    )
                self.executor = None
            self.logger.bind(tag=TAG).info("连接资源已释放")
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"关闭连接时出错: {e}")
        finally:
            # 确保停止事件被设置
            if self.stop_event:
                self.stop_event.set()

    def clear_queues(self):
        """清空所有任务队列"""
        if self.tts:
            self.logger.bind(tag=TAG).debug(
                f"开始清理: TTS队列大小={self.tts.tts_text_queue.qsize()}, 音频队列大小={self.tts.tts_audio_queue.qsize()}"
            )

            # 使用非阻塞方式清空队列
            for q in [
                self.tts.tts_text_queue,
                self.tts.tts_audio_queue,
                self.report_queue,
            ]:
                if not q:
                    continue
                while True:
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        break

            # 重置音频流控器（取消后台任务并清空队列）
            if hasattr(self, "audio_rate_controller") and self.audio_rate_controller:
                self.audio_rate_controller.reset()
                self.logger.bind(tag=TAG).debug("已重置音频流控器")

            self.logger.bind(tag=TAG).debug(
                f"清理结束: TTS队列大小={self.tts.tts_text_queue.qsize()}, 音频队列大小={self.tts.tts_audio_queue.qsize()}"
            )

    def reset_audio_states(self):
        """
        重置所有音频相关状态(VAD + ASR)
        """
        # Reset VAD states
        self.client_audio_buffer.clear()
        self.client_have_voice = False
        self.client_voice_stop = False
        self.client_voice_window.clear()
        self.last_is_voice = False

        # Clear ASR buffers
        self.asr_audio.clear()

        self.logger.bind(tag=TAG).debug("All audio states reset.")

    def chat_and_close(self, text):
        """Chat with the user and then close the connection"""
        try:
            # Use the existing chat method
            self.chat(text)

            # After chat is complete, close the connection
            self.close_after_chat = True
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"Chat and close error: {str(e)}")

    async def _check_timeout(self):
        """检查连接超时"""
        try:
            while not self.stop_event.is_set():
                last_activity_time = self.last_activity_time
                if self.need_bind:
                    last_activity_time = self.first_activity_time

                # 检查是否超时（只有在时间戳已初始化的情况下）
                if last_activity_time > 0.0:
                    current_time = time.time() * 1000
                    if current_time - last_activity_time > self.timeout_seconds * 1000:
                        if not self.stop_event.is_set():
                            self.logger.bind(tag=TAG).info("连接超时，准备关闭")
                            # 设置停止事件，防止重复处理
                            self.stop_event.set()
                            # 使用 try-except 包装关闭操作，确保不会因为异常而阻塞
                            try:
                                await self.close(self.websocket)
                            except Exception as close_error:
                                self.logger.bind(tag=TAG).error(
                                    f"超时关闭连接时出错: {close_error}"
                                )
                        break
                # 每10秒检查一次，避免过于频繁
                await asyncio.sleep(10)
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"超时检查任务出错: {e}")
        finally:
            self.logger.bind(tag=TAG).info("超时检查任务已退出")

    def _merge_tool_calls(self, tool_calls_list, tools_call):
        """合并工具调用列表

        Args:
            tool_calls_list: 已收集的工具调用列表
            tools_call: 新的工具调用
        """
        for tool_call in tools_call:
            tool_index = getattr(tool_call, "index", None)
            if tool_index is None:
                if tool_call.function.name:
                    # 有 function_name，说明是新的工具调用
                    tool_index = len(tool_calls_list)
                else:
                    tool_index = len(tool_calls_list) - 1 if tool_calls_list else 0

            # 确保列表有足够的位置
            if tool_index >= len(tool_calls_list):
                tool_calls_list.append({"id": "", "name": "", "arguments": ""})

            # 更新工具调用信息
            if tool_call.id:
                tool_calls_list[tool_index]["id"] = tool_call.id
            if tool_call.function.name:
                tool_calls_list[tool_index]["name"] = tool_call.function.name
            if tool_call.function.arguments:
                tool_calls_list[tool_index]["arguments"] += tool_call.function.arguments
