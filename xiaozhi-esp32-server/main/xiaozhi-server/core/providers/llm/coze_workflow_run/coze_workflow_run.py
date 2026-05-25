"""
扣子工作流 HTTP 对接（*.coze.site/run）

与 CozeSiteLLM（stream_run SSE）不同，本适配器使用同步 JSON 接口：
  POST {run_url}
  Body: {"user_message": "...", "chat_history": "..."}
  Response: {"ai_reply": "...", "chat_history": "[...]", "run_id": "..."}
"""

from __future__ import annotations

import copy
import json
from typing import Generator, List, Optional

import requests

from config.logger import setup_logging
from core.providers.llm.base import LLMProviderBase
from core.providers.llm.openai.openai import LLMProvider as OpenAICompatLLM
from core.providers.llm.system_prompt import get_system_prompt_for_function
from core.utils.util import check_model_key

TAG = __name__
logger = setup_logging()


def _normalize_token(raw: str) -> str:
    token = (raw or "").strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    return token


def _is_injected_tool_or_system_user_text(text: str) -> bool:
    """过滤 connection 注入的 MCP 工具提醒、系统提示等，避免当作 user_message。"""
    t = (text or "").strip()
    if not t:
        return True
    if t.startswith("<tool_calling>") or t.startswith("[系统提示]") or t.startswith(
        "[重要提醒]"
    ):
        return True
    if "【核心原则】你是拥有工具能力的智能助手" in t:
        return True
    if "当前可用工具:" in t and "调用" in t:
        return True
    return False


def _extract_user_message(dialogue: List[dict], plain_query: str = "") -> str:
    """当前轮用户输入，对应接口字段 user_message。"""
    d: List[dict] = copy.deepcopy(dialogue) if dialogue else []
    OpenAICompatLLM.normalize_dialogue(d)

    if len(d) > 1 and d[-1].get("role") == "tool":
        assistant_msg = "\ntool call result: " + (d[-1].get("content") or "") + "\n\n"
        while len(d) > 1:
            if d[-1].get("role") == "user":
                d[-1]["content"] = assistant_msg + (d[-1].get("content") or "")
                break
            d.pop()

    for m in reversed(d):
        if m.get("role") == "user":
            content = (m.get("content") or "").strip()
            if content and not _is_injected_tool_or_system_user_text(content):
                return content

    return (plain_query or "").strip()


def _dialogue_for_function_call(
    dialogue: List[dict], functions: Optional[List[dict]]
) -> List[dict]:
    d: List[dict] = copy.deepcopy(dialogue) if dialogue else []
    OpenAICompatLLM.normalize_dialogue(d)

    if functions is not None and len(functions) > 0 and len(d) == 2:
        last_msg = d[-1].get("content", "") or ""
        function_str = json.dumps(functions, ensure_ascii=False)
        d[-1]["content"] = get_system_prompt_for_function(function_str) + last_msg

    if len(d) > 1 and d[-1].get("role") == "tool":
        assistant_msg = "\ntool call result: " + (d[-1].get("content") or "") + "\n\n"
        while len(d) > 1:
            if d[-1].get("role") == "user":
                d[-1]["content"] = assistant_msg + (d[-1].get("content") or "")
                break
            d.pop()
    return d


def _json_for_log(obj, max_len: int = 8000) -> str:
    try:
        text = json.dumps(obj, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        text = str(obj)
    if len(text) > max_len:
        return text[:max_len] + f"\n... (truncated, total {len(text)} chars)"
    return text


def _yield_text_chunks(text: str, chunk_size: int = 8) -> Generator[str, None, None]:
    if not text:
        yield ""
        return
    for i in range(0, len(text), chunk_size):
        yield text[i : i + chunk_size]


class LLMProvider(LLMProviderBase):
    def __init__(self, config: dict):
        self.run_url = (config.get("run_url") or "").strip()
        self.api_token = _normalize_token(
            config.get("api_token")
            or config.get("personal_access_token")
            or ""
        )
        self.timeout = int(config.get("timeout", 120))
        self.stream_chunk_size = int(config.get("stream_chunk_size", 8))
        # session_id -> chat_history 字符串（扣子返回的原样透传）
        self._chat_history_by_session: dict[str, str] = {}

        err = check_model_key("CozeWorkflowRunLLM", self.api_token)
        if err:
            logger.bind(tag=TAG).error(err)
        if not self.run_url:
            logger.bind(tag=TAG).error(
                "CozeWorkflowRunLLM 需在配置中填写 run_url，例如 https://xxx.coze.site/run"
            )

    def _post_run(
        self, user_message: str, chat_history: str, session_id: str = ""
    ) -> dict:
        payload = {
            "user_message": user_message,
            "chat_history": chat_history or "",
        }
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }
        logger.bind(tag=TAG).info(
            f"CozeWorkflowRun 请求 >>>\n"
            f"session_id: {session_id or '(empty)'}\n"
            f"url: {self.run_url}\n"
            f"body:\n{_json_for_log(payload)}"
        )
        resp = requests.post(
            self.run_url,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            raise ValueError(f"invalid response type: {type(data)}")
        logger.bind(tag=TAG).info(
            f"CozeWorkflowRun 响应 <<<\n"
            f"session_id: {session_id or '(empty)'}\n"
            f"status: {resp.status_code}\n"
            f"body:\n{_json_for_log(data)}"
        )
        return data

    def response(self, session_id, dialogue, **kwargs) -> Generator[str, None, None]:
        plain_query = (kwargs.get("plain_query") or "").strip()
        user_message = _extract_user_message(dialogue, plain_query)
        if plain_query:
            user_message = plain_query
        if not user_message.strip():
            yield ""
            return

        sid = session_id or ""
        chat_history = self._chat_history_by_session.get(sid, "")

        try:
            data = self._post_run(user_message, chat_history, sid)
        except requests.RequestException as e:
            logger.bind(tag=TAG).error(f"CozeWorkflowRun 请求失败: {e}")
            yield f"扣子工作流请求失败：{e}"
            return
        except (json.JSONDecodeError, ValueError) as e:
            logger.bind(tag=TAG).error(f"CozeWorkflowRun 响应解析失败: {e}")
            yield f"扣子工作流响应异常：{e}"
            return

        ai_reply = data.get("ai_reply")
        if not isinstance(ai_reply, str):
            ai_reply = str(ai_reply) if ai_reply is not None else ""

        new_history = data.get("chat_history")
        if isinstance(new_history, str) and sid:
            self._chat_history_by_session[sid] = new_history

        run_id = data.get("run_id", "")
        logger.bind(tag=TAG).info(
            f"CozeWorkflowRun ok run_id={run_id} reply_len={len(ai_reply)}"
        )

        yield from _yield_text_chunks(ai_reply, self.stream_chunk_size)

    def response_with_functions(self, session_id, dialogue, functions=None, **kwargs):
        # 扣子 /run 自行维护 chat_history，勿把 MCP 工具长提示拼进 user_message
        for token in self.response(session_id, dialogue, **kwargs):
            yield token, None
