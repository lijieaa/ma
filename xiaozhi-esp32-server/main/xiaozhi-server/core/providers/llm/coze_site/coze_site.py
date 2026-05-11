"""
扣子编程（code.coze.cn）发布到 *.coze.site 的 stream_run SSE 接口。

与开放平台 CozeLLM（bot_id + cozepy）不同，本适配器使用：
  POST {stream_url}  +  Bearer Token  +  project_id  +  session_id
响应为 text/event-stream，行格式 data: {json}
"""

from __future__ import annotations

import copy
import json
import re
import uuid
from typing import Any, Generator, Iterator, List, Optional

import requests

from config.logger import setup_logging
from core.providers.llm.base import LLMProviderBase
from core.providers.llm.openai.openai import LLMProvider as OpenAICompatLLM
from core.providers.llm.system_prompt import get_system_prompt_for_function
from core.utils.util import check_model_key

TAG = __name__
logger = setup_logging()


def _iter_text_from_event(obj: Any) -> Iterator[str]:
    """从 stream_run 返回的 JSON 事件中尽量抽取可播报的文本增量。"""
    if obj is None:
        return
    if isinstance(obj, str):
        s = obj.strip()
        if s and s != "[DONE]":
            yield s
        return
    if not isinstance(obj, dict):
        return

    # 扣子 stream_run 常见信封：{ "type": "answer", "content": { "answer": "...", ... } }
    ctn = obj.get("content")
    if isinstance(ctn, dict):
        ans = ctn.get("answer")
        if isinstance(ans, str) and ans:
            yield ans
        err = ctn.get("error")
        if isinstance(err, str) and err.strip():
            yield f"【扣子返回错误】{err.strip()}"
        # 已按扣子结构消费 answer，避免再走下面 obj["answer"] / 递归把其它字段当正文
        if "answer" in ctn or obj.get("type") == "answer":
            return

    # OpenAI 风格 chunk
    choices = obj.get("choices")
    if isinstance(choices, list) and choices:
        c0 = choices[0] if choices else {}
        if isinstance(c0, dict):
            delta = c0.get("delta") or {}
            if isinstance(delta, dict):
                c = delta.get("content")
                if isinstance(c, str) and c:
                    yield c
            msg = c0.get("message")
            if isinstance(msg, dict):
                c = msg.get("content")
                if isinstance(c, str) and c:
                    yield c

    for key in ("text", "answer", "output", "content"):
        v = obj.get(key)
        if isinstance(v, str) and v.strip():
            yield v.strip()

    # 常见嵌套（event 常为 SSE 元数据字符串如 "message"，勿当正文输出）
    for nest_key in ("event", "data", "message", "delta", "result"):
        nested = obj.get(nest_key)
        if nest_key == "event" and isinstance(nested, str):
            continue
        if isinstance(nested, (dict, list)):
            yield from _iter_text_from_event(nested)

    content = obj.get("content")
    if isinstance(content, dict):
        q = content.get("query")
        if isinstance(q, dict):
            yield from _iter_text_from_event(q)

    for arr_key in ("messages", "parts", "chunks", "items"):
        arr = obj.get(arr_key)
        if isinstance(arr, list):
            for item in arr:
                yield from _iter_text_from_event(item)


def _is_sse_control_line(line: str) -> bool:
    """SSE 控制行，不应作为模型正文播报。"""
    ls = line.strip()
    if not ls or ls.startswith(":"):
        return True
    low = ls.lower()
    if low.startswith("event:"):
        return True
    if low.startswith("id:"):
        return True
    if low.startswith("retry:"):
        return True
    return False


def _iter_sse_data_lines(response: requests.Response) -> Iterator[str]:
    """
    从 HTTP 流中按行产出；兼容一次 chunk 内粘多行（无 \\n）的情况。
    标准 SSE 为 event:/data: 分行，扣子若粘连则按 event: / data: 切分。
    """
    buf = ""
    # 不使用 decode_unicode=True：避免按错误 charset（如 ascii）解码 SSE 正文触发编码异常
    for chunk in response.iter_content(chunk_size=2048, decode_unicode=False):
        if chunk is None:
            continue
        if isinstance(chunk, bytes):
            chunk = chunk.decode("utf-8", errors="replace")
        elif not isinstance(chunk, str):
            chunk = str(chunk)
        buf += chunk
        while True:
            nl = buf.find("\n")
            if nl < 0:
                break
            line = buf[:nl].rstrip("\r")
            buf = buf[nl + 1 :]
            if line.strip():
                yield line
    tail = buf.strip()
    if not tail:
        return
    # 无换行但含多段 event:/data:
    if "\n" not in tail and tail.count("event:") + tail.count("data:") > 1:
        for part in re.split(r"(?=\b(?:event|data):)", tail):
            p = part.strip()
            if p:
                yield p
    else:
        yield tail


def _compose_user_query(dialogue: List[dict]) -> str:
    """把对话转成一段发给扣子的 query 文本（含系统提示 + 近期上下文）。"""
    if not dialogue:
        return ""

    system_text = ""
    for m in dialogue:
        if m.get("role") == "system" and m.get("content"):
            system_text = (m.get("content") or "").strip()
            break

    # 取末尾若干条非 system 消息，避免 payload 过大
    tail: List[dict] = []
    for m in dialogue:
        if m.get("role") == "system":
            continue
        tail.append(m)
    tail = tail[-12:]

    parts: List[str] = []
    for m in tail:
        role, content = m.get("role"), (m.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            parts.append(f"用户：{content}")
        elif role == "assistant":
            parts.append(f"助手：{content}")
        elif role == "tool":
            parts.append(f"工具结果：{content}")

    body = "\n".join(parts)
    if system_text:
        return f"{system_text}\n\n{body}" if body else system_text
    return body


def _dialogue_for_function_call(
    dialogue: List[dict], functions: Optional[List[dict]]
) -> List[dict]:
    """
    与 OpenAI（AliLLM）/开放平台 CozeLLM 一致：规范化消息，并把可用工具说明挂到首轮用户句上；
    将末尾 tool 结果合并回 user，便于 stream_run 单段 query 携带上下文。
    """
    d: List[dict] = copy.deepcopy(dialogue) if dialogue else []
    OpenAICompatLLM.normalize_dialogue(d)

    if functions is not None and len(functions) > 0:
        # 对齐 core/providers/llm/coze/coze.py：仅 system+user 时注入 TOOL USE 长提示，避免每轮重复膨胀
        if len(d) == 2:
            last_msg = d[-1].get("content", "") or ""
            function_str = json.dumps(functions, ensure_ascii=False)
            d[-1]["content"] = get_system_prompt_for_function(function_str) + last_msg

        if len(d) > 1 and d[-1].get("role") == "tool":
            assistant_msg = (
                "\ntool call result: " + (d[-1].get("content", "") or "") + "\n\n"
            )
            while len(d) > 1:
                if d[-1].get("role") == "user":
                    d[-1]["content"] = assistant_msg + (d[-1].get("content", "") or "")
                    break
                d.pop()

    return d


class LLMProvider(LLMProviderBase):
    def __init__(self, config: dict):
        # 完整 URL，例如 https://5csccg2gcw.coze.site/stream_run
        self.stream_url = (config.get("stream_url") or "").strip()
        self.project_id = str(config.get("project_id", "")).strip()
        self.api_token = (
            config.get("api_token") or config.get("personal_access_token") or ""
        ).strip()
        self.timeout = int(config.get("timeout", 120))
        self._coze_session_by_xz: dict[str, str] = {}

        err = check_model_key("CozeSiteLLM", self.api_token)
        if err:
            logger.bind(tag=TAG).error(err)
        if not self.stream_url or not self.project_id:
            logger.bind(tag=TAG).error(
                "CozeSiteLLM 需在配置中填写 stream_url 与 project_id"
            )

    def _coze_session_id(self, xiaozhi_session_id: str) -> str:
        sid = xiaozhi_session_id or uuid.uuid4().hex
        if sid not in self._coze_session_by_xz:
            # 与扣子示例类似：短随机串；也可改为 persist
            self._coze_session_by_xz[sid] = uuid.uuid4().hex[:22]
        return self._coze_session_by_xz[sid]

    def response(self, session_id, dialogue, **kwargs) -> Generator[str, None, None]:
        query_text = _compose_user_query(dialogue)
        if not query_text.strip():
            yield ""
            return

        coze_sid = self._coze_session_id(session_id or "")
        payload = {
            "content": {
                "query": {
                    "prompt": [
                        {
                            "type": "text",
                            "content": {"text": query_text},
                        }
                    ]
                }
            },
            "type": "query",
            "session_id": coze_sid,
            "project_id": self.project_id,
        }
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        logger.bind(tag=TAG).debug(
            f"CozeSite stream_run POST {self.stream_url} project_id={self.project_id}"
        )

        try:
            with requests.post(
                self.stream_url,
                headers=headers,
                json=payload,
                stream=True,
                timeout=self.timeout,
            ) as resp:
                resp.raise_for_status()
                json_buffer = ""
                # 扣子 type=answer 时，content.answer 可能是「累计全文」；只播相对上次的后缀
                coze_last_answer = ""
                for line in _iter_sse_data_lines(resp):
                    if _is_sse_control_line(line):
                        continue

                    ls = line.strip()
                    if not ls:
                        continue

                    # 只处理 data: 行；忽略误当成正文的 event: 等
                    if ls.lower().startswith("data:"):
                        data_text = ls[5:].lstrip()
                    elif ls.startswith("{"):
                        data_text = ls
                    else:
                        continue

                    if not data_text or data_text == "[DONE]":
                        continue

                    # 多行拼一个 JSON
                    if data_text.startswith("{") and not data_text.endswith("}"):
                        json_buffer += data_text
                        continue
                    if json_buffer:
                        data_text = json_buffer + data_text
                        json_buffer = ""

                    try:
                        parsed = json.loads(data_text)
                    except json.JSONDecodeError:
                        # 仅当不像 SSE 元数据时才透出（避免把 event: 当正文）
                        if data_text and not re.match(
                            r"^\s*event\s*:", data_text, re.I
                        ):
                            yield data_text
                        continue

                    emitted = False
                    for chunk in _iter_text_from_event(parsed):
                        if not chunk:
                            continue
                        emitted = True
                        if parsed.get("type") == "answer":
                            if coze_last_answer and chunk.startswith(coze_last_answer):
                                delta = chunk[len(coze_last_answer) :]
                                coze_last_answer = chunk
                                if delta:
                                    yield delta
                            else:
                                coze_last_answer = chunk
                                yield chunk
                        else:
                            yield chunk
                    if isinstance(parsed, dict) and parsed.get("finish") is True:
                        coze_last_answer = ""
                    if not emitted and isinstance(parsed, dict):
                        logger.bind(tag=TAG).debug(
                            f"CozeSite 未识别字段的 SSE 对象 keys={list(parsed.keys())[:20]}"
                        )
        except requests.RequestException as e:
            logger.bind(tag=TAG).error(f"CozeSite 请求失败: {e}")
            yield f"扣子站点接口请求失败：{e}"

    def response_with_functions(self, session_id, dialogue, functions=None, **kwargs):
        """
        对齐 AliLLM（OpenAI 兼容）与开放平台 CozeLLM 的 function 路径：
        - 使用与 openai.LLMProvider 相同的 normalize_dialogue；
        - 首轮（system+user）把工具列表经 get_system_prompt_for_function 拼进用户句，
          引导模型用 <tool_call> JSON（connection 层可解析）；
        - tool 角色回合合并进 user，与 coze.py 行为一致。
        stream_run 响应本身仍无 OpenAI delta.tool_calls，故第二元组恒为 None。
        """
        d = _dialogue_for_function_call(dialogue, functions)
        for token in self.response(session_id, d, **kwargs):
            yield token, None
