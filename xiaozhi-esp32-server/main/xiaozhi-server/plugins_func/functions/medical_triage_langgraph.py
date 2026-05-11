"""
校园医疗分诊 — LangGraph 编排层

图结构（与产品文档对应）：
  START → safety_redline（自杀/自伤/无望感，立即结束）
        → fsm_execute（外伤 / 内科 / 心理 / 总开场 的确定性状态机）

说明：
- 具体问诊话术与状态迁移在 campus_medical_triage.execute_medical_turn 中维护；
- 本文件负责「红线优先」的可视化分支与 thread 级 checkpoint（便于后续接多节点、人工审核等）。
"""

from __future__ import annotations

from typing import TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from plugins_func.register import Action, ActionResponse


class MedicalTriageGraphState(TypedDict, total=False):
    user_text: str
    temperature_c: float | None
    response: str
    result: str
    halted: bool


def _conn(config: RunnableConfig):
    c = config.get("configurable", {}).get("conn")
    if c is None:
        raise ValueError("medical triage graph 缺少 configurable.conn")
    return c


def node_safety_redline(
    state: MedicalTriageGraphState, config: RunnableConfig
) -> MedicalTriageGraphState:
    from plugins_func.functions.campus_medical_triage import (
        _is_crisis_text,
        _emergency_response,
        _normalize,
    )

    conn = _conn(config)
    text = _normalize(state.get("user_text") or "")
    if _is_crisis_text(text):
        r = _emergency_response(conn)
        return {
            **state,
            "response": r.response or "",
            "result": r.result or "",
            "halted": True,
        }
    return {**state, "halted": False}


def node_fsm_execute(
    state: MedicalTriageGraphState, config: RunnableConfig
) -> MedicalTriageGraphState:
    from plugins_func.functions.campus_medical_triage import execute_medical_turn

    conn = _conn(config)
    r = execute_medical_turn(
        conn,
        state.get("user_text") or "",
        state.get("temperature_c"),
    )
    return {
        **state,
        "response": r.response or "",
        "result": getattr(r, "result", None) or "",
        "halted": state.get("halted", False),
    }


def _route_after_safety(state: MedicalTriageGraphState) -> str:
    return "end" if state.get("halted") else "fsm"


def build_medical_triage_graph():
    g = StateGraph(MedicalTriageGraphState)
    g.add_node("safety_redline", node_safety_redline)
    g.add_node("fsm_execute", node_fsm_execute)
    g.add_conditional_edges(
        "safety_redline",
        _route_after_safety,
        {"end": END, "fsm": "fsm_execute"},
    )
    g.add_edge("fsm_execute", END)
    g.add_edge(START, "safety_redline")
    return g.compile(checkpointer=MemorySaver())


_compiled = None


def run_triage_graph(
    conn,
    user_text: str,
    temperature_c: float | None = None,
) -> ActionResponse:
    """对单次用户输入运行分诊图，返回 ActionResponse。"""
    global _compiled
    if _compiled is None:
        _compiled = build_medical_triage_graph()

    thread_id = f"medtriage:{getattr(conn, 'device_id', 'unknown')}"
    out = _compiled.invoke(
        {"user_text": user_text, "temperature_c": temperature_c},
        config={
            "configurable": {
                "thread_id": thread_id,
                "conn": conn,
            }
        },
    )
    return ActionResponse(
        action=Action.RESPONSE,
        response=out.get("response") or "",
        result=out.get("result") or None,
    )
