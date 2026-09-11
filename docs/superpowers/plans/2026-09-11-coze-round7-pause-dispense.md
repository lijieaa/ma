# 第7轮「停顿3秒 + 舵机出药」Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在扣子固定话术第7轮中，实现「前半句播完 → 静默3秒 → 舵机出药 → 后半句继续播」，且 TTS 不念出控制标记。

**Architecture:** 扣子系统提示词输出带控制标记的固定文案；服务端在 `coze_workflow_run` 模式下先缓存完整回复、不立刻流式 TTS；解析 `<<<PAUSE:N>>>` / `<<<DISPENSE>>>` 后按段播报，并在停顿后调用已有设备 MCP `self.serial_servo.dispense_medicine`。

**Tech Stack:** xiaozhi-esp32-server（Python/asyncio）、扣子工作流系统提示词、设备 MCP（ESP32 `mcp_server.cc`）

**Spec:** 用户需求文案 + `e:\VisizenProject\ma\需求\系统提示词.txt`（需同步改第7轮）

## Global Constraints

- 扣子固定回复必须「一字不差」含控制标记；标记只给服务端解析，禁止 TTS 朗读。
- 不改固件出药物理行为：继续用现有 `dispense_medicine`（电机模式约转5秒）。
- 仅在 `coze_workflow_run` 主 LLM 路径启用编排；其它 LLM 保持原行为。
- 对话历史入库的 assistant 文本应去掉控制标记，保留用户可听语义。

---

## 0. 背景与差距

### 0.1 目标听感（用户期望）

1. 播：轻微擦伤说明 + 「已发送到智能医务室管理系统由医生确认，请稍等。」
2. **静默约 3 秒**（模拟医生确认）
3. 舵机旋转发一粒药（碘伏和敷料贴）
4. 播：「好的，医生已经确认，现在给你发放碘伏和敷料贴……」后续护理指导

### 0.2 当前提示词（文件现状）

`系统提示词.txt` 第7轮目前是：

> 这是轻微表皮擦伤，符合自助外用药品领取条件。给你发放碘伏和敷料贴，请在窗口取用……

**没有**「请稍等 / 医生确认 / 停顿3秒」段落。

服务端已有关键字出药：`_coze_workflow_reply_needs_dispense_medicine`（含「碘伏和敷料贴|自助外用药品领取」且含「发放」）会立刻调 MCP，但：

- 整段话一口气 TTS，**无停顿**
- 出药时机在流式文本入队之后立刻触发，**不对齐「请稍等」之后**

### 0.3 推荐总体时序

```text
用户第7轮话术
    │
    ▼
扣子返回带标记的完整固定文案（一次 /run）
    │
    ▼
服务端识别为 coze_workflow_run
    │  （本轮：不边收边 TTS，先拼完整字符串）
    ▼
解析分段：
  segment[0] = 前半段（到「请稍等。」）
  action     = PAUSE:3 + DISPENSE
  segment[1] = 后半段（「好的，医生已经确认…」）
    │
    ▼
TTS 播 segment[0] → 等播完
    │
    ▼
asyncio.sleep(3)
    │
    ▼
MCP: dispense_medicine（不二次播工具自带 say，避免抢麦）
    │
    ▼
TTS 播 segment[1] → 结束
```

---

## 1. 文件与职责

| 文件 | 改动 |
|------|------|
| `e:\VisizenProject\ma\需求\系统提示词.txt` | 改第7轮固定文案，加入标记 |
| 扣子平台「系统提示词」配置（线上） | 与本地文件同步 |
| `xiaozhi-esp32-server/main/xiaozhi-server/core/connection.py` | 解析标记、分段 TTS、停顿、触发出药；抑制本路径工具结果二次播报 |
| （可选）`core/utils/scripted_reply.py` 新建 | 把解析/编排从 `connection.py` 拆出，便于单测 |
| `AiXiaoZhi_Hiwonder_Sensor/main/mcp_server.cc` | **原则上不改**（出药已具备） |
| （可选单测）`tests/test_scripted_reply_markers.py` | 解析器单测 |

---

## 2. 控制标记协议（约定）

### 2.1 标记语法

| 标记 | 含义 | TTS |
|------|------|-----|
| `<<<PAUSE:3>>>` | 前一段播完后静默 3 秒（秒数为整数） | 不朗读 |
| `<<<DISPENSE>>>` | 在停顿之后、下一段开播前调用出药 MCP | 不朗读 |

规则：

- 标记独占一行，或紧贴在两段正文之间（解析时用正则全局匹配）。
- 同一回复可有多个 `PAUSE` / 多段文本；第7轮只需各出现一次。
- `DISPENSE` 可在 `PAUSE` 前或后；**执行顺序固定为：播完上一段 → 所有 PAUSE 累加等待 → DISPENSE → 播下一段**。
- 若只有关键字出药、无 `<<<DISPENSE>>>`：兼容旧逻辑（回复结束后触发出药），但第7轮应以显式标记为准。

### 2.2 第7轮目标固定文案（写入提示词）

```text
这是轻微表皮擦伤，符合自助外用药品领取条件。已将你的情况发送到智能医务室管理系统由医生确认，请稍等。
<<<PAUSE:3>>>
<<<DISPENSE>>>
好的，医生已经确认，现在给你发放碘伏和敷料贴，请在窗口取用碘伏和敷料贴，用碘伏由伤口中心向外画圈消毒两遍，再用敷料贴贴上。如果伤口继续疼痛红肿，请去学校医务室找医生处置，不可拖延。
```

说明：

- 去掉括号说明「（此处停留3秒）」「（舵机旋转360度…）」——这些若进模型输出会被念出来。
- 「智能医务室管理系统」外层引号可去掉，减少 TTS 念「引号」的概率。
- 对话历史存储用「去标记后的完整可读文本」。

---

## 3. 服务端详细设计

### 3.1 解析器

新增函数（建议独立模块）：

```python
# 伪代码
MARKER_RE = re.compile(
    r"<<<\s*(PAUSE\s*:\s*(\d+)|DISPENSE)\s*>>>"
)

def parse_scripted_reply(text: str) -> list[dict]:
    """
    返回有序 steps，例如：
      {"type":"speak","text":"..."}
      {"type":"pause","seconds":3}
      {"type":"dispense"}
      {"type":"speak","text":"..."}
    """
```

边界：

- 无任何标记 → 返回单步 `speak`（走原流式/原逻辑）。
- 标记夹在句子中间 → 按标记切开，空白段丢弃。
- `PAUSE:0` 或非法数字 → 忽略该 pause 或按 0 处理并打 warning。

### 3.2 何时启用「缓存后编排」

在 `ConnectionHandler` 的 LLM 流式处理中（现有 `coze_workflow_only` 分支）：

**若**主 LLM 为 `coze_workflow_run`：

1. 流式 token **只拼接到** `response_message`，**不要**立刻 `tts_text_queue.put(MIDDLE)`。
2. 流结束后得到 `full = "".join(response_message)`。
3. 若 `parse_scripted_reply(full)` 步数 > 1 或含 pause/dispense：
   - 走 `_play_scripted_coze_reply(steps)` 编排；
   - **跳过**原「边流边 TTS + 末尾 side_effect 立即出药」路径，避免重复出药/重复播报。
4. 若无标记：可回退为一次 `tts_one_sentence`/`put MIDDLE`，并保留现有 `_coze_workflow_reply_side_effect_tool_calls`（兼容第3轮测温等）。

> 注意：当前第3轮测温也是 side_effect。测温话术无 pause 标记时仍走旧 side_effect；第7轮有标记时走新编排。

### 3.3 播完等待

服务端「发送完音频」≠「设备播完」。可用组合策略：

1. 对当前段：`FIRST` + 文本 + `LAST`，走现有 TTS 管线。
2. 等待条件（建议）：
   - `tts_text_queue` 与 `tts_audio_queue` 近似空；
   - 复用/调用 `sendAudioHandle._wait_for_audio_completion`；
   - 再轮询 `conn.client_is_speaking is False`（`LAST` 时会置 False），超时例如 60s。
3. 再 `await asyncio.sleep(pause_seconds)`。

实现位置：编排函数应在 **asyncio loop** 上跑（`run_coroutine_threadsafe`），不要在阻塞线程里瞎 sleep 导致和 TTS 线程死锁；或在 chat 线程里用 `asyncio.run_coroutine_threadsafe(...).result()` 等待完成。

### 3.4 出药调用

复用现有：

- `_resolve_dispense_medicine_tool_name`
- `_dispense_medicine_tool_arguments`（固件已忽略外部 speed，可保留兼容）
- `_run_mcp_tool_calls` / `func_handler.handle_llm_function_call`

编排中调用时增加标志，例如 `conn.suppress_tool_say = True`，避免工具 JSON 里的 `say` 再插一句「正在旋转发放…」打断剧本。

出药与后半段 TTS 可并行启动（舵机转 5 秒、语音同时播发放说明），体验更好；**最小方案**可先串行：出药 MCP 返回后再播后半段（MCP 几乎立即返回，电机在固件线程里转）。

### 3.5 对话历史

```python
clean_text = strip_markers(full)  # 去掉 <<<...>>>
self.dialogue.put(Message(role="assistant", content=clean_text))
```

### 3.6 与旧出药关键字的关系

| 场景 | 行为 |
|------|------|
| 含 `<<<DISPENSE>>>` | 仅编排触发出药；**不要**再跑 `_coze_workflow_reply_needs_dispense_medicine` |
| 无标记但含「碘伏…发放」 | 保持旧 side_effect（兼容未改提示词的环境） |
| 用户直连口令「舵机旋转360度…」 | 现有 `_build_direct_mcp_voice_command_calls` 不变 |

---

## 4. 扣子 / 提示词改动步骤

1. 更新本地 `e:\VisizenProject\ma\需求\系统提示词.txt` 第7轮为第 2.2 节文案。
2. 在语言规则中增加一条（建议写入提示词「约束规则」）：

   > 固定回复中的 `<<<PAUSE:N>>>`、`<<<DISPENSE>>>` 为系统控制标记，必须原样输出，不得改写、翻译或删掉。

3. 同步到扣子 Bot / 工作流「系统提示」配置并发布。
4. 用扣子试运行第7轮，确认输出含两行标记且正文与要求一致。

---

## 5. 任务拆分（实现顺序）

### Task 1：提示词与协议定稿

- [ ] 修改 `系统提示词.txt` 第7轮为带标记文案
- [ ] 增加「控制标记必须原样输出」约束
- [ ] 同步扣子线上配置

**验证：** 扣子调试面板粘贴第7轮用户句，输出含 `<<<PAUSE:3>>>` 与 `<<<DISPENSE>>>`。

### Task 2：解析器 + 单测

- [ ] 新增 `parse_scripted_reply` / `strip_markers`
- [ ] 单测：无标记、仅 pause、pause+dispense、多余空白、多段 speak

**验证：** 单元测试通过。

### Task 3：编排播放（connection）

- [ ] `coze_workflow_only` 改为先缓存全文
- [ ] 有标记则调用 `_play_scripted_coze_reply`
- [ ] 实现 wait TTS + sleep + dispense + 第二段 TTS
- [ ] 抑制工具 say；对话历史去标记
- [ ] 有 `DISPENSE` 时跳过旧关键字出药，避免双触发

**验证：** 日志顺序为：`speak1` → `pause 3s` → `dispense_medicine` → `speak2`。

### Task 4：联调（设备）

- [ ] 本地/PyCharm 或 Docker 起 server，设备连 WS
- [ ] 走完 1–7 轮，听感确认停顿约 3 秒
- [ ] 确认舵机只转一次、窗口有药
- [ ] 确认 TTS 未念出 `PAUSE`/`DISPENSE` 字样
- [ ] 第3轮测温 side_effect 仍正常

### Task 5：回归

- [ ] 第8轮感谢话术正常
- [ ] 自由发挥（非固定轮）无标记时行为与改前一致
- [ ] 用户说「直接发放药品」直连 MCP 仍可用

---

## 6. 关键代码落点（伪代码）

```python
# connection.py 流结束处（coze_workflow_only）
full = "".join(response_message)
steps = parse_scripted_reply(full)
if any(s["type"] in ("pause", "dispense") for s in steps):
    clean = strip_markers(full)
    self.dialogue.put(Message(role="assistant", content=clean))
    fut = asyncio.run_coroutine_threadsafe(
        self._play_scripted_coze_reply(steps), self.loop
    )
    fut.result(timeout=120)
    return True
# else: 原 TTS + side_effect
```

```python
async def _play_scripted_coze_reply(self, steps):
    for step in steps:
        if step["type"] == "speak":
            self.sentence_id = uuid.uuid4().hex
            # 入队 FIRST/TEXT/LAST，等待播完
            await self._speak_and_wait(step["text"])
        elif step["type"] == "pause":
            await asyncio.sleep(step["seconds"])
        elif step["type"] == "dispense":
            self.suppress_tool_say = True
            try:
                self._run_mcp_tool_calls([{...dispense...}])
            finally:
                self.suppress_tool_say = False
```

---

## 7. 风险与缓解

| 风险 | 缓解 |
|------|------|
| 停顿感觉不到 3 秒（等播完不准） | 组合 queue empty + `client_is_speaking` + 小余量；现场用秒表调 |
| 扣子偶发丢掉标记 | 提示词强调；服务端可对「请稍等」硬切兜底（可选） |
| 出药两次 | 有标记路径禁用关键字 side_effect |
| 工具 say 打断剧本 | `suppress_tool_say` |
| 流式改缓存增加首包延迟 | 第7轮固定短文，可接受；仅 coze 路径 |

### 可选兜底（标记丢失时）

若全文含「请稍等」且含「医生已经确认」，无标记也自动在「请稍等。」后插入 pause=3 与 dispense。作为第二道保险，避免演示翻车。

---

## 8. 验收标准

1. 第7轮听感：前半段结束 → 约 3 秒静音 → 舵机动作与后半段发放说明衔接自然。
2. TTS 音频中无「PAUSE」「DISPENSE」「小于大于」等杂音。
3. 舵机仅触发 1 次。
4. 第3轮测温、直连出药口令、非剧本闲聊均不回归。
5. 对话日志/历史中 assistant 内容为去标记后的完整中文。

---

## 9. 不在本期范围

- 真连「智能医务室管理系统」医生审核 API（本期用 3 秒静默模拟）。
- 改舵机为精确 360° 位置模式（继续电机时长近似）。
- 把 8 轮全部改成服务端状态机（仍由扣子固定文案驱动）。
