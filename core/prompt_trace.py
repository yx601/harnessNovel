"""记录模型调用的实际 Prompt 与响应，供 Web 工作台实时展示。"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Callable


_TRACE_CALLBACK: ContextVar[Callable[[dict], None] | None] = ContextVar(
    "harness_novel_prompt_trace_callback", default=None,
)
_FILE_LOCK = threading.Lock()


def record_prompt(prompt: str, model: str = "", label: str = "") -> dict:
    """记录一次模型调用的请求阶段。

    返回带 id 的事件 dict，后续可用同一 id 调用 record_response 补充结果。
    """
    event = {
        "id": uuid.uuid4().hex,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "timestamp_ms": int(time.time() * 1000),
        "model": str(model or ""),
        "label": str(label or "模型调用"),
        "prompt": str(prompt or ""),
        "prompt_chars": len(str(prompt or "")),
        "status": "pending",
        "response": "",
        "response_chars": 0,
        "duration_sec": 0.0,
        "error": "",
    }
    trace_file = os.getenv("HARNESS_NOVEL_PROMPT_TRACE_FILE", "").strip()
    if trace_file:
        try:
            path = Path(trace_file)
            with _FILE_LOCK:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        except OSError:
            pass
    callback = _TRACE_CALLBACK.get()
    if callback:
        try:
            callback(dict(event))
        except Exception:
            pass
    return event


def record_response(
    event_id: str,
    response: str = "",
    elapsed_sec: float = 0.0,
    error: str = "",
    model: str = "",
    prompt: str = "",
) -> None:
    """补充一次模型调用的结果阶段。

    通过 callback 推送一个 response 事件，前端据此更新日志条目状态。
    """
    event = {
        "id": event_id,
        "type": "response",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "timestamp_ms": int(time.time() * 1000),
        "model": str(model or ""),
        "status": "error" if error else "completed",
        "response": str(response or ""),
        "response_chars": len(str(response or "")),
        "duration_sec": round(elapsed_sec, 2),
        "error": str(error or ""),
        "prompt_chars": len(str(prompt or "")),
    }
    trace_file = os.getenv("HARNESS_NOVEL_PROMPT_TRACE_FILE", "").strip()
    if trace_file:
        try:
            path = Path(trace_file)
            with _FILE_LOCK:
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        except OSError:
            pass
    callback = _TRACE_CALLBACK.get()
    if callback:
        try:
            callback(dict(event))
        except Exception:
            pass


@contextmanager
def capture_prompts(callback: Callable[[dict], None]):
    """在当前后台任务线程中捕获模型 Prompt，不影响其他并发任务。"""
    token = _TRACE_CALLBACK.set(callback)
    try:
        yield
    finally:
        _TRACE_CALLBACK.reset(token)
