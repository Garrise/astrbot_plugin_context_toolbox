"""AstrBot Context Toolbox

监控当前 AstrBot 实例中的每一次 LLM 请求（Provider.text_chat / text_chat_stream），
记录请求上下文结构（System Prompt、对话 contexts、工具定义、多模态内容等）与
响应内容（文本、思考、工具调用、Token 用量），并通过 WebUI 插件页面
（pages/llm-monitor）提供浏览、搜索、实时跟踪（SSE）与导出能力。

实现方式：
- 在插件 initialize() 时，对 Provider 基类的所有子类中定义的
  text_chat / text_chat_stream 方法进行包装（monkey patch），
  记录每次调用的入参与返回；terminate() 时恢复原始方法。
- 记录保存在内存环形缓冲区中（可配置容量），不写磁盘。
- 通过 context.register_web_api() 暴露 JSON API，供插件页面 bridge 调用。
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import json
import logging
import time
import uuid
from collections import deque
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any

from astrbot.api.star import Context, Star
from astrbot.api.web import (
    error_response,
    file_response,
    json_response,
    request,
    stream_response,
)
from astrbot.core.agent.message import Message
from astrbot.core.agent.tool import ToolSet
from astrbot.core.provider.entities import LLMResponse
from astrbot.core.provider.provider import Provider
from astrbot.core.utils.astrbot_path import get_astrbot_plugin_data_path

logger = logging.getLogger("astrbot")

PLUGIN_NAME = "astrbot_plugin_context_toolbox"

# 标记已包装的方法，避免插件重载时重复包装
_WRAPPED_ATTR = "_ctx_toolbox_wrapped"

# 重入保护：当一次请求已被记录时，其内部嵌套的 provider 调用不再重复记录
_depth: ContextVar[int] = ContextVar("ctx_toolbox_rec_depth", default=0)

# 需要记录的请求字段（对应 Provider.text_chat / text_chat_stream 的参数名）
_RECORDED_FIELDS = (
    "prompt",
    "session_id",
    "image_urls",
    "audio_urls",
    "func_tool",
    "contexts",
    "system_prompt",
    "tool_calls_result",
    "model",
    "extra_user_content_parts",
    "tool_choice",
    "request_max_retries",
)

# 搜索索引 blob 每条记录的最大长度，避免内存膨胀
_SEARCH_BLOB_MAX = 100_000


def _truncate(value: str, maxlen: int) -> str:
    if len(value) <= maxlen:
        return value
    return value[:maxlen] + f"...[truncated {len(value) - maxlen} chars]"


def _serialize(value: Any, maxlen: int) -> Any:
    """把任意请求/响应对象转换为可 JSON 序列化的结构，超长文本会被截断。"""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _truncate(value, maxlen)
    if isinstance(value, dict):
        return {str(k): _serialize(v, maxlen) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(v, maxlen) for v in value]
    if isinstance(value, set):
        return [_serialize(v, maxlen) for v in sorted(value, key=str)]
    if isinstance(value, ToolSet):
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": _serialize(t.parameters, maxlen),
            }
            for t in value.tools
        ]
    if isinstance(value, Message):
        try:
            return _serialize(value.model_dump(mode="json"), maxlen)
        except Exception:  # noqa: BLE001
            return {
                "role": getattr(value, "role", None),
                "content": _serialize(getattr(value, "content", None), maxlen),
                "tool_calls": _serialize(getattr(value, "tool_calls", None), maxlen),
                "tool_call_id": getattr(value, "tool_call_id", None),
            }
    # 其它 pydantic BaseModel
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return _serialize(dump(mode="json"), maxlen)
        except Exception:  # noqa: BLE001
            pass
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        try:
            return _serialize(dataclasses.asdict(value), maxlen)
        except Exception:  # noqa: BLE001
            pass
    return _truncate(repr(value), maxlen)


def _provider_info(provider: Any) -> dict:
    cfg = getattr(provider, "provider_config", None) or {}
    return {
        "id": cfg.get("id") or cfg.get("name") or "unknown",
        "type": cfg.get("type", "unknown"),
        "class": type(provider).__name__,
        "current_model": getattr(provider, "model_name", None),
    }


def _usage_dict(usage: Any) -> dict | None:
    if usage is None:
        return None
    return {
        "input_other": getattr(usage, "input_other", 0),
        "input_cached": getattr(usage, "input_cached", 0),
        "output": getattr(usage, "output", 0),
        "total": getattr(usage, "total", 0),
    }


def _serialize_response(resp: LLMResponse | None, maxlen: int) -> dict | None:
    """只提取 LLMResponse 中可读的字段，跳过 raw_completion 等大对象。"""
    if resp is None:
        return None
    reasoning = getattr(resp, "reasoning_content", None)
    return {
        "role": getattr(resp, "role", None),
        "completion_text": _truncate(
            getattr(resp, "_completion_text", "") or "", maxlen
        ),
        "reasoning_content": (
            _truncate(reasoning, maxlen) if isinstance(reasoning, str) else None
        ),
        "tools_call_name": list(getattr(resp, "tools_call_name", []) or []),
        "tools_call_args": _serialize(getattr(resp, "tools_call_args", []) or [], maxlen),
        "tools_call_ids": list(getattr(resp, "tools_call_ids", []) or []),
        "usage": _usage_dict(getattr(resp, "usage", None)),
        "id": getattr(resp, "id", None),
    }


def _last_user_preview(req: dict) -> str:
    """取请求中最后一条用户文本，作为列表预览。"""
    prompt = req.get("prompt")
    if isinstance(prompt, str) and prompt:
        return _truncate(prompt, 160)
    contexts = req.get("contexts")
    if isinstance(contexts, list):
        for msg in reversed(contexts):
            if not isinstance(msg, dict) or msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, str) and content:
                return _truncate(content, 160)
            if isinstance(content, list):
                for part in reversed(content):
                    if isinstance(part, dict) and part.get("type") == "text":
                        text = part.get("text") or ""
                        if text:
                            return _truncate(text, 160)
    return ""


def summarize(rec: dict) -> dict:
    """从完整记录生成轻量摘要，用于列表与 SSE 推送。"""
    req = rec.get("request") or {}
    resp = rec.get("response") or {}
    contexts = req.get("contexts")
    msg_count = len(contexts) if isinstance(contexts, list) else 0
    tools = req.get("func_tool")
    tool_names = (
        [t.get("name") for t in tools if isinstance(t, dict) and t.get("name")]
        if isinstance(tools, list)
        else []
    )
    usage = resp.get("usage") or {}
    return {
        "id": rec["id"],
        "ts": rec["ts"],
        "time": rec["time"],
        "provider_id": rec["provider"]["id"],
        "provider_type": rec["provider"]["type"],
        "model": rec["model"],
        "session_id": rec.get("session_id"),
        "streaming": rec.get("streaming", False),
        "status": rec.get("status", "ok"),
        "duration_ms": rec.get("duration_ms"),
        "message_count": msg_count,
        "has_system_prompt": bool(req.get("system_prompt")),
        "tool_count": len(tool_names),
        "tool_names": tool_names,
        "total_tokens": usage.get("total", 0),
        "input_tokens": (usage.get("input_other", 0) or 0)
        + (usage.get("input_cached", 0) or 0),
        "output_tokens": usage.get("output", 0),
        "preview": _last_user_preview(req),
        "response_preview": _truncate(
            resp.get("completion_text") or "", 160
        )
        if resp
        else "",
    }


class LLMRecorder:
    """内存环形缓冲区 + SSE 监听队列。"""

    def __init__(self) -> None:
        self.enabled = True
        self.record_response = True
        self.max_content_length = 50_000
        self.records: deque[dict] = deque(maxlen=200)
        self.listeners: set[asyncio.Queue] = set()

    def set_capacity(self, max_records: int) -> None:
        if max_records != (self.records.maxlen or 0):
            self.records = deque(self.records, maxlen=max_records)

    def commit(
        self,
        provider: Any,
        request_info: dict,
        response: LLMResponse | None,
        error: str | None,
        started: float,
        streaming: bool,
    ) -> None:
        """记录一次完成的 LLM 调用。必须在任何异常下安全执行。"""
        try:
            if not self.enabled:
                return
            now = time.time()
            maxlen = self.max_content_length
            rec: dict[str, Any] = {
                "id": uuid.uuid4().hex[:12],
                "ts": now,
                "time": datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S"),
                "provider": _provider_info(provider),
                "model": request_info.get("model")
                or getattr(provider, "model_name", None),
                "session_id": request_info.get("session_id"),
                "streaming": streaming,
                "status": "error" if error else "ok",
                "duration_ms": round((now - started) * 1000, 1),
                "request": request_info,
                "response": (
                    _serialize_response(response, maxlen)
                    if (self.record_response and isinstance(response, LLMResponse))
                    else None
                ),
                "error": error,
            }
            self.records.append(rec)
            summary = summarize(rec)
            payload = {"event": "request", "summary": summary}
            for q in list(self.listeners):
                try:
                    q.put_nowait(payload)
                except asyncio.QueueFull:
                    pass
        except Exception as exc:  # noqa: BLE001
            logger.warning("[context_toolbox] commit record failed: %s", exc)

    def search(self, q: str) -> list[dict]:
        """在完整记录里做不区分大小写的子串搜索，返回匹配的记录（倒序）。"""
        ql = q.lower()
        out = []
        for rec in reversed(self.records):
            try:
                blob = json.dumps(
                    {
                        "request": rec.get("request"),
                        "response": rec.get("response"),
                        "session_id": rec.get("session_id"),
                        "model": rec.get("model"),
                        "provider": rec.get("provider"),
                    },
                    ensure_ascii=False,
                    default=str,
                )
            except Exception:  # noqa: BLE001
                blob = ""
            if ql in blob[:_SEARCH_BLOB_MAX].lower():
                out.append(rec)
        return out


def _extract_request(
    sig: inspect.Signature, args: tuple, kwargs: dict, maxlen: int
) -> dict:
    """从调用参数中提取需要记录的请求字段。"""
    arguments: dict[str, Any] = {}
    try:
        bound = sig.bind_partial(None, *args, **kwargs)
        arguments = dict(bound.arguments)
        arguments.pop("self", None)
        # 若原函数使用 **kwargs（VAR_KEYWORD），把其中的字段展开到顶层
        var_kw = arguments.pop("kwargs", None)
        if isinstance(var_kw, dict):
            for k, v in var_kw.items():
                arguments.setdefault(k, v)
    except TypeError:
        arguments = dict(kwargs)
    info: dict[str, Any] = {}
    for name in _RECORDED_FIELDS:
        if name in arguments:
            info[name] = _serialize(arguments[name], maxlen)
    extra = {
        k: v for k, v in arguments.items() if k not in _RECORDED_FIELDS
    }
    if extra:
        info["extra"] = _serialize(extra, maxlen)
    return info


def _make_text_chat_wrapper(original, recorder: LLMRecorder):
    sig = inspect.signature(original)

    async def wrapper(self, *args, **kwargs):
        depth = _depth.get()
        if depth > 0 or not recorder.enabled:
            return await original(self, *args, **kwargs)
        token = _depth.set(depth + 1)
        started = time.time()
        try:
            request_info = _extract_request(sig, args, kwargs, recorder.max_content_length)
        except Exception:  # noqa: BLE001
            request_info = {}
        result = None
        error = None
        try:
            result = await original(self, *args, **kwargs)
            return result
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            _depth.reset(token)
            recorder.commit(
                self, request_info, result if isinstance(result, LLMResponse) else None,
                error, started, streaming=False,
            )

    setattr(wrapper, _WRAPPED_ATTR, True)
    return wrapper


def _make_stream_wrapper(original, recorder: LLMRecorder):
    sig = inspect.signature(original)

    async def wrapper(self, *args, **kwargs):
        depth = _depth.get()
        if depth > 0 or not recorder.enabled:
            async for item in original(self, *args, **kwargs):
                yield item
            return
        token = _depth.set(depth + 1)
        started = time.time()
        try:
            request_info = _extract_request(sig, args, kwargs, recorder.max_content_length)
        except Exception:  # noqa: BLE001
            request_info = {}
        final_resp: LLMResponse | None = None
        error = None
        try:
            async for resp in original(self, *args, **kwargs):
                if isinstance(resp, LLMResponse) and not getattr(resp, "is_chunk", False):
                    final_resp = resp
                yield resp
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            _depth.reset(token)
            recorder.commit(self, request_info, final_resp, error, started, streaming=True)

    setattr(wrapper, _WRAPPED_ATTR, True)
    return wrapper


def _all_subclasses(cls: type) -> set[type]:
    seen: set[type] = set()
    stack = [cls]
    while stack:
        current = stack.pop()
        for sub in current.__subclasses__():
            if sub not in seen:
                seen.add(sub)
                stack.append(sub)
    return seen


_sources_imported = False


def _ensure_provider_classes_loaded() -> None:
    """Provider 源模块在 AstrBot 中是懒加载的（晚于插件 initialize），
    这里主动导入所有内置 Provider 源模块，保证打补丁时类已存在。
    单个模块导入失败（缺少可选依赖）不影响其它模块。"""
    global _sources_imported
    if _sources_imported:
        return
    _sources_imported = True
    try:
        import importlib
        import pkgutil

        import astrbot.core.provider.sources as sources_pkg

        for modinfo in pkgutil.iter_modules(sources_pkg.__path__):
            try:
                importlib.import_module(f"{sources_pkg.__name__}.{modinfo.name}")
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "[context_toolbox] import provider source %s skipped: %s",
                    modinfo.name,
                    exc,
                )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[context_toolbox] ensure provider classes failed: %s", exc
        )


class ContextToolboxPlugin(Star):
    """LLM 请求上下文监控插件。"""

    def __init__(self, context: Context, config: dict | None = None) -> None:
        super().__init__(context, config)
        conf = config or {}
        self.recorder = LLMRecorder()
        self.recorder.enabled = bool(conf.get("enabled", True))
        self.recorder.record_response = bool(conf.get("record_response", True))
        try:
            self.recorder.max_content_length = max(
                1000, int(conf.get("max_content_length", 50000))
            )
        except (TypeError, ValueError):
            pass
        try:
            self.recorder.set_capacity(max(10, int(conf.get("max_records", 200))))
        except (TypeError, ValueError):
            pass
        # 记录被 patch 前的原始方法，terminate 时恢复
        self._originals: dict[tuple[type, str], Any] = {}
        self._register_web_apis(context)

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #

    async def initialize(self) -> None:
        """所有插件加载完成后，包装 Provider 的 text_chat / text_chat_stream。"""
        self._sweep_patch()

    def _sweep_patch(self) -> int:
        """扫描 Provider 及其所有子类，包装尚未包装的
        text_chat / text_chat_stream。幂等，可重复调用。
        返回本次新包装的方法数。"""
        _ensure_provider_classes_loaded()
        classes = {Provider} | _all_subclasses(Provider)
        patched = 0
        for cls in classes:
            for name in ("text_chat", "text_chat_stream"):
                fn = cls.__dict__.get(name)
                if fn is None or getattr(fn, _WRAPPED_ATTR, False):
                    continue
                try:
                    inspect.signature(fn)
                except (TypeError, ValueError):
                    continue
                self._originals[(cls, name)] = fn
                if name == "text_chat":
                    setattr(cls, name, _make_text_chat_wrapper(fn, self.recorder))
                else:
                    setattr(cls, name, _make_stream_wrapper(fn, self.recorder))
                patched += 1
        if patched:
            logger.info(
                "[context_toolbox] hooked %d provider methods across %d classes",
                patched,
                len(classes),
            )
        return patched

    async def terminate(self) -> None:
        """插件卸载时恢复原始方法，取消所有 SSE 监听。"""
        for (cls, name), fn in self._originals.items():
            try:
                setattr(cls, name, fn)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[context_toolbox] restore %s.%s failed: %s",
                    cls.__name__, name, exc,
                )
        self._originals.clear()
        for q in list(self.recorder.listeners):
            try:
                q.put_nowait({"event": "plugin_stopped"})
            except asyncio.QueueFull:
                pass
        self.recorder.listeners.clear()

    # ------------------------------------------------------------------ #
    # Web API
    # ------------------------------------------------------------------ #

    def _register_web_apis(self, context: Context) -> None:
        context.register_web_api(
            f"/{PLUGIN_NAME}/requests", self.api_list, ["GET"], "List LLM request summaries"
        )
        context.register_web_api(
            f"/{PLUGIN_NAME}/requests/<req_id>", self.api_detail, ["GET"], "Get one LLM request detail"
        )
        context.register_web_api(
            f"/{PLUGIN_NAME}/stats", self.api_stats, ["GET"], "Recorder statistics"
        )
        context.register_web_api(
            f"/{PLUGIN_NAME}/clear", self.api_clear, ["POST"], "Clear all records"
        )
        context.register_web_api(
            f"/{PLUGIN_NAME}/export", self.api_export, ["GET"], "Export records as JSON file"
        )
        context.register_web_api(
            f"/{PLUGIN_NAME}/stream", self.api_stream, ["GET"], "SSE live request stream"
        )

    async def api_list(self):
        # 补扫：捕获插件加载之后才导入的 Provider 类（幂等）
        self._sweep_patch()
        try:
            limit = int(request.query.get("limit", 100))
        except (TypeError, ValueError):
            limit = 100
        limit = max(1, min(500, limit))
        provider = request.query.get("provider")
        session = request.query.get("session")
        q = (request.query.get("q") or "").strip()

        if q:
            matched = self.recorder.search(q)
        else:
            matched = list(reversed(self.recorder.records))
        if provider:
            matched = [r for r in matched if r["provider"]["id"] == provider]
        if session:
            matched = [r for r in matched if (r.get("session_id") or "") == session]

        total = len(matched)
        items = [summarize(r) for r in matched[:limit]]
        return json_response(
            {
                "total": total,
                "items": items,
                "enabled": self.recorder.enabled,
                "capacity": self.recorder.records.maxlen,
            }
        )

    async def api_detail(self, req_id: str):
        for rec in reversed(self.recorder.records):
            if rec["id"] == req_id:
                return json_response(rec)
        return error_response("record not found", status_code=404)

    async def api_stats(self):
        self._sweep_patch()
        recs = self.recorder.records
        total = len(recs)
        errors = sum(1 for r in recs if r.get("status") == "error")
        total_tokens = 0
        input_tokens = 0
        output_tokens = 0
        per_provider: dict[str, dict] = {}
        for r in recs:
            usage = (r.get("response") or {}).get("usage") or {}
            t_in = (usage.get("input_other", 0) or 0) + (usage.get("input_cached", 0) or 0)
            t_out = usage.get("output", 0) or 0
            total_tokens += usage.get("total", 0) or 0
            input_tokens += t_in
            output_tokens += t_out
            pid = r["provider"]["id"]
            entry = per_provider.setdefault(
                pid, {"count": 0, "errors": 0, "tokens": 0, "models": set()}
            )
            entry["count"] += 1
            if r.get("status") == "error":
                entry["errors"] += 1
            entry["tokens"] += usage.get("total", 0) or 0
            if r.get("model"):
                entry["models"].add(r["model"])
        providers = [
            {
                "id": pid,
                "count": v["count"],
                "errors": v["errors"],
                "tokens": v["tokens"],
                "models": sorted(v["models"]),
            }
            for pid, v in sorted(per_provider.items(), key=lambda kv: -kv[1]["count"])
        ]
        return json_response(
            {
                "total": total,
                "errors": errors,
                "total_tokens": total_tokens,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "enabled": self.recorder.enabled,
                "capacity": self.recorder.records.maxlen,
                "providers": providers,
            }
        )

    async def api_clear(self):
        self.recorder.records.clear()
        return json_response({"cleared": True})

    async def api_export(self):
        records = list(self.recorder.records)
        data_dir = Path(get_astrbot_plugin_data_path()) / PLUGIN_NAME
        data_dir.mkdir(parents=True, exist_ok=True)
        fname = f"llm_requests_{time.strftime('%Y%m%d_%H%M%S')}.json"
        path = data_dir / fname
        path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return file_response(path, filename=fname, content_type="application/json")

    async def api_stream(self):
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        self.recorder.listeners.add(q)

        async def gen():
            try:
                yield 'event: ready\ndata: {"ok": true}\n\n'
                while True:
                    try:
                        item = await asyncio.wait_for(q.get(), timeout=25)
                        yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
                    except asyncio.TimeoutError:
                        yield ": keep-alive\n\n"
            finally:
                self.recorder.listeners.discard(q)

        return stream_response(gen())
