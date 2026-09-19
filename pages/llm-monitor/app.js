/* LLM 请求监控页面逻辑：通过 AstrBotPluginPage bridge 与插件后端通信 */
const bridge = window.AstrBotPluginPage;

const state = {
  items: [],
  total: 0,
  selectedId: null,
  live: false,
  sseId: null,
  filters: { q: "", provider: "" },
  knownProviders: new Set(),
};

const $ = (sel) => document.querySelector(sel);

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function fmtTime(ts) {
  const d = new Date(ts * 1000);
  const p = (n) => String(n).padStart(2, "0");
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

function fmtDuration(ms) {
  if (ms === null || ms === undefined) return "-";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

function fmtNum(n) {
  if (!n) return "0";
  return String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

function safeParseJSON(value) {
  if (typeof value !== "string") return value;
  try {
    return JSON.parse(value);
  } catch {
    return value;
  }
}

function jsonBlock(obj, cls) {
  let text;
  try {
    text = typeof obj === "string" ? obj : JSON.stringify(obj, null, 2);
  } catch {
    text = String(obj);
  }
  return el("pre", `json ${cls || ""}`, text);
}

/* ---------------- 统计栏 ---------------- */

async function loadStats() {
  try {
    const s = await bridge.apiGet("stats");
    renderStats(s);
  } catch (e) {
    console.error("stats failed", e);
  }
}

function renderStats(s) {
  const box = $("#stats");
  box.innerHTML = "";
  const chip = (label, value, cls) => {
    const c = el("div", `chip ${cls || ""}`);
    c.appendChild(el("span", "chip-value", value));
    c.appendChild(el("span", "chip-label", label));
    return c;
  };
  box.appendChild(chip("请求", fmtNum(s.total)));
  box.appendChild(chip("错误", fmtNum(s.errors), s.errors ? "err" : ""));
  box.appendChild(chip("Tokens", fmtNum(s.total_tokens)));
  box.appendChild(chip("容量", `${fmtNum(s.total)}/${fmtNum(s.capacity)}`));
  if (!s.enabled) {
    box.appendChild(el("span", "warn-badge", "记录已关闭"));
  }
}

/* ---------------- Provider 下拉 ---------------- */

function updateProviderOptions(items) {
  const sel = $("#provider");
  for (const it of items) {
    if (it.provider_id && !state.knownProviders.has(it.provider_id)) {
      state.knownProviders.add(it.provider_id);
      const opt = el("option", "", it.provider_id);
      opt.value = it.provider_id;
      sel.appendChild(opt);
    }
  }
}

/* ---------------- 请求列表 ---------------- */

async function loadList() {
  const params = { limit: 100 };
  if (state.filters.q) params.q = state.filters.q;
  if (state.filters.provider) params.provider = state.filters.provider;
  try {
    const res = await bridge.apiGet("requests", params);
    state.items = res.items || [];
    state.total = res.total || 0;
    updateProviderOptions(state.items);
    renderList();
  } catch (e) {
    console.error("list failed", e);
    $("#list").innerHTML = "";
    $("#list").appendChild(el("div", "empty", `加载失败: ${e.message}`));
  }
}

function renderList() {
  const list = $("#list");
  list.innerHTML = "";
  if (!state.items.length) {
    list.appendChild(el("div", "empty", "暂无记录，发送一条消息试试"));
    return;
  }
  for (const it of state.items) {
    const row = el("div", "req-row" + (it.id === state.selectedId ? " selected" : ""));

    const left = el("div", "rr-left");
    left.appendChild(el("span", "rr-time", fmtTime(it.ts)));
    const badges = el("div", "rr-badges");
    badges.appendChild(
      el("span", "badge " + (it.status === "error" ? "err" : "ok"), it.status === "error" ? "错误" : "OK")
    );
    if (it.streaming) badges.appendChild(el("span", "badge stream", "流式"));
    if (it.tool_count) badges.appendChild(el("span", "badge tools", `${it.tool_count} 工具`));
    left.appendChild(badges);
    row.appendChild(left);

    const mid = el("div", "rr-mid");
    mid.appendChild(el("div", "rr-provider", `${it.provider_id} · ${it.model || "-"}`));
    mid.appendChild(el("div", "rr-preview", it.preview || "(无文本)"));
    if (it.session_id) mid.appendChild(el("div", "rr-session", it.session_id));
    row.appendChild(mid);

    const right = el("div", "rr-right");
    right.appendChild(el("div", "", `${it.message_count} 条`));
    right.appendChild(el("div", "", it.total_tokens ? `${fmtNum(it.total_tokens)} tok` : "-"));
    right.appendChild(el("div", "", fmtDuration(it.duration_ms)));
    row.appendChild(right);

    row.addEventListener("click", () => selectRequest(it.id));
    list.appendChild(row);
  }
}

/* ---------------- 详情渲染 ---------------- */

async function selectRequest(id) {
  state.selectedId = id;
  document.querySelectorAll(".req-row").forEach((r) => {
    // 行上没有 id，重新渲染列表来同步选中态最简单
  });
  renderList();
  try {
    const rec = await bridge.apiGet(`requests/${id}`);
    renderDetail(rec);
  } catch (e) {
    $("#detail").innerHTML = "";
    $("#detail").appendChild(el("div", "empty", `加载详情失败: ${e.message}`));
  }
}

function section(title, bodyNode, open) {
  const d = el("details", "section");
  if (open) d.open = true;
  const s = el("summary", "section-title", title);
  d.appendChild(s);
  d.appendChild(bodyNode);
  return d;
}

function renderDetail(rec) {
  const box = $("#detail");
  box.innerHTML = "";

  const head = el("div", "detail-head");
  const titleRow = el("div", "detail-title");
  titleRow.appendChild(el("span", "dt-id", `#${rec.id}`));
  titleRow.appendChild(el("span", "dt-provider", `${rec.provider.id} · ${rec.model || "-"}`));
  head.appendChild(titleRow);

  const meta = el("div", "detail-meta");
  const metaChip = (k, v) => {
    const c = el("span", "meta-chip");
    c.appendChild(el("b", "", k + " "));
    c.appendChild(el("span", "", String(v ?? "-")));
    return c;
  };
  meta.appendChild(metaChip("时间", rec.time));
  meta.appendChild(metaChip("类型", rec.provider.type));
  meta.appendChild(metaChip("会话", rec.session_id || "-"));
  meta.appendChild(metaChip("流式", rec.streaming ? "是" : "否"));
  meta.appendChild(metaChip("耗时", fmtDuration(rec.duration_ms)));
  meta.appendChild(
    el("span", "badge " + (rec.status === "error" ? "err" : "ok"), rec.status === "error" ? "错误" : "OK")
  );
  head.appendChild(meta);
  box.appendChild(head);

  if (rec.error) {
    box.appendChild(el("div", "error-box", rec.error));
  }

  const req = rec.request || {};

  if (req.system_prompt) {
    box.appendChild(section(`System Prompt（${(req.system_prompt || "").length} 字符）`, jsonBlock(req.system_prompt), false));
  }

  const contexts = Array.isArray(req.contexts) ? req.contexts : null;
  if (contexts) {
    const wrap = el("div", "messages");
    contexts.forEach((msg, i) => wrap.appendChild(renderMessage(msg, i)));
    if (!contexts.length) wrap.appendChild(el("div", "empty", "（空上下文）"));
    box.appendChild(section(`对话上下文 contexts（${contexts.length} 条）`, wrap, true));
  }

  if (req.prompt && !contexts) {
    box.appendChild(section("Prompt", jsonBlock(req.prompt), true));
  }

  if (Array.isArray(req.extra_user_content_parts) && req.extra_user_content_parts.length) {
    const wrap = el("div", "messages");
    req.extra_user_content_parts.forEach((p, i) => wrap.appendChild(renderPart(p, i)));
    box.appendChild(section(`附加用户内容 extra_user_content_parts（${req.extra_user_content_parts.length}）`, wrap, false));
  }

  if (Array.isArray(req.image_urls) && req.image_urls.length) {
    box.appendChild(section(`image_urls（${req.image_urls.length}）`, jsonBlock(req.image_urls), false));
  }
  if (Array.isArray(req.audio_urls) && req.audio_urls.length) {
    box.appendChild(section(`audio_urls（${req.audio_urls.length}）`, jsonBlock(req.audio_urls), false));
  }

  const tools = Array.isArray(req.func_tool) ? req.func_tool : null;
  if (tools && tools.length) {
    const wrap = el("div", "tools");
    for (const t of tools) {
      const td = el("details", "tool-item");
      td.appendChild(el("summary", "tool-name", `🔧 ${t.name || "?"}`));
      if (t.description) td.appendChild(el("div", "tool-desc", t.description));
      td.appendChild(jsonBlock(t.parameters ?? {}));
      wrap.appendChild(td);
    }
    box.appendChild(section(`工具定义 func_tool（${tools.length} 个）`, wrap, false));
  }

  if (req.tool_calls_result !== undefined && req.tool_calls_result !== null) {
    box.appendChild(section("tool_calls_result", jsonBlock(req.tool_calls_result), false));
  }
  if (req.tool_choice !== undefined) {
    box.appendChild(section("tool_choice", jsonBlock(req.tool_choice), false));
  }
  if (req.extra) {
    box.appendChild(section("其它参数", jsonBlock(req.extra), false));
  }

  const resp = rec.response;
  if (resp) {
    const rwrap = el("div", "response");
    if (resp.reasoning_content) {
      rwrap.appendChild(
        section(`💭 思考 reasoning_content（${resp.reasoning_content.length} 字符）`, jsonBlock(resp.reasoning_content), false)
      );
    }
    rwrap.appendChild(
      section(`回复文本（${(resp.completion_text || "").length} 字符）`, jsonBlock(resp.completion_text || "（空）"), true)
    );
    if (resp.tools_call_name && resp.tools_call_name.length) {
      const tc = el("div", "resp-tools");
      resp.tools_call_name.forEach((name, i) => {
        const item = el("div", "toolcall");
        item.appendChild(el("div", "toolcall-name", `🔧 ${name}（id: ${(resp.tools_call_ids || [])[i] || "-"}）`));
        item.appendChild(jsonBlock(safeParseJSON((resp.tools_call_args || [])[i])));
        tc.appendChild(item);
      });
      rwrap.appendChild(section(`工具调用（${resp.tools_call_name.length} 次）`, tc, true));
    }
    if (resp.usage) {
      const u = resp.usage;
      const uw = el("div", "usage");
      uw.appendChild(el("span", "meta-chip", `输入 ${fmtNum(u.input_other)}`));
      uw.appendChild(el("span", "meta-chip", `缓存 ${fmtNum(u.input_cached)}`));
      uw.appendChild(el("span", "meta-chip", `输出 ${fmtNum(u.output)}`));
      uw.appendChild(el("span", "meta-chip", `合计 ${fmtNum(u.total)}`));
      rwrap.appendChild(section("Token 用量", uw, true));
    }
    box.appendChild(section("响应 response", rwrap, true));
  } else if (rec.status !== "error") {
    box.appendChild(el("div", "empty", "（未记录响应内容）"));
  }
}

function renderMessage(msg, idx) {
  const role = msg.role || "?";
  const card = el("div", `msg msg-${role}`);
  const head = el("div", "msg-head");
  head.appendChild(el("span", "msg-idx", `#${idx + 1}`));
  head.appendChild(el("span", `role role-${role}`, role));
  if (msg.tool_call_id) head.appendChild(el("span", "msg-tcid", `tool_call_id: ${msg.tool_call_id}`));
  card.appendChild(head);

  const content = msg.content;
  if (typeof content === "string") {
    card.appendChild(el("pre", "msg-text", content || "（空）"));
  } else if (Array.isArray(content)) {
    content.forEach((part, i) => card.appendChild(renderPart(part, i)));
  } else if (content && typeof content === "object") {
    card.appendChild(jsonBlock(content));
  } else {
    card.appendChild(el("div", "msg-empty", "（无内容）"));
  }

  if (Array.isArray(msg.tool_calls) && msg.tool_calls.length) {
    const tc = el("div", "msg-toolcalls");
    for (const call of msg.tool_calls) {
      const item = el("div", "toolcall");
      const fn = call.function || {};
      item.appendChild(el("div", "toolcall-name", `🔧 ${fn.name || "?"}（id: ${call.id || "-"}）`));
      item.appendChild(jsonBlock(safeParseJSON(fn.arguments)));
      tc.appendChild(item);
    }
    card.appendChild(tc);
  }
  return card;
}

function renderPart(part) {
  const t = part && part.type;
  if (t === "text") {
    return el("pre", "part part-text", part.text || "");
  }
  if (t === "think") {
    const d = el("details", "part part-think");
    d.appendChild(el("summary", "", `💭 think（${(part.think || "").length} 字符）`));
    d.appendChild(el("pre", "", part.think || ""));
    return d;
  }
  if (t === "image_url") {
    const wrap = el("div", "part part-image");
    const iu = part.image_url || {};
    const url = typeof iu === "string" ? iu : iu.url || "";
    wrap.appendChild(el("div", "part-label", "🖼 image_url"));
    if (typeof url === "string" && /^https?:/i.test(url)) {
      const img = el("img", "part-img");
      img.src = url;
      img.loading = "lazy";
      wrap.appendChild(img);
    }
    wrap.appendChild(el("pre", "part-url", typeof url === "string" ? (url.length > 300 ? url.slice(0, 300) + "…" : url) : JSON.stringify(url)));
    return wrap;
  }
  if (t === "audio_url") {
    const wrap = el("div", "part part-audio");
    const au = part.audio_url || {};
    const url = typeof au === "string" ? au : au.url || "";
    wrap.appendChild(el("div", "part-label", "🔊 audio_url"));
    if (typeof url === "string" && /^https?:/i.test(url)) {
      const audio = el("audio", "part-audio-el");
      audio.src = url;
      audio.controls = true;
      wrap.appendChild(audio);
    }
    wrap.appendChild(el("pre", "part-url", typeof url === "string" ? (url.length > 300 ? url.slice(0, 300) + "…" : url) : JSON.stringify(url)));
    return wrap;
  }
  return jsonBlock(part);
}

/* ---------------- 实时 SSE ---------------- */

function matchesFilters(summary) {
  if (state.filters.provider && summary.provider_id !== state.filters.provider) return false;
  if (state.filters.q) {
    const q = state.filters.q.toLowerCase();
    const hay = `${summary.preview || ""} ${summary.response_preview || ""} ${summary.session_id || ""} ${summary.model || ""} ${summary.provider_id || ""}`.toLowerCase();
    if (!hay.includes(q)) return false;
  }
  return true;
}

function updateLiveBtn() {
  const btn = $("#live");
  btn.textContent = state.live ? "⏸ 实时中" : "▶ 实时";
  btn.classList.toggle("active", state.live);
}

async function setLive(on) {
  if (on && !state.live) {
    try {
      state.sseId = await bridge.subscribeSSE(
        "stream",
        {
          onOpen() {
            state.live = true;
            updateLiveBtn();
          },
          onMessage(ev) {
            const data = ev.parsed;
            if (!data || typeof data !== "object") return;
            if (data.event === "plugin_stopped") {
              state.live = false;
              state.sseId = null;
              updateLiveBtn();
              return;
            }
            if (data.event === "request" && data.summary) {
              updateProviderOptions([data.summary]);
              if (matchesFilters(data.summary)) {
                state.items.unshift(data.summary);
                if (state.items.length > 100) state.items.pop();
                state.total += 1;
                renderList();
                loadStats();
              }
            }
          },
          onError() {
            state.live = false;
            state.sseId = null;
            updateLiveBtn();
          },
        },
        {},
      );
      state.live = true;
      updateLiveBtn();
    } catch (e) {
      console.error("SSE subscribe failed", e);
      state.live = false;
      updateLiveBtn();
    }
  } else if (!on && state.live) {
    if (state.sseId) {
      try {
        await bridge.unsubscribeSSE(state.sseId);
      } catch {
        /* ignore */
      }
    }
    state.live = false;
    state.sseId = null;
    updateLiveBtn();
  }
}

/* ---------------- 事件绑定 ---------------- */

let searchTimer = null;
$("#search").addEventListener("input", (e) => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => {
    state.filters.q = e.target.value.trim();
    loadList();
  }, 300);
});

$("#provider").addEventListener("change", (e) => {
  state.filters.provider = e.target.value;
  loadList();
});

$("#refresh").addEventListener("click", () => {
  loadList();
  loadStats();
});

$("#live").addEventListener("click", () => setLive(!state.live));

$("#export").addEventListener("click", async () => {
  try {
    await bridge.download("export", {}, "llm_requests.json");
  } catch (e) {
    alert(`导出失败: ${e.message}`);
  }
});

$("#clear").addEventListener("click", async () => {
  if (!confirm("确定清空内存中的所有 LLM 请求记录吗？")) return;
  try {
    await bridge.apiPost("clear", {});
    state.items = [];
    state.total = 0;
    state.selectedId = null;
    renderList();
    loadStats();
    $("#detail").innerHTML = "";
    $("#detail").appendChild(el("div", "empty", "已清空"));
  } catch (e) {
    alert(`清空失败: ${e.message}`);
  }
});

window.addEventListener("beforeunload", () => {
  if (state.sseId) {
    try {
      bridge.unsubscribeSSE(state.sseId);
    } catch {
      /* ignore */
    }
  }
});

/* ---------------- 启动 ---------------- */

async function main() {
  await bridge.ready();
  await loadStats();
  await loadList();
}

main().catch((e) => {
  console.error("init failed", e);
  $("#list").innerHTML = "";
  $("#list").appendChild(el("div", "empty", `初始化失败: ${e.message}`));
});
