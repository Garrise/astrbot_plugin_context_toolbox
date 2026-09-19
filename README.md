# astrbot_plugin_context_toolbox

LLM 请求上下文监控面板（Context Toolbox）。

监控当前 AstrBot 实例中的**每一次 LLM 请求**，记录请求的上下文结构与响应内容，
并可在 AstrBot WebUI 的插件页面中浏览、搜索、实时跟踪与导出。

## 功能

- **插件页面**：WebUI → 插件 → 本插件详情页 → `LLM 请求监控` 页面。
- **查看每一个 LLM 请求**：列表展示时间、Provider、模型、会话、消息数、Token 用量、耗时、状态（OK/错误）、是否流式、工具数量。
- **查看每个请求的上下文结构和内容**：
  - System Prompt
  - 对话 `contexts`（每条消息的角色、文本/思考/图片/音频内容分块、tool_calls、tool_call_id）
  - 附加用户内容 `extra_user_content_parts`、`image_urls`、`audio_urls`
  - 工具定义 `func_tool`（名称、描述、JSON Schema 参数）
  - `tool_calls_result`、`tool_choice` 等其它请求参数
  - 响应 `response`：回复文本、思考内容、工具调用、Token 用量、错误信息
- **实时跟踪**：SSE 实时推送新请求（页面点击"实时"开启）。
- **搜索/过滤**：全文搜索请求与响应内容，按 Provider 过滤。
- **可选持久化**：配置 `persist_enabled` 后记录异步落盘到
  `data/plugin_data/astrbot_plugin_context_toolbox/records.jsonl`，重启 AstrBot 后自动加载最近记录；页面顶部实时显示持久化状态与保存路径。
- **导出**：一键导出全部记录为 JSON 文件。

## 实现说明

- 通过包装 `Provider.text_chat` / `Provider.text_chat_stream`（monkey patch，插件卸载时自动恢复）
  捕获所有走 AstrBot Provider 的 LLM 调用，包括 Agent 工具循环中的多轮请求。
- 记录保存在内存环形缓冲区中（默认 200 条，可配置）。
- 开启 `persist_enabled` 后，每条记录经异步队列追加写入
  `data/plugin_data/astrbot_plugin_context_toolbox/records.jsonl`（JSONL 格式），
  不阻塞 LLM 请求路径；插件启动时自动加载最近 `max_records` 条并压缩文件。
  关闭时仅存内存，重启即清空。
- 超长文本按配置截断（默认单字段 50000 字符），避免内存与文件膨胀。
- 页面"清空"会同时清空内存与持久化文件；"导出"始终生成独立 JSON 快照。
- 页面通过 AstrBot 插件 Pages bridge（`window.AstrBotPluginPage`）访问后端 Web API。

## 配置（_conf_schema.json）

| 配置项 | 默认 | 说明 |
| --- | --- | --- |
| `enabled` | `true` | 是否记录 LLM 请求 |
| `persist_enabled` | `false` | 是否持久化到磁盘（`data/plugin_data/astrbot_plugin_context_toolbox/records.jsonl`），重启后自动加载 |
| `max_records` | `200` | 内存保留的最大记录条数（持久化模式下重启加载同条数） |
| `max_content_length` | `50000` | 单个文本字段最大字符数，超出截断 |
| `record_response` | `true` | 是否记录响应内容（关闭则只记录请求侧） |

修改配置后需要重载插件生效。

## 使用

1. 将本目录放入 `data/plugins/`，在 WebUI 中启用/重载插件。
2. 正常聊天，让 Bot 产生 LLM 请求。
3. 打开本插件详情页的 `LLM 请求监控` 页面查看。

## 后端 Web API

注册于 `/{plugin}/...`，页面端通过 bridge 以相对 endpoint 访问：

- `GET requests?limit=&provider=&session=&q=` — 摘要列表
- `GET requests/<id>` — 单条完整记录
- `GET stats` — 统计信息
- `POST clear` — 清空记录
- `GET export` — 导出 JSON 文件
- `GET stream` — SSE 实时流
