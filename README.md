# DoubleHighC

# AI 文章爬虫与智能筛选系统

**DoubleHighC**是一个自动化爬虫与 AI 智能筛选系统，用于抓取文章，通过LLM进行相关性筛选，并将结果存储到 PostgreSQL 和 Redis，同时支持 Notion 自动化记录。

---

## 1. 功能概览

- **爬虫模块**：从指定网站（如 Tophub）抓取文章列表。
- **AI 智能筛选**：使用 Google Generative AI（Gemini）对文章标题进行相关性评分，筛选出高价值内容。
- **数据存储**：
  - 将筛选结果存储到 PostgreSQL 的 `screening_results` 表。
  - 将通过筛选的文章存入 `articles` 表，(并推送任务 ID 到 Redis 队列)。
- **Notion 自动化**：将筛选结果自动记录到 Notion 页面（可选）。
- **并发与重试机制**：支持并发筛选，并内置重试逻辑，提高稳定性。

---

## 2. 环境要求

- Python 3.8+
- PostgreSQL
- Redis（可选）
- Google Generative AI API 密钥
- Notion Integration Token（可选）

---

## 3. 联系作者

