# DoubleHighC

# AI 文章爬虫与智能筛选系统

**DoubleHighC** 是一个自动化爬虫与 AI 智能筛选系统，用于抓取文章，通过 LLM 进行相关性筛选，并将结果存储到 PostgreSQL 和 Redis，同时支持 Notion 自动化记录。

---

## 1. 功能概览

- **爬虫模块**：从指定网站抓取文章列表。
- **AI 智能筛选**：使用 Google Generative AI（Gemini）对文章标题进行相关性评分，筛选出高价值内容。
- **深度分析 Worker**：对通过筛选的文章进行全文抓取与深度 LLM 分析。
- **数据存储**：
  - 将筛选结果存储到 PostgreSQL 的 `screening_results` 表。
  - 将通过筛选的文章存入 `articles` 表（并可推送任务 ID 到 Redis 队列）。
  - 将深度分析结果存入 `analysis_results` 表。
- **Notion 自动化**：将分析结果自动记录到 Notion 页面（可选）。
- **并发与重试机制**：支持并发筛选，API 密钥池轮换，内置指数退避重试逻辑。
- **Prompt 注入防护**：内置 PromptGuard 检测关键词、正则、相似度等注入手段。

---

## 2. 项目结构

```
DoubleHighC/
├── src/                    # 重构后的源码包
│   ├── config.py           # 集中配置管理（环境变量、路径、API 密钥池）
│   ├── database.py         # 数据库操作（建表、查询、事务管理）
│   ├── llm.py              # LLM 调用（筛选、深度分析、JSON 解析）
│   ├── scraper.py          # 网页抓取与 HTML 解析
│   ├── prompt_guard.py     # Prompt 注入检测
│   ├── notion_client.py    # Notion API 集成
│   ├── crawler.py          # 爬虫主程序入口
│   └── worker.py           # 深度分析 Worker 入口
├── tests/                  # 单元测试
│   ├── test_config.py
│   ├── test_llm.py
│   ├── test_prompt_guard.py
│   └── test_scraper.py
├── prompts/                # 提示词模板目录
│   ├── screening.md        # 筛选提示词
│   └── analysis.md         # 深度分析提示词
├── requirements.txt        # Python 依赖
├── example.env             # 环境变量模板
└── README.md
```

---

## 3. 环境要求

- Python 3.8+
- PostgreSQL
- Redis（可选）
- Google Generative AI API 密钥
- Notion Integration Token（可选）

---

## 4. 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 复制并编辑环境配置
cp example.env .env
# 编辑 .env 填入你的配置

# 运行爬虫（抓取并筛选文章）
python -m src.crawler

# 运行 Worker（深度分析已筛选文章）
python -m src.worker

# 运行测试
python -m pytest tests/ -v
```

---

## 5. 环境变量说明

| 变量名 | 必需 | 说明 |
|--------|------|------|
| `DAILY_DATABASE` | 是 | PostgreSQL 数据库名 |
| `DAILY_USER` | 是 | PostgreSQL 用户名 |
| `DAILY_PASSWORD` | 是 | PostgreSQL 密码 |
| `DAILY_HOST` | 是 | PostgreSQL 主机地址 |
| `DAILY_PORT` | 否 | PostgreSQL 端口（默认 5432） |
| `Y*` | 是 | Google Gemini API 密钥（Y 开头的环境变量，支持多个） |
| `CRAWLER_BASE_URL` | 是 | 爬虫目标网站 URL |
| `NOTION_API_KEY` | 否 | Notion Integration Token |
| `NOTION_PAGE_ID` | 否 | Notion 目标页面 ID |
| `PROMPT_DIR` | 否 | 提示词文件目录（默认 `prompts/`） |
| `SCREENING_MODEL` | 否 | 筛选模型名称（默认 `gemini-2.5-flash-lite`） |
| `ANALYSIS_MODEL` | 否 | 分析模型名称（默认 `gemini-2.5-pro`） |

---

## 6. 联系作者

