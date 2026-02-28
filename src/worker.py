"""Article analysis worker — deep analysis pipeline.

Fetches pending articles, runs deep LLM analysis, stores results,
and optionally syncs to Notion.

Usage:
    python -m src.worker
"""

import time
import logging

from src.config import (
    setup_logging,
    get_db_config,
    APIKeyPool,
    ANALYSIS_MODEL,
    ANALYSIS_PROMPT_FILE,
    NOTION_API_KEY,
    load_prompt,
)
from src.database import (
    get_connection,
    create_analysis_results_table,
    get_and_lock_task,
    mark_task_done,
    mark_task_error,
)
from src.scraper import scrape_full_content
from src.llm import run_deep_analysis, parse_analysis_json
from src.notion_client import create_notion_page

logger = logging.getLogger(__name__)

# Polling interval when no tasks are available (seconds)
IDLE_POLL_INTERVAL = 10
# Delay after an error before retrying (seconds)
ERROR_RETRY_DELAY = 5


def process_task(task, api_key_pool, model_name, system_prompt):
    """Process a single task through the full analysis pipeline.

    Args:
        task: Task dict with 'id', 'url', 'title'.
        api_key_pool: APIKeyPool instance.
        model_name: LLM model name.
        system_prompt: System prompt text.

    Returns:
        tuple: (analysis_data dict, raw_report str)

    Raises:
        Exception: If any step in the pipeline fails.
    """
    # Step 1: Scrape full content
    full_content = scrape_full_content(task["url"])

    # Step 2: Run deep analysis
    raw_report = run_deep_analysis(
        content=full_content,
        api_key_pool=api_key_pool,
        model_name=model_name,
        system_prompt=system_prompt,
    )

    # Step 3: Optionally push to Notion
    if NOTION_API_KEY:
        try:
            create_notion_page(raw_report, title_suffix=task["title"])
        except Exception as e:
            logger.warning("推送到Notion失败 (非致命): %s", e)

    # Step 4: Parse structured analysis data
    analysis_data = parse_analysis_json(raw_report)

    return analysis_data, raw_report


def main_loop():
    """Main worker loop — continuously processes pending tasks."""
    setup_logging()

    db_config = get_db_config()
    if not db_config:
        logger.error("由于缺少数据库配置，Worker未启动。")
        return

    api_key_pool = APIKeyPool()

    # Load system prompt
    try:
        system_prompt = load_prompt(ANALYSIS_PROMPT_FILE)
    except FileNotFoundError:
        logger.warning("分析提示词文件不存在，将使用空提示词。")
        system_prompt = ""

    pg_conn = get_connection(db_config)

    # Set timezone and ensure tables exist
    with pg_conn.cursor() as cur:
        cur.execute("SET TIME ZONE 'Asia/Shanghai';")
    create_analysis_results_table(pg_conn)

    logger.info("Worker进程已启动，使用PostgreSQL作为任务队列。")

    while True:
        task = None
        try:
            task = get_and_lock_task(pg_conn)

            if task is None:
                time.sleep(IDLE_POLL_INTERVAL)
                continue

            logger.info("开始处理任务 %d: %s", task["id"], task["title"])

            analysis_data, raw_report = process_task(
                task, api_key_pool, ANALYSIS_MODEL, system_prompt
            )

            mark_task_done(pg_conn, task["id"], analysis_data, raw_report)

        except Exception as e:
            task_id = task["id"] if task else "N/A"
            logger.error("处理任务 %s 时发生错误: %s", task_id, e)
            if task:
                mark_task_error(pg_conn, task["id"])
            time.sleep(ERROR_RETRY_DELAY)


if __name__ == "__main__":
    main_loop()
