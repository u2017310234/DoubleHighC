"""Article crawler — fetches articles and screens them with LLM.

Usage:
    python -m src.crawler
"""

import time
import logging
import concurrent.futures

from src.config import (
    setup_logging,
    get_db_config,
    get_redis_client,
    APIKeyPool,
    SCREENING_MODEL,
    SCREENING_PROMPT_FILE,
    CRAWLER_BASE_URL,
    CRAWLER_MAX_PAGES,
    CRAWLER_CHUNK_SIZE,
    CRAWLER_MAX_WORKERS,
    CRAWLER_RATE_LIMIT_SECONDS,
    load_prompt,
)
from src.database import (
    get_connection,
    create_articles_table,
    create_screening_results_table,
    batch_insert_screening_results,
    insert_relevant_articles,
    push_tasks_to_redis,
)
from src.scraper import fetch_page, parse_articles_from_html
from src.llm import screen_article

logger = logging.getLogger(__name__)


def screen_single_article(article, api_key_pool, model_name, system_prompt):
    """Screen a single article and return a result dict.

    Args:
        article: Dict with 'title' and 'link'.
        api_key_pool: APIKeyPool instance.
        model_name: LLM model name.
        system_prompt: System prompt text.

    Returns:
        dict: Screening result with article metadata.
    """
    result = screen_article(
        title=article["title"],
        api_key_pool=api_key_pool,
        model_name=model_name,
        system_prompt=system_prompt,
    )
    return {
        "article_url": article["link"],
        "article_title": article["title"],
        "relevance_score": result["relevance_score"],
        "is_relevant": result["is_relevant"],
        "model_used": result["model_used"],
        "screening_duration_ms": result["screening_duration_ms"],
        "processed_at": time.strftime("%Y-%m-%d %H:%M:%S %Z", time.gmtime()),
    }


def filter_articles_parallel(articles, api_key_pool, model_name, system_prompt):
    """Screen articles in parallel using a thread pool.

    Args:
        articles: List of article dicts.
        api_key_pool: APIKeyPool instance.
        model_name: LLM model name.
        system_prompt: System prompt text.

    Returns:
        list: List of screening result dicts.
    """
    all_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=CRAWLER_MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                screen_single_article, article, api_key_pool, model_name, system_prompt
            ): article
            for article in articles
        }
        logger.info("已提交 %d 篇文章进行并发筛选...", len(articles))

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                logger.error("处理筛选结果时出错: %s", e)
            time.sleep(CRAWLER_RATE_LIMIT_SECONDS)

    logger.info("筛选完成，共处理 %d 篇文章。", len(all_results))
    return all_results


def main():
    """Main crawler entry point."""
    setup_logging()

    db_config = get_db_config()
    if not db_config:
        logger.error("由于缺少数据库配置，程序未运行。")
        return

    if not CRAWLER_BASE_URL:
        logger.error("由于缺少CRAWLER_BASE_URL配置，程序未运行。")
        return

    # Load API key pool
    api_key_pool = APIKeyPool()

    # Load system prompt
    try:
        system_prompt = load_prompt(SCREENING_PROMPT_FILE)
    except FileNotFoundError:
        logger.warning("筛选提示词文件不存在，将使用空提示词。")
        system_prompt = ""

    # Optional Redis
    redis_client = get_redis_client()

    # Connect to database and ensure tables exist
    pg_conn = get_connection(db_config)
    try:
        create_articles_table(pg_conn)
        create_screening_results_table(pg_conn)

        # Scrape articles from all pages
        all_articles = []
        for page_num in range(1, CRAWLER_MAX_PAGES + 1):
            page_url = f"{CRAWLER_BASE_URL}&p={page_num}"
            html = fetch_page(page_url)
            page_articles = parse_articles_from_html(html, page_url)
            all_articles.extend(page_articles)
            logger.info(
                "已抓取第 %d 页，累积文章数: %d", page_num, len(all_articles)
            )
            time.sleep(1)  # Polite crawling delay

        # Process in chunks
        total_chunks = (len(all_articles) + CRAWLER_CHUNK_SIZE - 1) // CRAWLER_CHUNK_SIZE
        for i in range(total_chunks):
            start = i * CRAWLER_CHUNK_SIZE
            chunk = all_articles[start : start + CRAWLER_CHUNK_SIZE]

            if not chunk:
                continue

            logger.info(
                "===== 开始处理批次 %d/%d (共 %d 篇文章) =====",
                i + 1,
                total_chunks,
                len(chunk),
            )

            # Screen articles
            results = filter_articles_parallel(
                chunk, api_key_pool, SCREENING_MODEL, system_prompt
            )

            # Save screening results
            batch_insert_screening_results(pg_conn, results)

            # Filter relevant articles
            relevant = [r for r in results if r["is_relevant"]]
            if relevant:
                new_ids = insert_relevant_articles(pg_conn, relevant)
                if new_ids and redis_client:
                    push_tasks_to_redis(redis_client, "article_tasks", new_ids)

            logger.info("===== 批次 %d/%d 处理完成 =====", i + 1, total_chunks)

    finally:
        if pg_conn:
            pg_conn.close()
            logger.info("PostgreSQL连接已关闭。")


if __name__ == "__main__":
    main()
