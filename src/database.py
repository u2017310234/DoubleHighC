"""Database operations for article management.

Provides functions for creating tables and managing articles/screening results
in PostgreSQL.
"""

import logging

import psycopg2
from psycopg2.extras import execute_batch

logger = logging.getLogger(__name__)


def get_connection(db_config):
    """Create and return a PostgreSQL connection.

    Args:
        db_config: Database configuration dict.

    Returns:
        psycopg2.connection: A database connection.

    Raises:
        psycopg2.Error: If connection fails.
    """
    conn = psycopg2.connect(**db_config)
    logger.info("成功连接到PostgreSQL数据库。")
    return conn


def create_articles_table(conn):
    """Create the articles table if it does not exist.

    Args:
        conn: PostgreSQL connection.
    """
    query = """
    CREATE TABLE IF NOT EXISTS articles (
        id SERIAL PRIMARY KEY,
        url TEXT UNIQUE NOT NULL,
        title TEXT,
        status VARCHAR(20) NOT NULL DEFAULT 'pending',
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            conn.commit()
        logger.info("'articles' 表已成功创建或已存在。")
    except psycopg2.Error as e:
        logger.error("创建 articles 表失败: %s", e)
        conn.rollback()
        raise


def create_screening_results_table(conn):
    """Create the screening_results table if it does not exist.

    Args:
        conn: PostgreSQL connection.
    """
    query = """
    CREATE TABLE IF NOT EXISTS screening_results (
        id SERIAL PRIMARY KEY,
        article_url TEXT UNIQUE NOT NULL,
        article_title TEXT,
        relevance_score FLOAT,
        is_relevant BOOLEAN,
        model_used VARCHAR(100),
        screening_duration_ms INTEGER,
        scraped_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        processed_at TIMESTAMPTZ
    );
    """
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            conn.commit()
        logger.info("'screening_results' 表已成功创建或已存在。")
    except psycopg2.Error as e:
        logger.error("创建 screening_results 表失败: %s", e)
        conn.rollback()
        raise


def create_analysis_results_table(conn):
    """Create the analysis_results table if it does not exist.

    Args:
        conn: PostgreSQL connection.
    """
    query = """
    CREATE TABLE IF NOT EXISTS analysis_results (
        id SERIAL PRIMARY KEY,
        article_id INTEGER UNIQUE NOT NULL,
        core_relevance INTEGER,
        novelty_of_insight INTEGER,
        decision_value INTEGER,
        overall_priority_score INTEGER,
        recommendation TEXT,
        associated_domains TEXT[],
        summary TEXT,
        full_markdown_report TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        CONSTRAINT fk_article
            FOREIGN KEY(article_id)
            REFERENCES articles(id)
            ON DELETE CASCADE
    );
    CREATE INDEX IF NOT EXISTS idx_analysis_results_article_id
        ON analysis_results(article_id);
    """
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            conn.commit()
        logger.info("'analysis_results' 表已成功创建或已存在。")
    except psycopg2.Error as e:
        logger.error("创建 analysis_results 表失败: %s", e)
        conn.rollback()
        raise


def batch_insert_screening_results(conn, results):
    """Insert screening results in batch, ignoring duplicates.

    Args:
        conn: PostgreSQL connection.
        results: List of result dicts with keys: article_url, article_title,
            relevance_score, is_relevant, model_used, screening_duration_ms,
            processed_at.
    """
    if not results:
        return
    query = """
    INSERT INTO screening_results (
        article_url, article_title, relevance_score, is_relevant,
        model_used, screening_duration_ms, processed_at
    ) VALUES (
        %(article_url)s, %(article_title)s, %(relevance_score)s, %(is_relevant)s,
        %(model_used)s, %(screening_duration_ms)s, %(processed_at)s
    ) ON CONFLICT (article_url) DO NOTHING;
    """
    try:
        with conn.cursor() as cur:
            execute_batch(cur, query, results)
            conn.commit()
        logger.info("已将 %d 条筛选结果写入 'screening_results' 表。", len(results))
    except psycopg2.Error as e:
        logger.error("批量插入筛选结果失败: %s", e)
        conn.rollback()
        raise


def insert_relevant_articles(conn, relevant_results):
    """Insert relevant articles into the main articles table.

    Args:
        conn: PostgreSQL connection.
        relevant_results: List of dicts with 'article_url' and 'article_title'.

    Returns:
        list: List of newly inserted article IDs.
    """
    if not relevant_results:
        return []

    query = (
        "INSERT INTO articles (url, title) VALUES (%s, %s) "
        "ON CONFLICT (url) DO NOTHING RETURNING id;"
    )
    new_ids = []
    try:
        with conn.cursor() as cur:
            for result in relevant_results:
                cur.execute(query, (result["article_url"], result["article_title"]))
                res = cur.fetchone()
                if res:
                    new_ids.append(res[0])
            conn.commit()
        logger.info("已将 %d 条通过筛选的文章存入主 'articles' 表。", len(new_ids))
    except psycopg2.Error as e:
        logger.error("插入相关文章失败: %s", e)
        conn.rollback()
        raise
    return new_ids


def get_and_lock_task(conn):
    """Fetch and lock the next pending task from the articles table.

    Uses SELECT ... FOR UPDATE SKIP LOCKED for safe concurrent access.

    Args:
        conn: PostgreSQL connection.

    Returns:
        dict or None: Task dict with 'id', 'url', 'title', or None if no tasks.
    """
    with conn.cursor() as cur:
        try:
            cur.execute("""
                SELECT id, url, title FROM articles
                WHERE status = 'pending'
                ORDER BY created_at
                LIMIT 1
                FOR UPDATE SKIP LOCKED;
            """)
            record = cur.fetchone()
            if record:
                task_id, task_url, task_title = record
                cur.execute(
                    "UPDATE articles SET status = 'processing', updated_at = NOW() "
                    "WHERE id = %s",
                    (task_id,),
                )
                conn.commit()
                logger.info("获取并锁定任务 %d", task_id)
                return {"id": task_id, "url": task_url, "title": task_title}
            conn.commit()
            return None
        except psycopg2.Error as e:
            conn.rollback()
            logger.error("获取任务时数据库出错: %s", e)
            return None


def mark_task_done(conn, task_id, analysis_data, raw_report):
    """Save analysis results and mark task as done in a single transaction.

    Args:
        conn: PostgreSQL connection.
        task_id: The article/task ID.
        analysis_data: Dict with analysis fields.
        raw_report: Full markdown report text.
    """
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO analysis_results (
                    article_id, core_relevance, novelty_of_insight, decision_value,
                    overall_priority_score, recommendation, associated_domains, summary,
                    full_markdown_report
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (article_id) DO UPDATE SET
                    core_relevance = EXCLUDED.core_relevance,
                    novelty_of_insight = EXCLUDED.novelty_of_insight,
                    decision_value = EXCLUDED.decision_value,
                    overall_priority_score = EXCLUDED.overall_priority_score,
                    recommendation = EXCLUDED.recommendation,
                    associated_domains = EXCLUDED.associated_domains,
                    summary = EXCLUDED.summary,
                    full_markdown_report = EXCLUDED.full_markdown_report;
                """,
                (
                    task_id,
                    analysis_data.get("core_relevance"),
                    analysis_data.get("novelty_of_insight"),
                    analysis_data.get("decision_value"),
                    analysis_data.get("overall_priority_score"),
                    analysis_data.get("recommendation"),
                    analysis_data.get("associated_domains"),
                    analysis_data.get("summary"),
                    raw_report,
                ),
            )
            cur.execute(
                "UPDATE articles SET status = 'done', updated_at = NOW() WHERE id = %s",
                (task_id,),
            )
            conn.commit()
        logger.info("任务 %d 处理完成，分析结果已存入数据库。", task_id)
    except psycopg2.Error as e:
        logger.error("保存分析结果失败: %s", e)
        conn.rollback()
        raise


def mark_task_error(conn, task_id):
    """Mark a task as errored.

    Args:
        conn: PostgreSQL connection.
        task_id: The article/task ID.
    """
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE articles SET status = 'error', updated_at = NOW() WHERE id = %s",
                (task_id,),
            )
            conn.commit()
        logger.info("任务 %d 已被标记为 'error' 状态。", task_id)
    except psycopg2.Error as e:
        logger.error("标记任务错误状态失败: %s", e)
        conn.rollback()


def push_tasks_to_redis(redis_conn, queue_name, article_ids):
    """Push article IDs to a Redis queue.

    Args:
        redis_conn: Redis client connection.
        queue_name: Name of the Redis list/queue.
        article_ids: List of article IDs to push.
    """
    if not article_ids or not redis_conn:
        return
    redis_conn.lpush(queue_name, *article_ids)
    logger.info("成功将 %d 个任务ID推送到Redis队列 '%s'。", len(article_ids), queue_name)
