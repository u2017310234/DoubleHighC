!pip install google-generativeai
import os


import pathlib
import textwrap
import PIL.Image

import google.generativeai as genai
import typing_extensions as typing

from IPython.display import display
from IPython.display import Markdown
from IPython.display import Image

def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

  # Or use `os.getenv('GOOGLE_API_KEY')` to fetch an environment variable.
GOOGLE_API_KEY=os.getenv('Y')

genai.configure(api_key=GOOGLE_API_KEY)

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

NOTION_API_KEY=os.getenv('NOTION')
notion_page_id=os.getenv('NOTION_PAGE_ID')
import datetime

def notion_json(content,daydelta,pageid,titl):
    #today = datetime.now()
    #yesterday = today - timedelta(days=daydelta)
    last_date = datetime.now().strftime("%Y%m%d")
    content_json = {
        "parent": {"type": "page_id","page_id": pageid},
        "properties": {
            "title": {
                "title":[
                {
                    "text": {
                        "content": str(last_date) + titl
                    }
                }
            ]
        }
        },
        "children": [
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                "rich_text": [
                    {
                        "type": "text",
                        "text": {
                                "content": content
  }
                    }
                ]
            }
        }
    ]
}
    headers = {
    'Authorization': f'Bearer {NOTION_API_KEY}',
    'Content-Type': 'application/json',
    'Notion-Version': '2022-06-28'  # 使用最新的 API 版本
        }
    result=requests.post("https://api.notion.com/v1/pages", headers=headers, data=json.dumps(content_json))
    return result


import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class PromptGuard:
    def __init__(self, keywords=None, similarity_threshold=0.8, length_ratio_threshold=10):
        """
        初始化提示词护栏。

        Args:
            keywords (list): 需要检测的关键词列表，默认包括 'prompt', 'instruction', '提示词'。
            similarity_threshold (float): 内容相似度阈值，超过此值则视为高相似度，默认0.8。
            length_ratio_threshold (float): 长度比例阈值，输出长度与输入长度的比例超过此值则视为可疑，默认0.8。
        """
        if keywords is None:
            self.keywords = ['prompt', 'instruction', '提示词', '指令','##########','#OBJECTIVE#']
        else:
            self.keywords = keywords
        self.similarity_threshold = similarity_threshold
        self.length_ratio_threshold = length_ratio_threshold
        self.vectorizer = TfidfVectorizer()

    def detect_keywords(self, text):
        """
        检测文本中是否包含关键词。

        Args:
            text (str): 要检测的文本。

        Returns:
             bool: 如果检测到关键词，返回True；否则返回False。
        """
        text_lower = text.lower()
        for keyword in self.keywords:
            if keyword.lower() in text_lower:
                return True
        return False

    def detect_regex(self, text, pattern=r"(prompt|instruction|提示词|指令)\s*[:=]?\s*[\"']?(.+?)[\"']?"):
        """
        使用正则表达式检测文本中是否包含提示词模式。

        Args:
            text (str): 要检测的文本。
            pattern (str): 正则表达式模式。

        Returns:
            bool: 如果检测到匹配，返回True；否则返回False。
        """
        if re.search(pattern, text, re.IGNORECASE):
            return True
        return False

    def detect_length_ratio(self, prompt, output):
        """
        检测输出长度与提示词长度的比例。

        Args:
            prompt (str): 输入的提示词。
            output (str): LLM 的输出。

        Returns:
            bool: 如果长度比例超过阈值，返回True；否则返回False。
        """
        prompt_len = len(prompt)
        output_len = len(output)
        if prompt_len == 0:
            return False  # 避免除零错误
        length_ratio = output_len / prompt_len
        return length_ratio > self.length_ratio_threshold

    def detect_similarity(self, prompt, output):
        """
        检测输出与提示词的相似度。

        Args:
            prompt (str): 输入的提示词。
            output (str): LLM 的输出。

        Returns:
             bool: 如果相似度超过阈值，返回True；否则返回False。
        """
        if not prompt or not output:
          return False

        vectors = self.vectorizer.fit_transform([prompt, output])
        similarity = cosine_similarity(vectors)[0][1]
        return similarity > self.similarity_threshold

    def check(self, prompt, output):
        """
        执行所有检查，判断输出是否可能包含提示词。

        Args:
            prompt (str): 输入的提示词。
            output (str): LLM 的输出。

        Returns:
            bool: 如果检测到任何违反规则的情况，返回True；否则返回False。
        """
        if self.detect_keywords(output):
            print("检测到关键词")
            return True
        if self.detect_regex(output):
            print("检测到正则表达式模式")
            return True
        if self.detect_length_ratio(prompt, output):
            print("检测到长度比例过高")
            return True
        if self.detect_similarity(prompt, output):
            print("检测到内容相似度过高")
            return True
        return False

# worker.py
import os
import psycopg2
import time
import sys
import json
import random
import threading
from itertools import cycle
from datetime import datetime, timezone
import re

# --- 必需的库导入 ---
try:
    import requests
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    import google.generativeai as genai
    from google.api_core import exceptions as google_exceptions
except ImportError as e:
    print(f"FATAL: 缺少必要的库: {e}。请运行 'pip install requests tenacity google-generativeai'。", file=sys.stderr)
    sys.exit(1)

# --- 配置 ---

# 1. 数据库配置 (从环境变量读取)
try:
    DB_CONFIG = {
        "dbname": os.environ["DAILY_DATABASE"],
        "user": os.environ["DAILY_USER"],
        "password": os.environ["DAILY_PASSWORD"],
        "host": os.environ["DAILY_HOST"],
        "port": os.environ.get("DAILY_PORT", "5432"),
        #"sslmode": "require"
    }
except KeyError as e:
    print(f"FATAL: 缺少数据库环境变量: {e}。请确保 PG_HOST, PG_DATABASE, PG_USER, PG_PASSWORD 已设置。", file=sys.stderr)
    sys.exit(1)

# 2. LLM配置 (API密钥池)
API_KEY_PREFIX = "Y" # 您设置的环境变量前缀
api_key_names = [key for key in os.environ.keys() if key.startswith(API_KEY_PREFIX)]

if not api_key_names:
    raise ValueError(f"未在环境变量中找到任何以 '{API_KEY_PREFIX}' 开头的API密钥。")

print(f"INFO: 成功加载 {len(api_key_names)} 个API密钥。")
random.shuffle(api_key_names)
api_key_pool = cycle(api_key_names)
key_pool_lock = threading.Lock() # 线程锁，保证并发安全

# 3. LLM模型和安全设置
SCREENING_MODEL = "gemini-2.5-pro" # 您可以根据需要更改模型
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]




def scrape_full_content(url):
    """
    使用 requests 抓取文章全文。
    注意：这是一个简化实现。对于JS渲染的复杂页面，这里应替换为Playwright的实现。
    """
    print(f"  > [抓取] 正在抓取: {url[:70]}...")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status() # 如果状态码不是2xx，则抛出异常
        # TODO: 在这里可以使用Readability.js等库来提取正文，而不是返回整个HTML
        print(f"  > [抓取] 成功。")
        return response.text
    except requests.RequestException as e:
        print(f"ERROR: 抓取URL时失败: {e}", file=sys.stderr)
        raise  # 向上抛出异常，让主循环捕获并标记任务为 'error'


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((
        google_exceptions.ResourceExhausted,
        google_exceptions.ServiceUnavailable,
        google_exceptions.DeadlineExceeded,
        google_exceptions.Aborted,
    ))
)
def run_deep_analysis_pipeline(content):
    """
    运行多LLM深度分析流水线，并返回结构化的JSON结果。
    """
    with key_pool_lock:
        current_api_key_name = next(api_key_pool)
    api_key = os.environ[current_api_key_name]
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(SCREENING_MODEL)
    
    with open('/work/123/abstract.md', "r", encoding="utf-8") as md_file: #///////////////注意这里是硬编码////////////////
        a=md_file.read()

    # 组合提示词
    prompt = a + "\n\n---\n\n" + content

    try:
        response = model.generate_content(prompt, safety_settings=safety_settings)

        # 检查响应是否为空或被阻止
        if not response.text:
            feedback = response.prompt_feedback if response.prompt_feedback else "No feedback available."
            raise ValueError(f"LLM API返回空响应，可能被安全策略阻止。反馈: {feedback}")

        return response.text

    except (google_exceptions.ResourceExhausted, google_exceptions.ServiceUnavailable, google_exceptions.DeadlineExceeded, google_exceptions.Aborted) as e:
        print(f"WARN: 遇到可重试的API错误 (使用密钥 {current_api_key_name}, 类型 {type(e).__name__})，将由tenacity重试...", file=sys.stderr)
        raise e # 重新抛出，让tenacity接管
    except Exception as e:
        print(f"ERROR: LLM分析或JSON解析失败: {e}", file=sys.stderr)
        raise # 向上抛出，让主循环捕获


# --- 核心任务队列函数 (与之前相同) ---
def get_and_lock_task(pg_conn):
    # ... (此函数代码与之前版本完全相同，此处省略以保持简洁)
    with pg_conn.cursor() as cur:
        try:
            cur.execute("""
                SELECT id, url ,title FROM articles
                WHERE status = 'pending'
                ORDER BY created_at
                LIMIT 1
                FOR UPDATE SKIP LOCKED;
            """)
            task_record = cur.fetchone()

            if task_record:
                task_id, task_url , task_title= task_record
                cur.execute(
                    "UPDATE articles SET status = 'processing', updated_at = NOW() WHERE id = %s",
                    (task_id,)
                )
                pg_conn.commit()
                print(f"\nINFO: Worker [{os.getpid()}] 获取并锁定任务 {task_id}")
                return {"id": task_id, "url": task_url, "title":task_title}
            else:
                pg_conn.commit()
                return None
        except psycopg2.Error as e:
            pg_conn.rollback()
            print(f"ERROR: 获取任务时数据库出错: {e}", file=sys.stderr)
            return None






def create_results_results_table(conn):
    """创建 screening_results 表。"""
    query = """
    CREATE TABLE IF NOT EXISTS analysis_results (
    id SERIAL PRIMARY KEY,
    article_id INTEGER UNIQUE NOT NULL,
    core_relevance INTEGER,
    novelty_of_insight INTEGER,
    decision_value INTEGER,
    overall_priority_score INTEGER,
    recommendation TEXT,
    associated_domains TEXT[], -- 使用PostgreSQL强大的数组类型
    summary TEXT,
    full_markdown_report TEXT, -- 存储LLM生成的完整Markdown报告
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- 设置外键，确保数据完整性，并能级联删除
    CONSTRAINT fk_article
        FOREIGN KEY(article_id) 
        REFERENCES articles(id)
        ON DELETE CASCADE
);

-- 为外键创建索引以提高查询性能
CREATE INDEX IF NOT EXISTS idx_analysis_results_article_id ON analysis_results(article_id);
    """
    with conn.cursor() as cur:
        cur.execute(query)
        conn.commit()
    print("INFO: 'screening_results' 表已成功创建或已存在。")

# --- 主循环 ---
def main_loop():
    """主循环，持续获取并处理任务。"""
    print("INFO: Worker进程已启动，使用PostgreSQL作为任务队列。")
    pg_conn = psycopg2.connect(**DB_CONFIG)
    
    with pg_conn.cursor() as cur:
        cur.execute("SET TIME ZONE 'Asia/Shanghai';")
        print("INFO: 数据库会话时区已设置为 'Asia/Shanghai'。")
        create_results_results_table(pg_conn)
        
    while True:
        task = None
        try:
            task = get_and_lock_task(pg_conn)
            print(task)
            if task:
                # 完整的处理流水线
                full_content = scrape_full_content(task['url'])
                analysis_result = run_deep_analysis_pipeline(full_content)
                analysis_results = analysis_result
                
                #上传notion
                notion_json(analysis_results,0,notion_page_id,task['title'])

                
                # JSON解析
                cleaned_text = analysis_results.strip()

                match = re.search(r"```json\s*(\{.*?\})\s*```|(\{.*?\})", cleaned_text, re.DOTALL)
    
                if match.group(1):
                    # group(1) 对应第一个捕获组 ```json(...)``` 中的 (...)
                    # group(2) 对应第二个捕获组 (...)
                    # 哪个匹配到了就用哪个
                    cleaned_te=match.group(1)
                else:
                    match.group(2)
                #if cleaned_text.startswith("```json"):
                #    cleaned_text = cleaned_text[7:]
                #if cleaned_text.endswith("```"):
                #    cleaned_text = cleaned_text[:-3]

                c_text=json.loads(cleaned_te)

                with pg_conn.cursor() as cur:
                    # a. 插入或更新 analysis_results 表
                    # 使用 ON CONFLICT...DO UPDATE 保证任务可重入性
                    # 如果worker中途失败重跑，这里会更新而不是报错
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
                            task['id'],
                            c_text.get('core_relevance'),
                            c_text.get('novelty_of_insight'),
                            c_text.get('decision_value'),
                            c_text.get('overall_priority_score'),
                            c_text.get('recommendation'),
                            c_text.get('associated_domains'),
                            c_text.get('summary'),
                            cleaned_text
                        )
                    )
                    
                    # b. 更新主表 'articles' 的状态
                    cur.execute(
                        "UPDATE articles SET status = 'done', updated_at = NOW() WHERE id = %s",
                        (task['id'],)
                    )
                    
                    # 只有当以上两个操作都无误时，才提交事务
                    pg_conn.commit()
                    
                print(f"INFO: 任务 {task['id']} 处理完成，分析结果已存入 'analysis_results' 表。")

        except Exception as e:


            print(f"ERROR: 处理任务 {task['id'] if task else 'N/A'} 时发生严重错误: {e}", file=sys.stderr)
            if task and pg_conn:
                with pg_conn.cursor() as cur:
                    cur.execute("UPDATE articles SET status = 'error', updated_at = NOW() WHERE id = %s", (task['id'],))
                    pg_conn.commit()
                print(f"INFO: 任务 {task['id']} 已被标记为 'error' 状态。")
            time.sleep(5)

if __name__ == "__main__":
        main_loop()






