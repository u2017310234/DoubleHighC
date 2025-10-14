!pip install bs4
# Cell 1: 导入与配置
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import psycopg2
#import redis
import time
import concurrent.futures

!pip install google-generativeai
import json
import random
from itertools import cycle
import google.generativeai as genai
import google.api_core.exceptions as google_exceptions
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type 
import threading
import tenacity

# --- PG数据库配置 (从Deepnote环境变量读取) ---
try:
    DB_CONFIG = {
        "dbname": os.environ["DAILY_DATABASE"],
        "user": os.environ["DAILY_USER"],
        "password": os.environ["DAILY_PASSWORD"],
        "host": os.environ["DAILY_HOST"],
        "port": os.environ.get("DAILY_PORT", "5432"),
        #"sslmode": "require"
    }
    
except KeyError:
    print("FATAL: 缺少PG数据库环境变量。")
    DB_CONFIG = None

# --- Redis配置 (从Deepnote环境变量读取) ---
try:
    redis_client = redis.Redis(
        host=os.environ["YOUR_REDIS_INTEGRATION_HOST"],
        port=os.environ.get("YOUR_REDIS_INTEGRATION_PORT", 6380),
        password=os.environ["YOUR_REDIS_INTEGRATION_PASSWORD"],
        ssl=True,
        db=0,
        decode_responses=True
    )
    redis_client.ping()
    print("INFO: 成功连接到Redis。")
except Exception as e:
    print(f"FATAL: 连接到Redis失败: {e}")
    redis_client = None

# --- LLM配置 ---
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

SCREENING_MODEL = "gemini-2.5-flash-lite"
# 1. 从环境变量中加载所有  API 密钥
#    请将您的密钥设置为 _API_KEY_1, _API_KEY_2, ...
api_key_names = [
    key for key in os.environ.keys() if key.startswith("Y")
]

if not api_key_names:
    raise ValueError(
        "未在环境变量中找到任何  API 密钥。请设置形如 '_API_KEY_n' 的变量。"
    )

# 2. 为了避免每次都从第一个密钥开始，可以随机打乱列表
random.shuffle(api_key_names)
api_key_pool = cycle(api_key_names)
key_pool_lock = threading.Lock()


1# --- 爬虫与URL处理函数 ---
def resolve_url(base, url):
    return urljoin(base, url)

def fetch_and_scrape(base_url = "YOUR_URL"):
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    print(f"INFO: 正在从 {base_url} 获取页面内容...")
    try:
        response = requests.get(base_url, headers=headers, timeout=15)
        response.raise_for_status()
        response.encoding = 'utf-8'
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"ERROR: 访问 失败: {e}")
        return None

def parse_html_for_articles(html_content,base_url):
    if not html_content: return []
    soup = BeautifulSoup(html_content, 'html.parser')
    link_elements = soup.select('div.cc-cd-cb-l > a')
    articles = []

    for a_tag in link_elements:
        title_span = a_tag.select_one('span.t')
        raw_link = a_tag.get('href')
        if title_span and raw_link:
            title = title_span.get_text(strip=True)
            final_link = resolve_url(base_url, raw_link)
            if title:
                articles.append({'title': title, 'link': final_link})
    print(f"INFO: 爬虫解析完成，共发现 {len(articles)} 篇文章。")
    return articles



# --- LLM粗筛函数 (返回更丰富的信息) ---
@retry(
    # 设置重试策略：等待时间为指数级增长，初始2秒，最多等待10秒
    wait=wait_exponential(multiplier=1, min=2, max=10),
    # 设置停止条件：最多重试3次
    stop=stop_after_attempt(3),
    # 设置重试条件：只在遇到特定的、可恢复的API错误时才重试
    retry=retry_if_exception_type((
        google_exceptions.ResourceExhausted, # 速率限制错误 (429)
        google_exceptions.ServiceUnavailable, # 服务器不可用错误 (503)
        google_exceptions.DeadlineExceeded,   # 超时错误
        google_exceptions.Aborted,            # 请求中止
    ))
)
def screen_article_relevance(article):
    """
    调用LLM对单篇文章进行粗筛，返回一个包含详细结果的字典。
    """
    start_time = time.time()
    title = article['title']
    
    # 默认返回失败结果
    result_data = {
        'is_relevant': False,
        'relevance_score': 0.0,
        'model_used': SCREENING_MODEL
    }

    try:
        # 【线程安全地获取下一个API Key】
        with key_pool_lock:
            current_api_key_name = next(api_key_pool)
        
        api_key = os.environ[current_api_key_name]
        
        # 为本次调用配置genai客户端
        genai.configure(api_key=api_key)
        
        # 创建模型实例
        model = genai.GenerativeModel(SCREENING_MODEL)
        
        with open('/work/123/tick.md', "r", encoding="utf-8") as md_file:   #//////////////////注意这里是硬编码/////////////
            b=md_file.read()

        # 更新Prompt以适应Gemini
        prompt = b+f"""
        Respond ONLY with a valid JSON object containing three keys:
        1. "is_relevant": a boolean (true or false).
        2. "relevance_score": a float from 0.0 to 1.0.
        3. "reasoning": a brief string explanation.

        Title: "{title}"
        """

        # 调用API，并设置JSON响应格式
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"},
            safety_settings=safety_settings
        )
        
        result = json.loads(response.text)
        print(result)
        result_data['is_relevant'] = result.get('is_relevant', False)
        result_data['relevance_score'] = result.get('relevance_score', 0.0)

    except Exception as e:
        print(f"WARN: LLM筛选标题 '{title}' 时出错: {e}")
        # 即使出错，也保持函数能正常返回，不中断整个并发流程
        
    end_time = time.time()
    duration_ms = int((end_time - start_time) * 1000)
    
    # 组合最终返回的完整字典
    return {
        'article_url': article['link'],
        'article_title': title,
        'relevance_score': result_data['relevance_score'],
        'is_relevant': result_data['is_relevant'],
        'model_used': result_data['model_used'],
        'screening_duration_ms': duration_ms,
        'processed_at': time.strftime('%Y-%m-%d %H:%M:%S %Z', time.gmtime()) # 使用UTC时间
    }

def filter_articles_in_parallel(articles):
    """并发执行粗筛，并返回所有文章的详细筛选结果。"""
    all_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future_to_article = {executor.submit(screen_article_relevance, article): article for article in articles}
        print(f"INFO: 已提交 {len(articles)} 篇文章进行并发筛选...")

        for future in concurrent.futures.as_completed(future_to_article):
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                print(f"ERROR: 处理筛选结果时出错: {e}")
            time.sleep(4.2) #
    print(f"INFO: 筛选完成，共处理 {len(all_results)} 篇文章。")
    return all_results

# --- 数据库操作函数 ---

def create_articles_table(conn):
    """在数据库中创建 articles 表（如果不存在）。"""
    create_table_query = """
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
            cur.execute(create_table_query)
            conn.commit()
            print("INFO: 'articles' 表已成功创建或已存在。")
    except psycopg2.Error as e:
        print(f"ERROR: 创建表失败: {e}")
        conn.rollback()
        raise

# 创建新表的函数
def create_screening_results_table(conn):
    """创建 screening_results 表。"""
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
    with conn.cursor() as cur:
        cur.execute(query)
        conn.commit()
    print("INFO: 'screening_results' 表已成功创建或已存在。")

# 批量插入筛选结果到新表的函数
def batch_insert_screening_results(conn, results):
    """将所有筛选结果批量写入 screening_results 表，并忽略已存在的URL。"""
    if not results: return
    
    query = """
    INSERT INTO screening_results (
        article_url, article_title, relevance_score, is_relevant,
        model_used, screening_duration_ms, processed_at
    ) VALUES (
        %(article_url)s, %(article_title)s, %(relevance_score)s, %(is_relevant)s,
        %(model_used)s, %(screening_duration_ms)s, %(processed_at)s
    ) ON CONFLICT (article_url) DO NOTHING;
    """
    with conn.cursor() as cur:
        # 使用 psycopg2.extras.execute_batch 可以高效执行批量插入
        from psycopg2.extras import execute_batch
        execute_batch(cur, query, results)
        conn.commit()
    print(f"INFO: 已将 {len(results)} 条筛选结果写入 'screening_results' 表。")

# 一个独立的函数，用于将通过筛选的文章存入主表并返回ID
def insert_relevant_articles_to_main_table(conn, relevant_articles_results):
    """将通过筛选的文章存入主'articles'表，并返回新插入的ID。"""
    if not relevant_articles_results: return []
    
    query = "INSERT INTO articles (url, title) VALUES (%s, %s) ON CONFLICT (url) DO NOTHING RETURNING id;"
    new_ids = []
    with conn.cursor() as cur:
        for result in relevant_articles_results:
            cur.execute(query, (result['article_url'], result['article_title']))
            res = cur.fetchone()
            if res:
                new_ids.append(res[0])
        conn.commit()
    print(f"INFO: 已将 {len(new_ids)} 条通过筛选的文章存入主 'articles' 表。")
    return new_ids

def push_tasks_to_redis(redis_conn, queue_name, article_ids):
    if not article_ids: return
    redis_conn.lpush(queue_name, *article_ids)
    print(f"INFO: 成功将 {len(article_ids)} 个任务ID推送到Redis。")

# Cell 3: 主逻辑
if all([DB_CONFIG]):
        pg_conn = None
    #try:
            
        # --- 连接数据库并建表 ---
        pg_conn = psycopg2.connect(**DB_CONFIG)
        # 确保两张表都存在
        create_articles_table(pg_conn)
        create_screening_results_table(pg_conn)
        
        # --- 爬取文章 ---
        all_articles_to_process = []
        for i in range(1,19):
            base_url="https://your.url/c/ai?order=ID&p="+str(i)
            html = fetch_and_scrape(base_url)
            articles_from_web = parse_html_for_articles(html,base_url)
            all_articles_to_process.extend(articles_from_web)
            print(f"INFO: 已抓取第 {i} 页，累积文章数: {len(all_articles_to_process)}")
            time.sleep(1) # 友好抓取，在抓取每页之间稍作停顿
        

        CHUNK_SIZE = 50  # 定义每个小批量的大小，您可以根据API配额调整
        total_chunks = (len(all_articles_to_process) + CHUNK_SIZE - 1) // CHUNK_SIZE
        for i in range(total_chunks):
            start_index = i * CHUNK_SIZE
            end_index = start_index + CHUNK_SIZE
            article_chunk = all_articles_to_process[start_index:end_index]
            print(f"\n===== 开始处理批次 {i+1}/{total_chunks} (共 {len(article_chunk)} 篇文章) =====")

            if not article_chunk:
                continue

            # --- 5. 对每一个小批量，执行完整的“处理-保存”流程 ---
            #try:
            # a. 并发进行粗筛
            all_screening_results = filter_articles_in_parallel(article_chunk)
            
            # b. 将筛选结果写入日志表 (这是一个独立的事务，立刻保存)
            inserted_count = batch_insert_screening_results(pg_conn, all_screening_results)
            print(f"INFO: 批次 {i+1} - 已将 {inserted_count} 条新筛选结果写入 'screening_results' 表。")

            # c. 筛选出相关的文章
            relevant_articles = [res for res in all_screening_results if res['is_relevant']]
            
            if relevant_articles:
                # d. 将相关的文章存入主表并获取ID (这是另一个独立事务)
                new_ids = insert_relevant_articles_to_main_table(pg_conn, relevant_articles)
                
                # e. 将新ID推送到Redis任务队列
                #if new_ids:
                #    push_tasks_to_redis(redis_client, REDIS_QUEUE_NAME, new_ids)
            
            print(f"===== 批次 {i+1}/{total_chunks} 处理完成 =====\n")

            #except Exception as e:
            #    print(f"FATAL: 处理批次 {i+1} 时遇到严重错误: {e}")
            #    print("INFO: API配额可能已用尽，程序将终止。已处理的数据已保存。")
                # 当quota用尽等严重错误发生时，直接跳出循环

    #  except Exception as e:
    #      print(f"FATAL: 主程序出错: {e}")
    #  finally:
    #      if pg_conn:
    #          pg_conn.close()
    #          print("INFO: PostgreSQL连接已关闭。")
else:
    print("ERROR: 由于缺少关键配置（数据库、Redis或），主程序未运行。")
