"""Web scraping utilities.

Provides functions for fetching and parsing web pages.
"""

import logging

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

from src.config import DEFAULT_HEADERS

logger = logging.getLogger(__name__)


def resolve_url(base, url):
    """Resolve a relative URL against a base URL.

    Args:
        base: The base URL.
        url: The URL to resolve (may be relative).

    Returns:
        str: The resolved absolute URL.
    """
    return urljoin(base, url)


def fetch_page(url, timeout=15):
    """Fetch a web page and return its HTML content.

    Args:
        url: The URL to fetch.
        timeout: Request timeout in seconds.

    Returns:
        str or None: HTML content, or None on failure.
    """
    logger.info("正在从 %s 获取页面内容...", url)
    try:
        response = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
        response.raise_for_status()
        response.encoding = "utf-8"
        return response.text
    except requests.exceptions.RequestException as e:
        logger.error("访问 %s 失败: %s", url, e)
        return None


def parse_articles_from_html(html_content, base_url):
    """Parse article titles and links from HTML content.

    Expects a page structure with 'div.cc-cd-cb-l > a' selectors containing
    'span.t' title elements.

    Args:
        html_content: Raw HTML string.
        base_url: Base URL for resolving relative links.

    Returns:
        list: List of dicts with 'title' and 'link' keys.
    """
    if not html_content:
        return []
    soup = BeautifulSoup(html_content, "html.parser")
    link_elements = soup.select("div.cc-cd-cb-l > a")
    articles = []

    for a_tag in link_elements:
        title_span = a_tag.select_one("span.t")
        raw_link = a_tag.get("href")
        if title_span and raw_link:
            title = title_span.get_text(strip=True)
            final_link = resolve_url(base_url, raw_link)
            if title:
                articles.append({"title": title, "link": final_link})

    logger.info("爬虫解析完成，共发现 %d 篇文章。", len(articles))
    return articles


def scrape_full_content(url, timeout=20):
    """Scrape the full content of an article page.

    Args:
        url: The article URL.
        timeout: Request timeout in seconds.

    Returns:
        str: The page HTML content.

    Raises:
        requests.RequestException: If the request fails.
    """
    logger.info("正在抓取全文: %s...", url[:70] if len(url) > 70 else url)
    response = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
    response.raise_for_status()
    logger.info("抓取成功。")
    return response.text
