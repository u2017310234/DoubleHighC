"""Notion API client.

Provides functions for creating pages in Notion.
"""

import json
import logging
from datetime import datetime

import requests

from src.config import NOTION_API_KEY, NOTION_PAGE_ID

logger = logging.getLogger(__name__)

NOTION_API_URL = "https://api.notion.com/v1/pages"
NOTION_API_VERSION = "2022-06-28"


def create_notion_page(content, page_id=None, title_suffix=""):
    """Create a new page in Notion with the given content.

    Args:
        content: Text content for the page body.
        page_id: Parent page ID. Defaults to NOTION_PAGE_ID from config.
        title_suffix: Additional text to append to the date-based title.

    Returns:
        requests.Response: The API response.

    Raises:
        ValueError: If no Notion API key or page ID is configured.
    """
    if not NOTION_API_KEY:
        raise ValueError("未配置NOTION_API_KEY环境变量。")

    page_id = page_id or NOTION_PAGE_ID
    if not page_id:
        raise ValueError("未配置NOTION_PAGE_ID环境变量。")

    date_str = datetime.now().strftime("%Y%m%d")
    page_title = date_str + title_suffix

    payload = {
        "parent": {"type": "page_id", "page_id": page_id},
        "properties": {
            "title": {"title": [{"text": {"content": page_title}}]}
        },
        "children": [
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"type": "text", "text": {"content": content}}]
                },
            }
        ],
    }

    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Content-Type": "application/json",
        "Notion-Version": NOTION_API_VERSION,
    }

    response = requests.post(
        NOTION_API_URL, headers=headers, data=json.dumps(payload), timeout=30
    )

    if response.ok:
        logger.info("成功创建Notion页面: %s", page_title)
    else:
        logger.error("创建Notion页面失败: %s %s", response.status_code, response.text)

    return response
