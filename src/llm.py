"""LLM interaction utilities.

Provides functions for calling Google Gemini API with retry logic
and API key rotation.
"""

import json
import time
import logging

import google.generativeai as genai
import google.api_core.exceptions as google_exceptions
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.config import LLM_SAFETY_SETTINGS

logger = logging.getLogger(__name__)

# Retryable exception types
RETRYABLE_EXCEPTIONS = (
    google_exceptions.ResourceExhausted,
    google_exceptions.ServiceUnavailable,
    google_exceptions.DeadlineExceeded,
    google_exceptions.Aborted,
)


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
)
def screen_article(title, api_key_pool, model_name, system_prompt=""):
    """Screen a single article title for relevance using the LLM.

    Args:
        title: The article title to screen.
        api_key_pool: APIKeyPool instance for getting API keys.
        model_name: Name of the LLM model to use.
        system_prompt: Optional system prompt to prepend.

    Returns:
        dict: Result with keys 'is_relevant', 'relevance_score', 'model_used',
            'screening_duration_ms'.
    """
    start_time = time.time()
    result_data = {
        "is_relevant": False,
        "relevance_score": 0.0,
        "model_used": model_name,
    }

    try:
        api_key = api_key_pool.get_next_key()
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)

        prompt = system_prompt + f"""
        Respond ONLY with a valid JSON object containing three keys:
        1. "is_relevant": a boolean (true or false).
        2. "relevance_score": a float from 0.0 to 1.0.
        3. "reasoning": a brief string explanation.

        Title: "{title}"
        """

        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"},
            safety_settings=LLM_SAFETY_SETTINGS,
        )

        result = json.loads(response.text)
        result_data["is_relevant"] = result.get("is_relevant", False)
        result_data["relevance_score"] = result.get("relevance_score", 0.0)

    except RETRYABLE_EXCEPTIONS:
        raise  # Let tenacity handle retries
    except Exception as e:
        logger.warning("LLM筛选标题 '%s' 时出错: %s", title, e)

    duration_ms = int((time.time() - start_time) * 1000)
    result_data["screening_duration_ms"] = duration_ms
    return result_data


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
)
def run_deep_analysis(content, api_key_pool, model_name, system_prompt=""):
    """Run deep analysis on article content using the LLM.

    Args:
        content: Full article content/HTML.
        api_key_pool: APIKeyPool instance for getting API keys.
        model_name: Name of the LLM model to use.
        system_prompt: Optional system prompt to prepend.

    Returns:
        str: The LLM response text (may contain JSON in markdown code blocks).

    Raises:
        ValueError: If the LLM returns an empty response.
    """
    api_key = api_key_pool.get_next_key()
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    prompt = system_prompt + "\n\n---\n\n" + content

    try:
        response = model.generate_content(prompt, safety_settings=LLM_SAFETY_SETTINGS)

        if not response.text:
            feedback = (
                response.prompt_feedback
                if response.prompt_feedback
                else "No feedback available."
            )
            raise ValueError(
                f"LLM API返回空响应，可能被安全策略阻止。反馈: {feedback}"
            )

        return response.text

    except RETRYABLE_EXCEPTIONS as e:
        logger.warning("遇到可重试的API错误 (%s)，将由tenacity重试...", type(e).__name__)
        raise
    except Exception as e:
        logger.error("LLM分析失败: %s", e)
        raise


def parse_analysis_json(raw_text):
    """Parse JSON from LLM analysis response, handling markdown code blocks.

    Args:
        raw_text: Raw response text that may contain JSON in ```json blocks.

    Returns:
        dict: Parsed JSON data.

    Raises:
        ValueError: If no valid JSON could be extracted.
    """
    import re

    cleaned = raw_text.strip()

    # Try to extract JSON from markdown code block first
    match = re.search(r"```json\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if match:
        return json.loads(match.group(1))

    # Try to find raw JSON object
    match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
    if match:
        return json.loads(match.group(1))

    raise ValueError(f"无法从LLM响应中提取JSON: {cleaned[:200]}...")
