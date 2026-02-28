"""Prompt injection guard.

Provides the PromptGuard class for detecting potential prompt injection
or leakage in LLM outputs.
"""

import re
import logging

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class PromptGuard:
    """Guard against prompt injection and leakage in LLM outputs.

    Uses keyword detection, regex patterns, length ratio analysis,
    and TF-IDF similarity to detect suspicious outputs.

    Args:
        keywords: List of keywords to detect. Defaults to common prompt-related terms.
        similarity_threshold: Cosine similarity threshold (0-1). Default 0.8.
        length_ratio_threshold: Output/input length ratio threshold. Default 10.
    """

    DEFAULT_KEYWORDS = [
        "prompt",
        "instruction",
        "提示词",
        "指令",
        "##########",
        "#OBJECTIVE#",
    ]

    def __init__(self, keywords=None, similarity_threshold=0.8, length_ratio_threshold=10):
        self.keywords = keywords if keywords is not None else self.DEFAULT_KEYWORDS
        self.similarity_threshold = similarity_threshold
        self.length_ratio_threshold = length_ratio_threshold
        self.vectorizer = TfidfVectorizer()

    def detect_keywords(self, text):
        """Check if text contains any monitored keywords.

        Args:
            text: Text to check.

        Returns:
            bool: True if a keyword is found.
        """
        text_lower = text.lower()
        for keyword in self.keywords:
            if keyword.lower() in text_lower:
                return True
        return False

    def detect_regex(self, text, pattern=None):
        """Check if text matches prompt injection regex patterns.

        Args:
            text: Text to check.
            pattern: Custom regex pattern. Uses default if None.

        Returns:
            bool: True if a pattern match is found.
        """
        if pattern is None:
            pattern = (
                r"(prompt|instruction|提示词|指令)\s*[:=]?\s*[\"']?(.+?)[\"']?"
            )
        return bool(re.search(pattern, text, re.IGNORECASE))

    def detect_length_ratio(self, prompt, output):
        """Check if output length ratio is suspicious.

        Args:
            prompt: Input prompt text.
            output: LLM output text.

        Returns:
            bool: True if length ratio exceeds threshold.
        """
        prompt_len = len(prompt)
        if prompt_len == 0:
            return False
        return len(output) / prompt_len > self.length_ratio_threshold

    def detect_similarity(self, prompt, output):
        """Check if output is suspiciously similar to the prompt.

        Args:
            prompt: Input prompt text.
            output: LLM output text.

        Returns:
            bool: True if similarity exceeds threshold.
        """
        if not prompt or not output:
            return False
        vectors = self.vectorizer.fit_transform([prompt, output])
        sim = cosine_similarity(vectors)[0][1]
        return bool(sim > self.similarity_threshold)

    def check(self, prompt, output):
        """Run all detection checks on an LLM output.

        Args:
            prompt: Input prompt text.
            output: LLM output text.

        Returns:
            bool: True if any check detects a violation.
        """
        if self.detect_keywords(output):
            logger.warning("检测到关键词")
            return True
        if self.detect_regex(output):
            logger.warning("检测到正则表达式模式")
            return True
        if self.detect_length_ratio(prompt, output):
            logger.warning("检测到长度比例过高")
            return True
        if self.detect_similarity(prompt, output):
            logger.warning("检测到内容相似度过高")
            return True
        return False
