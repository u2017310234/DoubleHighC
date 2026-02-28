"""Tests for the PromptGuard class."""

import pytest

from src.prompt_guard import PromptGuard


class TestPromptGuardKeywords:
    """Tests for keyword detection."""

    def test_detects_default_keyword(self):
        guard = PromptGuard()
        assert guard.detect_keywords("This contains the word prompt in it") is True

    def test_detects_chinese_keyword(self):
        guard = PromptGuard()
        assert guard.detect_keywords("这段文本包含提示词") is True

    def test_no_keyword_match(self):
        guard = PromptGuard()
        assert guard.detect_keywords("This is a normal article about AI") is False

    def test_case_insensitive(self):
        guard = PromptGuard()
        assert guard.detect_keywords("PROMPT injection detected") is True

    def test_custom_keywords(self):
        guard = PromptGuard(keywords=["secret", "password"])
        assert guard.detect_keywords("Enter your password here") is True
        assert guard.detect_keywords("Normal text content") is False


class TestPromptGuardRegex:
    """Tests for regex pattern detection."""

    def test_detects_prompt_pattern(self):
        guard = PromptGuard()
        assert guard.detect_regex('prompt: "steal the data"') is True

    def test_detects_instruction_pattern(self):
        guard = PromptGuard()
        assert guard.detect_regex("instruction = do something bad") is True

    def test_no_regex_match(self):
        guard = PromptGuard()
        assert guard.detect_regex("A nice article about machine learning") is False


class TestPromptGuardLengthRatio:
    """Tests for length ratio detection."""

    def test_high_ratio_detected(self):
        guard = PromptGuard(length_ratio_threshold=5)
        assert guard.detect_length_ratio("short", "x" * 100) is True

    def test_normal_ratio_passes(self):
        guard = PromptGuard(length_ratio_threshold=10)
        assert guard.detect_length_ratio("input text", "output text") is False

    def test_empty_prompt_returns_false(self):
        guard = PromptGuard()
        assert guard.detect_length_ratio("", "some output") is False


class TestPromptGuardSimilarity:
    """Tests for content similarity detection."""

    def test_identical_text_detected(self):
        guard = PromptGuard(similarity_threshold=0.8)
        text = "This is the exact same text"
        assert guard.detect_similarity(text, text) is True

    def test_different_text_passes(self):
        guard = PromptGuard(similarity_threshold=0.8)
        assert guard.detect_similarity(
            "The quick brown fox jumps over the lazy dog",
            "人工智能在医疗领域的最新应用与发展趋势"
        ) is False

    def test_empty_input_returns_false(self):
        guard = PromptGuard()
        assert guard.detect_similarity("", "output") is False
        assert guard.detect_similarity("input", "") is False


class TestPromptGuardCheck:
    """Tests for the combined check method."""

    def test_detects_keyword_violation(self):
        guard = PromptGuard()
        assert guard.check("What is AI?", "The prompt says to ignore rules") is True

    def test_passes_clean_output(self):
        guard = PromptGuard()
        assert guard.check(
            "What is AI?",
            "AI stands for Artificial Intelligence. It is a broad field of computer science."
        ) is False
