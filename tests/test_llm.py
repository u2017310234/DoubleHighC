"""Tests for LLM utility functions."""

import pytest

from src.llm import parse_analysis_json


class TestParseAnalysisJson:
    """Tests for JSON extraction from LLM responses."""

    def test_parses_plain_json(self):
        raw = '{"core_relevance": 8, "summary": "test"}'
        result = parse_analysis_json(raw)
        assert result["core_relevance"] == 8
        assert result["summary"] == "test"

    def test_parses_json_in_code_block(self):
        raw = """Here is the analysis:

```json
{"core_relevance": 9, "novelty_of_insight": 7}
```

That's the result.
"""
        result = parse_analysis_json(raw)
        assert result["core_relevance"] == 9
        assert result["novelty_of_insight"] == 7

    def test_parses_json_with_whitespace(self):
        raw = '  \n  {"key": "value"}  \n  '
        result = parse_analysis_json(raw)
        assert result["key"] == "value"

    def test_raises_on_no_json(self):
        with pytest.raises(ValueError, match="无法从LLM响应中提取JSON"):
            parse_analysis_json("This is just plain text with no JSON")

    def test_raises_on_empty_input(self):
        with pytest.raises(ValueError):
            parse_analysis_json("")

    def test_parses_json_with_surrounding_text(self):
        raw = 'The result is {"score": 42} and that is it.'
        result = parse_analysis_json(raw)
        assert result["score"] == 42
