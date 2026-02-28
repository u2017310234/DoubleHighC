"""Tests for scraper utilities."""

import pytest

from src.scraper import resolve_url, parse_articles_from_html


class TestResolveUrl:
    """Tests for URL resolution."""

    def test_resolves_relative_url(self):
        assert resolve_url("https://example.com/path/", "article.html") == \
            "https://example.com/path/article.html"

    def test_resolves_absolute_url(self):
        assert resolve_url("https://example.com/", "https://other.com/page") == \
            "https://other.com/page"

    def test_resolves_root_relative(self):
        assert resolve_url("https://example.com/path/page", "/article") == \
            "https://example.com/article"


class TestParseArticles:
    """Tests for HTML article parsing."""

    def test_parses_articles_correctly(self):
        html = """
        <html><body>
        <div class="cc-cd-cb-l">
            <a href="/article/1">
                <span class="t">Article One</span>
            </a>
        </div>
        <div class="cc-cd-cb-l">
            <a href="/article/2">
                <span class="t">Article Two</span>
            </a>
        </div>
        </body></html>
        """
        articles = parse_articles_from_html(html, "https://example.com")
        assert len(articles) == 2
        assert articles[0]["title"] == "Article One"
        assert articles[0]["link"] == "https://example.com/article/1"
        assert articles[1]["title"] == "Article Two"

    def test_empty_html_returns_empty(self):
        assert parse_articles_from_html("", "https://example.com") == []

    def test_none_html_returns_empty(self):
        assert parse_articles_from_html(None, "https://example.com") == []

    def test_no_matching_elements_returns_empty(self):
        html = "<html><body><p>No articles here</p></body></html>"
        assert parse_articles_from_html(html, "https://example.com") == []

    def test_skips_entries_without_title(self):
        html = """
        <html><body>
        <div class="cc-cd-cb-l">
            <a href="/article/1">
                <span class="t"></span>
            </a>
        </div>
        <div class="cc-cd-cb-l">
            <a href="/article/2">
                <span class="t">Valid Title</span>
            </a>
        </div>
        </body></html>
        """
        articles = parse_articles_from_html(html, "https://example.com")
        assert len(articles) == 1
        assert articles[0]["title"] == "Valid Title"
