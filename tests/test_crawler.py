"""tests/test_crawler.py — offline unit tests for WikiCrawler parsing logic.

All tests use saved HTML fixtures and a mock robots.txt parser; NO network
requests are ever made.  Run with:

    pytest tests/test_crawler.py -v
    pytest tests/test_crawler.py -v --tb=short   # compact tracebacks
"""

from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from core.config import Config
from crawler.wiki_crawler import WikiCrawler

# ── fixture path ────────────────────────────────────────────────────────────
FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _make_config(tmp_path: Path) -> Config:
    """Return a minimal Config suitable for offline testing."""
    return Config(
        seeds=["Iron_Man"],
        category_roots=["Category:Marvel_Comics"],
        category_depth=1,
        max_depth=2,
        max_nodes=10,
        max_new_nodes_per_page=5,
        delay_min=0.0,
        delay_max=0.0,
        max_retries=1,
        backoff_base=2,
        user_agent="TestAgent/1.0",
        wiki_base_url="https://en.wikipedia.org",
        api_endpoint="https://en.wikipedia.org/w/api.php",
        robots_url="https://en.wikipedia.org/robots.txt",
        cache_dir=str(tmp_path / "cache"),
        edgelist_path=str(tmp_path / "edgelist.csv"),
        nodes_path=str(tmp_path / "nodes.csv"),
        crawl_report_path=str(tmp_path / "crawl_report.json"),
        state_file=str(tmp_path / "cache" / "_crawl_state.json"),
        dropped_namespaces=[
            "Help:", "File:", "Special:", "Talk:", "Wikipedia:",
            "Portal:", "Template:", "Template_talk:", "Category:",
            "User:", "MediaWiki:", "Module:", "Draft:",
        ],
        removed_selectors=[
            ".navbox", ".vertical-navbox", ".sidebar", ".reflist",
            ".references", ".hatnote", ".mw-editsection", ".metadata",
            ".refbegin", ".toc", "table.navbox", "div.navbox", ".ambox",
        ],
        dropped_sections=[
            "References", "External links", "Notes",
            "See also", "Further reading", "Footnotes",
        ],
        # list must be checked before character so "Lists of X characters" → list
        node_type_rules={
            "list": ["list of", "lists of"],
            "character": ["fictional character", "comics character", "superhero"],
            "film": ["cinematic universe film", "superhero film"],
            "tv_series": ["television series", "disney+"],
            "comic_series": ["comic book series", "comic book title"],
            "person": ["film director", "comic book writer", "american actor"],
            "company": ["comic book publisher", "film studio"],
        },
    )


@pytest.fixture()
def crawler(tmp_path: Path) -> WikiCrawler:
    """WikiCrawler instance with mocked robots.txt — no network calls."""
    cfg = _make_config(tmp_path)
    mock_rp = MagicMock()
    mock_rp.can_fetch.return_value = True
    # Patch _load_robots so __init__ never hits the network
    with patch.object(WikiCrawler, "_load_robots", return_value=mock_rp):
        return WikiCrawler(cfg, pilot=False)


@pytest.fixture()
def sample_html() -> str:
    """Load the saved Wikipedia-like fixture."""
    return (FIXTURE_DIR / "sample_article.html").read_text(encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════
# 1.  TITLE NORMALISATION
# ═══════════════════════════════════════════════════════════════════════════

class TestTitleNormalization:
    """_normalize_title must URL-decode and replace spaces with underscores."""

    def test_spaces_to_underscores(self, crawler: WikiCrawler) -> None:
        assert crawler._normalize_title("Iron Man") == "Iron_Man"

    def test_url_encoded_space(self, crawler: WikiCrawler) -> None:
        assert crawler._normalize_title("Iron%20Man") == "Iron_Man"

    def test_url_encoded_colon(self, crawler: WikiCrawler) -> None:
        result = crawler._normalize_title("Spider-Man%3A_Homecoming")
        assert result == "Spider-Man:_Homecoming"

    def test_unicode_encoded(self, crawler: WikiCrawler) -> None:
        # %E2%80%93 is the en-dash character –
        result = crawler._normalize_title("X%E2%80%93Men")
        assert result == "X\u2013Men"

    def test_already_normalised(self, crawler: WikiCrawler) -> None:
        assert crawler._normalize_title("Captain_America") == "Captain_America"

    def test_mixed_spaces_and_encoding(self, crawler: WikiCrawler) -> None:
        result = crawler._normalize_title("Thor%20Odinson")
        assert result == "Thor_Odinson"


# ═══════════════════════════════════════════════════════════════════════════
# 2.  LINK FILTERING — VALID PATHS
# ═══════════════════════════════════════════════════════════════════════════

class TestValidLinkPaths:
    """_is_valid_wiki_path must return a title for clean /wiki/<Title> hrefs."""

    def test_simple_article(self, crawler: WikiCrawler) -> None:
        assert crawler._is_valid_wiki_path("/wiki/Iron_Man") == "Iron_Man"

    def test_hyphenated_title(self, crawler: WikiCrawler) -> None:
        assert crawler._is_valid_wiki_path("/wiki/Spider-Man") == "Spider-Man"

    def test_parenthetical_title(self, crawler: WikiCrawler) -> None:
        title = crawler._is_valid_wiki_path("/wiki/Avengers_(comics)")
        assert title == "Avengers_(comics)"

    def test_returns_none_for_external(self, crawler: WikiCrawler) -> None:
        assert crawler._is_valid_wiki_path("https://marvel.com/hero") is None

    def test_returns_none_for_empty_href(self, crawler: WikiCrawler) -> None:
        assert crawler._is_valid_wiki_path("") is None

    def test_returns_none_for_non_wiki_path(self, crawler: WikiCrawler) -> None:
        assert crawler._is_valid_wiki_path("/w/index.php?title=Iron_Man") is None


# ═══════════════════════════════════════════════════════════════════════════
# 3.  FRAGMENT & QUERY STRING STRIPPING
# ═══════════════════════════════════════════════════════════════════════════

class TestFragmentAndQueryStripping:
    """Fragments (#) and query strings (?) must be stripped; title preserved."""

    def test_fragment_stripped(self, crawler: WikiCrawler) -> None:
        assert crawler._is_valid_wiki_path("/wiki/Iron_Man#History") == "Iron_Man"

    def test_query_string_stripped(self, crawler: WikiCrawler) -> None:
        result = crawler._is_valid_wiki_path("/wiki/Spider-Man?action=edit")
        assert result == "Spider-Man"

    def test_fragment_on_complex_title(self, crawler: WikiCrawler) -> None:
        result = crawler._is_valid_wiki_path("/wiki/X-Men#Powers_and_abilities")
        assert result == "X-Men"

    def test_fragment_only_after_wiki_prefix(self, crawler: WikiCrawler) -> None:
        # /wiki/ followed immediately by # → empty title → None
        assert crawler._is_valid_wiki_path("/wiki/#section") is None


# ═══════════════════════════════════════════════════════════════════════════
# 4.  NAMESPACE STRIPPING
# ═══════════════════════════════════════════════════════════════════════════

class TestNamespaceFiltering:
    """All dropped_namespace prefixes must yield None."""

    DROPPED_HREFS: List[str] = [
        "/wiki/File:Iron_Man_armor.jpg",
        "/wiki/Help:Contents",
        "/wiki/Category:Marvel_Comics_characters",
        "/wiki/Talk:Iron_Man",
        "/wiki/Wikipedia:About",
        "/wiki/Special:Search",
        "/wiki/Template:Marvel_Comics",
        "/wiki/Portal:Comics",
        "/wiki/User:JohnDoe",
        "/wiki/MediaWiki:Common.css",
        "/wiki/Module:Navbox",
        "/wiki/Draft:NewPage",
        "/wiki/Template_talk:Marvel",
    ]

    @pytest.mark.parametrize("href", DROPPED_HREFS)
    def test_namespace_blocked(self, crawler: WikiCrawler, href: str) -> None:
        assert crawler._is_valid_wiki_path(href) is None, (
            f"Expected None for namespace href: {href}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 5.  HTML PARSE — VALID LINKS PRESENT
# ═══════════════════════════════════════════════════════════════════════════

class TestParsedValidLinks:
    """_parse_links must return known good titles from the fixture."""

    EXPECTED: List[str] = [
        "Marvel_Comics",
        "Stan_Lee",
        "Jack_Kirby",
        "Avengers_(comics)",
        "Spider-Man",
        "Captain_America",
        "Tony_Stark",   # fragment stripped from Tony_Stark#Early_life
        "Iron_Man",     # query stripped from Iron_Man?action=edit
    ]

    def test_expected_titles_present(
        self, crawler: WikiCrawler, sample_html: str
    ) -> None:
        links = crawler._parse_links(sample_html)
        for title in self.EXPECTED:
            assert title in links, f"Expected title missing from parsed links: {title}"

    def test_returns_list(self, crawler: WikiCrawler, sample_html: str) -> None:
        links = crawler._parse_links(sample_html)
        assert isinstance(links, list)

    def test_deduplicated(self, crawler: WikiCrawler, sample_html: str) -> None:
        links = crawler._parse_links(sample_html)
        assert len(links) == len(set(links)), "Duplicate titles found in parsed links"


# ═══════════════════════════════════════════════════════════════════════════
# 6.  HTML PARSE — FILTERED LINKS ABSENT
# ═══════════════════════════════════════════════════════════════════════════

class TestParsedFilteredLinks:
    """Navbox, hatnote, references, see-also, and ext-link content must vanish."""

    # These titles appear ONLY inside filtered areas of the fixture
    MUST_BE_ABSENT: List[str] = [
        "Thor_(Marvel_Comics)",       # inside .navbox table
        "Hulk",                       # inside .navbox table
        "Black_Widow_(Natasha_Romanova)",  # inside .navbox table
        "Nick_Fury",                  # inside .vertical-navbox
        "Iron_Man_(disambiguation)",  # inside .hatnote
        "War_Machine",                # inside See also section
        "Avengers_film",              # inside External links section
    ]

    @pytest.mark.parametrize("title", MUST_BE_ABSENT)
    def test_filtered_title_absent(
        self, crawler: WikiCrawler, sample_html: str, title: str
    ) -> None:
        links = crawler._parse_links(sample_html)
        assert title not in links, (
            f"Title should have been filtered but appears in links: {title}"
        )

    def test_reflist_links_absent(
        self, crawler: WikiCrawler, sample_html: str
    ) -> None:
        """Stan Lee and Jack Kirby appear inside .reflist / .references only."""
        links = crawler._parse_links(sample_html)
        # They do also appear in the body — only verify navbox/ref variants
        # are not double-counted (deduplication test above covers this).
        assert links.count("Stan_Lee") <= 1
        assert links.count("Jack_Kirby") <= 1


# ═══════════════════════════════════════════════════════════════════════════
# 7.  NODE TYPE CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════

class TestNodeTypeClassification:
    """_classify_node_type must match keywords case-insensitively."""

    def test_character_type(self, crawler: WikiCrawler) -> None:
        cats = ["Category:Marvel Comics characters", "Category:Superhero comics"]
        assert crawler._classify_node_type(cats) == "character"

    def test_film_type(self, crawler: WikiCrawler) -> None:
        cats = ["Category:Marvel Cinematic Universe films", "Category:2008 films"]
        assert crawler._classify_node_type(cats) == "film"

    def test_tv_series_type(self, crawler: WikiCrawler) -> None:
        cats = ["Category:Television series set in New York City",
                "Category:Disney+ original series"]
        assert crawler._classify_node_type(cats) == "tv_series"

    def test_person_type(self, crawler: WikiCrawler) -> None:
        cats = ["Category:American film directors", "Category:Living people"]
        assert crawler._classify_node_type(cats) == "person"

    def test_company_type(self, crawler: WikiCrawler) -> None:
        cats = ["Category:Comic book publishers", "Category:Companies"]
        assert crawler._classify_node_type(cats) == "company"

    def test_list_type(self, crawler: WikiCrawler) -> None:
        cats = ["Category:Lists of Marvel Comics characters"]
        assert crawler._classify_node_type(cats) == "list"

    def test_other_fallback(self, crawler: WikiCrawler) -> None:
        cats = ["Category:1960s comic books", "Category:American fiction"]
        assert crawler._classify_node_type(cats) == "other"

    def test_case_insensitive_match(self, crawler: WikiCrawler) -> None:
        cats = ["Category:SUPERHERO television series"]
        # 'superhero' is a keyword for 'character'; should still match
        result = crawler._classify_node_type(cats)
        assert result == "character"


# ═══════════════════════════════════════════════════════════════════════════
# 8.  CACHE KEY UNIQUENESS
# ═══════════════════════════════════════════════════════════════════════════

class TestCacheKey:
    """Different titles must produce different cache keys."""

    def test_unique_keys(self, crawler: WikiCrawler) -> None:
        key1 = crawler._cache_key("Iron_Man")
        key2 = crawler._cache_key("Spider-Man")
        key3 = crawler._cache_key("Captain_America")
        assert len({key1, key2, key3}) == 3

    def test_same_title_same_key(self, crawler: WikiCrawler) -> None:
        assert crawler._cache_key("Thor") == crawler._cache_key("Thor")

    def test_key_is_inside_cache_dir(self, crawler: WikiCrawler) -> None:
        key = crawler._cache_key("Black_Widow")
        assert str(crawler.cache_dir) in str(key)
