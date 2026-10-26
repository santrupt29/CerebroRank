"""crawler/wiki_crawler.py — Ethical hybrid BFS + category-membership Wikipedia crawler.

════════════════════════════════════════════════════════════════════════════════
EDGE POLICY
    An edge u → v is recorded ONLY when BOTH u AND v are accepted nodes
    (i.e., fetched, validated, and present in self.nodes).  Pages discovered
    on the frontier that were never fetched are NOT nodes and therefore
    appear in NO edge.  max_new_nodes_per_page limits queue growth only —
    it never drops edges between already-accepted nodes.

CRAWL STRATEGY
    Phase 1 — Allowed-set: query the MediaWiki categorymembers API for all
               article titles reachable from category_roots within
               category_depth subcategory levels.  Every API response is
               cached.
    Phase 2 — BFS: seeds enqueued first, allowed-set titles second.  A page
               is accepted if (a) it is in the allowed set, OR (b) a batched
               live category-API check finds it belongs to an allowed
               category.  BFS stops at max_depth hops OR max_nodes nodes.
    Phase 3 — Edges: deferred until BFS is done so both endpoints can be
               confirmed as nodes.

ETHICAL CONTROLS
    • robots.txt parsed once (cached); can_fetch() enforced before every
      HTML request.
    • Explicit User-Agent on every request (HTML + API).
    • Randomised uniform(delay_min, delay_max) sleep between requests.
    • Exponential back-off on HTTP 429/503; Retry-After header honoured.
    • All responses cached in data/cache/ keyed by content hash of params.
      Re-runs make ZERO network requests for cached items.
    • Full crawl state persisted every 10 nodes → resumable after crash.
    • Only public article links; no personal data collected.
════════════════════════════════════════════════════════════════════════════════
"""

import csv
import hashlib
import json
import logging
import random
import re
import time
import urllib.parse
import urllib.robotparser
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
from bs4 import BeautifulSoup, NavigableString, Tag

from core.config import Config

logger = logging.getLogger(__name__)

# How often (in nodes) to flush resumable state to disk
_STATE_SAVE_INTERVAL = 10


class WikiCrawler:
    """Ethical Wikipedia crawler — BFS + MediaWiki category-membership gating."""

    def __init__(self, cfg: Config, pilot: bool = False) -> None:
        """Initialise crawler.  Loads robots.txt and any prior resumable state.

        Args:
            cfg:   Loaded :class:`~core.config.Config` object.
            pilot: If True cap at 30 nodes for quick smoke-testing.
        """
        self.cfg = cfg
        self.pilot = pilot
        self.max_nodes: int = 30 if pilot else cfg.max_nodes

        # ── core data ──────────────────────────────────────────────────────
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: Set[Tuple[str, str]] = set()
        self.allowed_set: Set[str] = set()
        # All category titles seen while building the allowed-set (used in
        # live category checks to match transitive membership).
        self.allowed_categories: Set[str] = set()
        self.visited: Set[str] = set()
        # outgoing_links[src] = raw link list; edges recorded post-BFS
        self.outgoing_links: Dict[str, List[str]] = {}

        # ── statistics ─────────────────────────────────────────────────────
        self.stats: Dict[str, Any] = {
            "pages_fetched": 0,
            "api_calls": 0,
            "cache_hits": 0,
            "robots_skipped": 0,
            "retries": 0,
            "backoff_events": 0,
            "rejected_by_category": 0,
            "total_runtime": 0.0,
            "nodes": 0,
            "edges": 0,
        }

        # ── paths ──────────────────────────────────────────────────────────
        self.cache_dir = Path(cfg.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        Path(cfg.edgelist_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.crawl_report_path).parent.mkdir(parents=True, exist_ok=True)

        # ── HTTP session ───────────────────────────────────────────────────
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": cfg.user_agent})

        # ── robots.txt (must happen before any HTML fetch) ─────────────────
        self.robot_parser = self._load_robots()

        # ── resume previous run if state exists ────────────────────────────
        self._load_state()
        logger.info(
            "WikiCrawler ready | pilot=%s max_nodes=%d resumed_nodes=%d",
            pilot, self.max_nodes, len(self.nodes),
        )

    # ══════════════════════════════════════════════════════════════════════
    # ROBOTS / RATE-LIMITING
    # ══════════════════════════════════════════════════════════════════════

    def _load_robots(self) -> urllib.robotparser.RobotFileParser:
        """Load robots.txt from cache; fetch from network only on first run."""
        rp = urllib.robotparser.RobotFileParser()
        robots_cache = self.cache_dir / "_robots.txt"
        if robots_cache.exists():
            rp.parse(robots_cache.read_text(encoding="utf-8").splitlines())
            logger.debug("robots.txt loaded from cache.")
        else:
            resp = self.session.get(self.cfg.robots_url, timeout=15)
            resp.raise_for_status()
            robots_cache.write_text(resp.text, encoding="utf-8")
            rp.parse(resp.text.splitlines())
            logger.info("robots.txt fetched and cached.")
        return rp

    def _can_fetch(self, url: str) -> bool:
        """Return True if robots.txt allows the crawler to fetch *url*."""
        return self.robot_parser.can_fetch(self.cfg.user_agent, url)

    def _sleep(self) -> None:
        """Uniform random delay — NOT seeded so actual wall-clock spacing varies."""
        time.sleep(random.uniform(self.cfg.delay_min, self.cfg.delay_max))

    def _get_with_retry(
        self,
        url: str,
        params: Optional[Dict[str, str]] = None,
        is_api: bool = False,
    ) -> Optional[requests.Response]:
        """GET *url* with exponential back-off on 429/503.

        Robots.txt is checked for non-API requests.  Returns None on
        persistent failure so callers can skip gracefully.
        """
        if not is_api:
            if not self._can_fetch(url):
                logger.warning("robots.txt blocks: %s", url)
                self.stats["robots_skipped"] += 1
                return None

        for attempt in range(self.cfg.max_retries):
            try:
                resp = self.session.get(url, params=params, timeout=20)
                if resp.status_code == 200:
                    return resp
                if resp.status_code in (429, 503):
                    wait = int(resp.headers.get(
                        "Retry-After",
                        self.cfg.backoff_base ** (attempt + 1),
                    ))
                    wait = min(wait, 120)
                    logger.warning(
                        "HTTP %d on %s → waiting %ds (attempt %d/%d)",
                        resp.status_code, url, wait, attempt + 1, self.cfg.max_retries,
                    )
                    self.stats["backoff_events"] += 1
                    self.stats["retries"] += 1
                    time.sleep(wait)
                else:
                    logger.error("HTTP %d on %s — giving up.", resp.status_code, url)
                    return None
            except requests.RequestException as exc:
                wait = self.cfg.backoff_base ** (attempt + 1)
                logger.warning("Request error on %s: %s → retry in %ds", url, exc, wait)
                self.stats["retries"] += 1
                time.sleep(wait)

        logger.error("All %d retries exhausted for %s", self.cfg.max_retries, url)
        return None

    # ══════════════════════════════════════════════════════════════════════
    # CACHING UTILITIES
    # ══════════════════════════════════════════════════════════════════════

    def _cache_key(self, title: str) -> Path:
        """Deterministic cache path: safe prefix + MD5 suffix avoids collisions."""
        safe = re.sub(r"[^\w\-]", "_", title)[:60]
        digest = hashlib.md5(title.encode("utf-8")).hexdigest()[:10]
        return self.cache_dir / f"{safe}_{digest}.json"

    def _api_cache_key(self, params: Dict[str, str]) -> Path:
        """Cache path for an API call keyed by sorted JSON of params."""
        payload = json.dumps(params, sort_keys=True).encode("utf-8")
        digest = hashlib.md5(payload).hexdigest()[:14]
        return self.cache_dir / f"api_{digest}.json"

    # ══════════════════════════════════════════════════════════════════════
    # TITLE NORMALISATION
    # ══════════════════════════════════════════════════════════════════════

    def _normalize_title(self, raw: str) -> str:
        """URL-decode a wiki title and normalise spaces → underscores."""
        decoded = urllib.parse.unquote(raw)
        return decoded.replace(" ", "_")

    def _is_valid_wiki_path(self, href: str) -> Optional[str]:
        """Return normalised title if *href* is a valid article link, else None.

        Drops: non-/wiki/ paths, fragments, query strings, dropped namespaces,
        and empty titles.
        """
        if href.startswith(("https://en.wikipedia.org/wiki/", "http://en.wikipedia.org/wiki/")):
            raw = href.split("/wiki/", 1)[1]
        elif href.startswith("//en.wikipedia.org/wiki/"):
            raw = href.split("/wiki/", 1)[1]
        elif href.startswith("/wiki/"):
            raw = href[6:]
        else:
            return None
        raw = raw.split("#")[0]      # strip fragment
        raw = raw.split("?")[0]      # strip query string
        if not raw:
            return None
        for ns in self.cfg.dropped_namespaces:
            if raw.startswith(ns) or raw.startswith(ns.replace(":", "_")):
                return None
        return self._normalize_title(raw)

    def _title_from_url(self, url: str) -> str:
        """Extract normalised title from a Wikipedia article URL."""
        path = urllib.parse.urlparse(url).path
        if path.startswith("/wiki/"):
            return self._normalize_title(path[6:])
        return ""

    # ══════════════════════════════════════════════════════════════════════
    # MEDIAWIKI API
    # ══════════════════════════════════════════════════════════════════════

    def _api_request(self, params: Dict[str, str]) -> Optional[Dict]:
        """Make a single MediaWiki API request; result is always cached."""
        params = {**params, "format": "json"}
        cache_file = self._api_cache_key(params)
        if cache_file.exists():
            self.stats["cache_hits"] += 1
            return json.loads(cache_file.read_text(encoding="utf-8"))

        self._sleep()
        resp = self._get_with_retry(self.cfg.api_endpoint, params=params, is_api=True)
        if resp is None:
            return None
        data = resp.json()
        cache_file.write_text(json.dumps(data), encoding="utf-8")
        self.stats["api_calls"] += 1
        return data

    def _build_allowed_set(self) -> None:
        """Populate self.allowed_set via categorymembers API (BFS over subcats).

        Uses continuation tokens to page through results.  Caches every
        individual API response so re-runs are fully offline.
        """
        def _collect(cat_title: str, depth: int) -> None:
            if depth < 0 or cat_title in self.allowed_categories:
                return
            self.allowed_categories.add(cat_title)
            cont: Optional[str] = None
            while True:
                params: Dict[str, str] = {
                    "action": "query",
                    "list": "categorymembers",
                    "cmtitle": cat_title,
                    "cmtype": "page|subcat",
                    "cmlimit": "500",
                }
                if cont:
                    params["cmcontinue"] = cont
                data = self._api_request(params)
                if not data:
                    break
                for member in data.get("query", {}).get("categorymembers", []):
                    ns = member.get("ns", 0)
                    norm = self._normalize_title(member["title"])
                    if ns == 0:          # article namespace
                        self.allowed_set.add(norm)
                    elif ns == 14 and depth > 0:   # Category namespace
                        _collect(member["title"], depth - 1)
                cont = data.get("continue", {}).get("cmcontinue")
                if not cont:
                    break

        for root in self.cfg.category_roots:
            logger.info("Building allowed-set from: %s", root)
            _collect(root, self.cfg.category_depth)
        logger.info("Allowed-set: %d articles | %d categories",
                    len(self.allowed_set), len(self.allowed_categories))

    def _live_category_check(self, titles: List[str]) -> Set[str]:
        """Batch-check titles (≤50 per request) against allowed_categories.

        Returns the subset of *titles* that belong to at least one allowed
        category (direct membership only — transitive via allowed_categories set).
        """
        approved: Set[str] = set()
        for i in range(0, len(titles), 50):
            batch = titles[i : i + 50]
            params: Dict[str, str] = {
                "action": "query",
                "prop": "categories",
                "titles": "|".join(t.replace("_", " ") for t in batch),
                "cllimit": "100",
            }
            data = self._api_request(params)
            if not data:
                continue
            for page in data.get("query", {}).get("pages", {}).values():
                norm = self._normalize_title(page.get("title", ""))
                page_cats = {c["title"] for c in page.get("categories", [])}
                if page_cats & self.allowed_categories:
                    approved.add(norm)
        return approved

    # ══════════════════════════════════════════════════════════════════════
    # HTML PARSING
    # ══════════════════════════════════════════════════════════════════════

    def _remove_dropped_sections(self, content: Tag) -> None:
        """Remove dropped sections (heading + body) in-place.

        Handles both old Wikipedia heading style
        (<h2><span class='mw-headline'>…</span></h2>) and the new wrapper
        style (<div class='mw-heading'><h2>…</h2></div>).
        """
        drop_names = set(self.cfg.dropped_sections)
        # Collect all heading-like elements (old and new style)
        candidates = content.find_all(
            lambda tag: tag.name in ("h2", "h3", "h4")
            or (tag.name == "div" and "mw-heading" in tag.get("class", []))
        )
        for heading in candidates:
            text = heading.get_text(separator=" ", strip=True)
            if not any(d.lower() in text.lower() for d in drop_names):
                continue
            # Remove everything up to the next sibling heading
            siblings = list(heading.next_siblings)
            for sib in siblings:
                if isinstance(sib, Tag) and sib.name in ("h2", "h3", "h4"):
                    break
                if isinstance(sib, Tag) and "mw-heading" in sib.get("class", []):
                    break
                if hasattr(sib, "decompose"):
                    sib.decompose()
            heading.decompose()

    def _parse_links(self, html: str) -> List[str]:
        """Extract deduplicated, normalised article titles from page HTML.

        Steps:
        1. Target only #mw-content-text .mw-parser-output.
        2. Remove noisy elements (navboxes, refs, hatnotes …).
        3. Remove dropped sections (References, External links …).
        4. Collect /wiki/ hrefs; filter namespaces; strip fragments/queries.
        5. Deduplicate; drop self-links (handled by caller).
        """
        soup = BeautifulSoup(html, "lxml")
        content = soup.find("div", id="mw-content-text")
        if not content:
            return []

        # Step 2: strip noisy structural elements
        for selector in self.cfg.removed_selectors:
            for el in content.select(selector):
                el.decompose()

        # Step 3: strip dropped sections
        self._remove_dropped_sections(content)  # type: ignore[arg-type]

        parser_out = content.find("div", class_="mw-parser-output") or content
        seen: Set[str] = set()
        result: List[str] = []
        for a_tag in parser_out.find_all("a", href=True):
            title = self._is_valid_wiki_path(a_tag["href"])
            if title and title not in seen:
                seen.add(title)
                result.append(title)
        return result

    # ══════════════════════════════════════════════════════════════════════
    # FETCH & NODE CREATION
    # ══════════════════════════════════════════════════════════════════════

    def _fetch_and_resolve(self, title: str) -> Tuple[Optional[str], str]:
        """Return (html, canonical_title); read from cache when available."""
        cache_file = self._cache_key(title)
        if cache_file.exists():
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            self.stats["cache_hits"] += 1
            return cached.get("html"), cached.get("canonical", title)

        url = f"{self.cfg.wiki_base_url}/wiki/{urllib.parse.quote(title)}"
        self._sleep()
        resp = self._get_with_retry(url)
        if resp is None:
            return None, title

        canonical = self._title_from_url(resp.url) or title
        payload = {"title": title, "canonical": canonical, "html": resp.text}
        cache_file.write_text(json.dumps(payload), encoding="utf-8")
        self.stats["pages_fetched"] += 1
        return resp.text, canonical

    def _fetch_page_categories(self, title: str) -> List[str]:
        """Return the list of raw category title strings for *title*."""
        params: Dict[str, str] = {
            "action": "query",
            "prop": "categories",
            "titles": title.replace("_", " "),
            "cllimit": "100",
        }
        data = self._api_request(params)
        if not data:
            return []
        for page in data.get("query", {}).get("pages", {}).values():
            return [c["title"] for c in page.get("categories", [])]
        return []

    def _classify_node_type(self, categories: List[str]) -> str:
        """Map raw category strings to a node_type label via keyword rules.

        First matching rule in config wins; falls back to 'other'.
        """
        cat_text = " ".join(categories).lower()
        for node_type, keywords in self.cfg.node_type_rules.items():
            if any(kw.lower() in cat_text for kw in keywords):
                return node_type
        return "other"

    def _add_node(self, title: str, categories: List[str]) -> None:
        """Insert a new node into self.nodes with metadata."""
        node_type = self._classify_node_type(categories)
        url = f"{self.cfg.wiki_base_url}/wiki/{urllib.parse.quote(title)}"
        self.nodes[title] = {
            "id": len(self.nodes),
            "title": title,
            "url": url,
            "node_type": node_type,
            # Store at most 15 category names; full list may be huge
            "categories": "|".join(categories[:15]),
            # Degrees filled in Phase 3
            "degree": 0,
            "in_degree": 0,
            "out_degree": 0,
        }
        logger.info(
            "[%d/%d] Added node: %s (type=%s)",
            len(self.nodes), self.max_nodes, title, node_type,
        )

    # ══════════════════════════════════════════════════════════════════════
    # BFS HELPERS
    # ══════════════════════════════════════════════════════════════════════

    def _is_accepted(self, title: str) -> bool:
        """Return True if *title* passes the allowed-set or live-category gate."""
        if title in self.allowed_set:
            return True
        approved = self._live_category_check([title])
        if title not in approved:
            logger.debug("Rejected (category): %s", title)
            self.stats["rejected_by_category"] += 1
            return False
        return True

    def _initialize_queue(self) -> Tuple[deque, Set[str]]:
        """Seed the BFS queue: seeds first, then remaining allowed-set titles.

        Seeds get priority (depth=0, added first).  This ensures the most
        important pages are visited even if max_nodes is hit quickly.
        """
        seed_titles = [self._normalize_title(s) for s in self.cfg.seeds]
        seed_set = set(seed_titles)

        # Pre-populate with already-visited nodes so we resume correctly
        queue: deque = deque()
        queued: Set[str] = set(self.visited)

        for s in seed_titles:
            if s not in queued:
                queue.append((s, 0))
                queued.add(s)

        for t in self.allowed_set:
            if t not in queued and t not in seed_set:
                queue.append((t, 0))
                queued.add(t)

        logger.info("Queue initialised: %d entries", len(queue))
        return queue, queued

    def _enqueue_links(
        self,
        links: List[str],
        depth: int,
        queue: deque,
        queued: Set[str],
    ) -> None:
        """Add at most max_new_nodes_per_page new titles to the BFS queue.

        Prefers links already in the allowed-set (more likely to be accepted).
        """
        priority = [l for l in links if l in self.allowed_set and l not in queued]
        others = [l for l in links if l not in self.allowed_set and l not in queued]
        added = 0
        for target in priority + others:
            if added >= self.cfg.max_new_nodes_per_page:
                break
            queue.append((target, depth + 1))
            queued.add(target)
            added += 1

    # ══════════════════════════════════════════════════════════════════════
    # MAIN BFS LOOP
    # ══════════════════════════════════════════════════════════════════════

    def _run_bfs(self, queue: deque, queued: Set[str]) -> None:
        """Process the BFS queue until max_nodes or max_depth is reached."""
        while queue and len(self.nodes) < self.max_nodes:
            title, depth = queue.popleft()

            if title in self.visited:
                continue
            self.visited.add(title)

            if depth > self.cfg.max_depth:
                continue

            # Gate: allowed-set membership or live category check
            if not self._is_accepted(title):
                continue

            html, canonical = self._fetch_and_resolve(title)
            if html is None:
                continue

            # Redirect: use canonical title as the node identifier
            if canonical != title:
                if canonical in self.nodes:
                    continue          # canonical already added via another path
                title = canonical
                self.visited.add(canonical)

            if title in self.nodes:
                continue

            categories = self._fetch_page_categories(title)
            self._add_node(title, categories)

            # Persist state periodically for resumability
            if len(self.nodes) % _STATE_SAVE_INTERVAL == 0:
                self._save_state()

            if depth >= self.cfg.max_depth:
                continue             # do not expand further — this is a leaf

            links = self._parse_links(html)
            # Store for edge recording (post-BFS) — drop self-links now
            self.outgoing_links[title] = [l for l in links if l != title]
            self._enqueue_links(links, depth + 1, queue, queued)

    # ══════════════════════════════════════════════════════════════════════
    # POST-BFS: EDGES & DEGREES
    # ══════════════════════════════════════════════════════════════════════

    def _record_edges(self) -> None:
        """Apply edge policy: u→v only if both u and v are accepted nodes."""
        node_set = set(self.nodes.keys())
        for src, targets in self.outgoing_links.items():
            if src not in node_set:
                continue
            for tgt in targets:
                if tgt in node_set and tgt != src:
                    self.edges.add((src, tgt))
        logger.info("Edges recorded: %d", len(self.edges))

    def _compute_degrees(self) -> None:
        """Fill in degree, in_degree, out_degree for every node."""
        in_deg: Dict[str, int] = {t: 0 for t in self.nodes}
        out_deg: Dict[str, int] = {t: 0 for t in self.nodes}
        for src, tgt in self.edges:
            out_deg[src] = out_deg.get(src, 0) + 1
            in_deg[tgt] = in_deg.get(tgt, 0) + 1
        for title, node in self.nodes.items():
            node["in_degree"] = in_deg[title]
            node["out_degree"] = out_deg[title]
            node["degree"] = node["in_degree"] + node["out_degree"]

    # ══════════════════════════════════════════════════════════════════════
    # RESUMABLE STATE
    # ══════════════════════════════════════════════════════════════════════

    def _save_state(self) -> None:
        """Persist nodes, edges, visited, and outgoing_links to disk."""
        state = {
            "nodes": self.nodes,
            "edges": list(self.edges),
            "visited": list(self.visited),
            "allowed_set": list(self.allowed_set),
            "allowed_categories": list(self.allowed_categories),
            "outgoing_links": self.outgoing_links,
            "stats": self.stats,
        }
        path = Path(self.cfg.state_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state), encoding="utf-8")
        logger.debug("State saved (%d nodes).", len(self.nodes))

    def _load_state(self) -> None:
        """Restore prior state if the state file exists."""
        path = Path(self.cfg.state_file)
        if not path.exists():
            return
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            self.nodes = state.get("nodes", {})
            self.edges = {tuple(e) for e in state.get("edges", [])}  # type: ignore[misc]
            self.visited = set(state.get("visited", []))
            self.allowed_set = set(state.get("allowed_set", []))
            self.allowed_categories = set(state.get("allowed_categories", []))
            self.outgoing_links = state.get("outgoing_links", {})
            self.stats.update(state.get("stats", {}))
            logger.info(
                "Resumed from state: %d nodes, %d edges.",
                len(self.nodes), len(self.edges),
            )
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Could not load state file (%s) — starting fresh.", exc)

    # ══════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ══════════════════════════════════════════════════════════════════════

    def crawl(self) -> None:
        """Run the full hybrid crawl (all phases)."""
        start = time.time()

        # Phase 1 — build allowed-set (skipped if already loaded from state)
        if not self.allowed_set:
            logger.info("Phase 1: building allowed-set …")
            self._build_allowed_set()
        else:
            logger.info("Phase 1: allowed-set already loaded (%d titles).",
                        len(self.allowed_set))

        # Phase 2 — BFS
        logger.info("Phase 2: BFS crawl starting …")
        queue, queued = self._initialize_queue()
        self._run_bfs(queue, queued)

        # Phase 3 — record edges & compute degrees
        logger.info("Phase 3: recording edges …")
        self._record_edges()
        self._compute_degrees()

        elapsed = round(time.time() - start, 2)
        self.stats.update({
            "total_runtime": elapsed,
            "nodes": len(self.nodes),
            "edges": len(self.edges),
        })

        # Final state flush
        self._save_state()
        logger.info(
            "Crawl complete: %d nodes, %d edges in %.1fs",
            len(self.nodes), len(self.edges), elapsed,
        )

    def save_outputs(self) -> None:
        """Write edgelist.csv, nodes.csv, and crawl_report.json."""
        # edgelist.csv ──────────────────────────────────────────────────
        with open(self.cfg.edgelist_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["source", "target"])
            for src, tgt in sorted(self.edges):
                writer.writerow([src, tgt])

        # nodes.csv ─────────────────────────────────────────────────────
        fields = ["id", "title", "url", "degree", "in_degree",
                  "out_degree", "node_type", "categories"]
        with open(self.cfg.nodes_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for node in sorted(self.nodes.values(), key=lambda n: n["id"]):
                writer.writerow(node)

        # crawl_report.json ─────────────────────────────────────────────
        with open(self.cfg.crawl_report_path, "w", encoding="utf-8") as fh:
            json.dump(self.stats, fh, indent=2)

        logger.info(
            "Outputs written → %s | %s | %s",
            self.cfg.edgelist_path, self.cfg.nodes_path, self.cfg.crawl_report_path,
        )
