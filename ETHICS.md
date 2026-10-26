# ETHICS.md — Data Collection Ethics Statement & Compliance

> **Project**: Influence Ranking in the Marvel Wikipedia Link Graph using PageRank and HITS  
> **Institution**: Sardar Patel College of Engineering (SPCE), Mumbai — B.Tech Computer Engineering  
> **Course**: Web Data Mining (Semester VII)  
> **Academic Year**: 2026 | Team of 4

---

## 1. Compliance with robots.txt & Technical Enforcement

The project strictly complies with the [Wikimedia robots policy](https://meta.wikimedia.org/wiki/Robots.txt) and parses the live `robots.txt` before issuing any HTTP request.

### Code Enforcement (`crawler/wiki_crawler.py`):
```python
def _load_robots(self) -> urllib.robotparser.RobotFileParser:
    """Load robots.txt from cache; fetch from network only on first run."""
    rp = urllib.robotparser.RobotFileParser()
    robots_cache = self.cache_dir / "_robots.txt"
    if robots_cache.exists():
        rp.parse(robots_cache.read_text(encoding="utf-8").splitlines())
    else:
        resp = self._get_with_retry(self.cfg.robots_url, is_api=True)
        robots_cache.write_text(resp.text, encoding="utf-8")
        rp.parse(resp.text.splitlines())
    return rp

def _can_fetch(self, url: str) -> bool:
    """Return True if robots.txt allows the crawler to fetch url."""
    return self.robot_parser.can_fetch(self.cfg.user_agent, url)
```

Before **every single article fetch**, `_can_fetch(url)` is invoked. Any disallowed path (e.g. `/w/index.php`, Special namespaces, search query endpoints) is immediately rejected, logged, and incremented in `stats["robots_skipped"]`. Disallowed URLs never enter the graph.

---

## 2. Identification & User-Agent Policy

All requests (both raw HTML document retrievals and MediaWiki API queries) declare a descriptive, transparent User-Agent header complying with Wikimedia’s [User-Agent policy](https://meta.wikimedia.org/wiki/User-Agent_policy):

```
MarvelGraphRank/1.0 (Academic mini-project, SPCE Mumbai, Web Data Mining Sem VII; contact: team@example.com; github.com/example/MarvelGraphRank)
```

This header explicitly declares:
1. Application name and version (`MarvelGraphRank/1.0`).
2. Academic institution and course (`SPCE Mumbai, Web Data Mining Sem VII`).
3. Point of contact email (`team@example.com`).
4. Public project repository URL.

This ensures Wikimedia system operators can easily identify, monitor, or throttle our traffic if necessary.

---

## 3. Polite Rate Limiting & Exponential Back-Off

To minimize load on Wikimedia production servers, our crawler implements polite, randomized spacing and exponential back-off:

1. **Polite Random Delays**: Between consecutive requests, `_sleep()` introduces a uniformly distributed random sleep interval $U(delay\_min, delay\_max)$ (default $1.0 - 1.5$ seconds in `config.yaml`). Unseeded randomization prevents synchronous traffic bursts.
2. **HTTP 429 / 503 Handling**: If a `429 Too Many Requests` or `503 Service Unavailable` status is encountered:
   - The crawler inspects the response headers for a `Retry-After` directive and respects it.
   - If absent, it applies exponential back-off: $wait = \min(backoff\_base^{attempt+1}, 120)$ seconds.
   - The crawler logs the backoff event and re-attempts up to `max_retries` (default 5). If retries are exhausted, the URL is skipped gracefully without crashing the crawl.

---

## 4. Local Caching & Offline Reproducibility

To prevent repetitive network load during development, debugging, and grading:
- **Raw HTML Pages**: Saved to `data/cache/{title}_{hash}.json`.
- **MediaWiki API Responses**: Saved to `data/cache/api_{param_hash}.json`.
- **Zero Duplicate Requests**: Before querying the network, `_cache_key()` checks for an existing local file. On subsequent runs, cache hits bypass the network entirely, resulting in **zero network traffic**.
- **Grading Mode**: Evaluation commands (`python main.py build`, `python main.py rank`, `python main.py report`, and `python main.py all --offline`) run completely disconnected from the internet.

---

## 5. Data Minimization & Privacy Protection

The crawler strictly adheres to data minimization principles:
- **Public Articles Only**: The crawler targets the main article namespace (`ns=0`). All user profile pages (`User:`), user discussion threads (`Talk:`, `User talk:`), edit histories, IP addresses, and user-generated metadata are excluded via `dropped_namespaces`.
- **Structural Content Only**: Text bodies are analyzed solely to extract internal link references (`/wiki/...`). No personal data, cookies, authentication credentials, or external session states are captured or retained.
- **Category Truncation**: Only the top category labels needed for semantic node-type classification (`character`, `film`, `comic_series`, `list`) are retained in `data/nodes.csv`.

---

## 6. Intellectual Property & CC BY-SA 4.0 Attribution

All text, category metadata, and hyperlinks harvested from English Wikipedia (`en.wikipedia.org`) are the intellectual property of the Wikimedia Foundation and its volunteer contributors.

- **License**: Content is utilized under the **Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0)**.
- **Academic Fair Use**: Data collection is conducted strictly for educational, non-commercial purposes within the SPCE Mumbai Web Data Mining curriculum.
- **No Redistribution**: The raw article contents are retained solely in local ephemeral cache files and will not be redistributed commercially or published as standalone text corpora.

---

## 7. High-Volume Alternatives: MediaWiki API & Wikimedia Dumps

While our crawler uses a hybrid approach for dynamic graph discovery (categorymembers API + bounded HTML BFS), production-scale data mining projects should evaluate Wikimedia's high-efficiency bulk distribution channels:

1. **Wikimedia Enterprise & Public Dumps (`dumps.wikimedia.org`)**:
   - Wikimedia provides monthly compressed XML dumps (`pages-articles.xml.bz2`) and SQL table dumps (`pagelinks.sql.gz`).
   - *Advantage*: Downloading dumps eliminates live HTTP requests entirely, allowing offline SQL parsing of millions of links in minutes without server strain.
2. **MediaWiki API (`/w/api.php`)**:
   - For category expansion, our crawler uses `list=categorymembers` and `prop=categories` in batches of up to 500 members per call, significantly reducing the request count compared to scraping category HTML.

---

## 8. How to Pause, Stop, and Resume Crawling

The crawler is engineered to be crash-resilient and resumable:

- **Graceful Termination**: Pressing `Ctrl + C` (SIGINT) causes Python to raise a standard interrupt. Because state is flushed every 10 nodes, crawled data up to the last interval is safely written to disk.
- **Resumable State File (`data/cache/_crawl_state.json`)**:
  - Automatically records the set of visited articles, the dictionary of nodes, the queue frontier, and the outgoing link mapping.
  - When `python main.py crawl` is re-launched, `_load_state()` inspects `data/cache/_crawl_state.json`. If present, it loads previously accepted nodes, pre-populates the queue, and resumes seamlessly without re-fetching existing pages.
- **Clean Reset**: To wipe all state and restart crawling from seed definitions:
  ```bash
  rm -f data/cache/_crawl_state.json data/edgelist.csv data/nodes.csv
  ```

---

## 9. Statement on Responsible Use

The code and models in this repository are designed strictly for academic research into network topology, eigenvector centrality, and knowledge graph structure. They must not be adapted for aggressive scraping, denial-of-service stress testing, or unauthorized mass extraction of Wikimedia resources. Users of this code are expected to abide by the [Wikimedia Terms of Use](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use).
