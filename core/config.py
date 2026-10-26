"""core/config.py — typed configuration loader.

Design decision: we load config.yaml into a dataclass (not a dict) so
every module gets IDE-friendly, typed access to settings.  The loader
filters unknown YAML keys, so adding a comment-only key to config.yaml
never breaks the code.

Stage 2 fields are given defaults so Stage 1 tests that construct Config
manually remain valid without specifying every new key.
"""

import dataclasses
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


@dataclass
class Config:
    """Typed mirror of config.yaml.  Add a field here AND in the YAML together."""

    # ── Stage 1: Crawl seeds & scope ────────────────────────────────────────
    seeds: List[str]
    category_roots: List[str]
    category_depth: int

    # BFS limits
    max_depth: int
    max_nodes: int
    max_new_nodes_per_page: int

    # Rate-limiting
    delay_min: float
    delay_max: float
    max_retries: int
    backoff_base: int

    # Identity
    user_agent: str

    # Endpoints
    wiki_base_url: str
    api_endpoint: str
    robots_url: str

    # Paths (Stage 1)
    cache_dir: str
    edgelist_path: str
    nodes_path: str
    crawl_report_path: str
    state_file: str

    # Parsing filters
    dropped_namespaces: List[str]
    removed_selectors: List[str]
    dropped_sections: List[str]

    # Node-type classification
    node_type_rules: Dict[str, List[str]] = field(default_factory=dict)

    # ── Stage 2: Graph builder ───────────────────────────────────────────────
    keep_giant_wcc: bool = True

    # Algorithm hyper-parameters
    pagerank_d: float = 0.85
    pagerank_tol: float = 1e-6
    pagerank_max_iter: int = 100
    hits_tol: float = 1e-6
    hits_max_iter: int = 100

    # Output paths (Stage 2)
    scores_path: str = "data/scores.csv"
    data_quality_path: str = "reports/data_quality.json"
    validation_path: str = "reports/validation.json"

    # ── Stage 3: Evaluation & Reports ─────────────────────────────────────────
    figures_dir: str = "reports/figures"
    tables_dir: str = "reports/tables"
    summary_path: str = "reports/summary.md"
    network_top_n: int = 75
    network_color_by: str = "authority"
    damping_factors: List[float] = field(default_factory=lambda: [0.5, 0.7, 0.85, 0.95])

def load_config(path: Path = _DEFAULT_CONFIG_PATH) -> "Config":
    """Load *path* (YAML) and return a :class:`Config`.

    Unknown YAML keys are silently ignored so the YAML can carry comments
    and extra annotations without crashing the loader.
    """
    with open(path, "r", encoding="utf-8") as fh:
        raw: dict = yaml.safe_load(fh)

    known = {f.name for f in dataclasses.fields(Config)}
    filtered = {k: v for k, v in raw.items() if k in known}

    # Check that every required field (no default) is present
    required = {
        f.name
        for f in dataclasses.fields(Config)
        if f.default is dataclasses.MISSING
        and f.default_factory is dataclasses.MISSING  # type: ignore[misc]
    }
    missing = required - filtered.keys()
    if missing:
        raise ValueError(f"config.yaml is missing required keys: {missing}")

    cfg = Config(**filtered)
    logger.info(
        "Config loaded from %s (max_nodes=%d, max_depth=%d, pr_d=%.2f)",
        path, cfg.max_nodes, cfg.max_depth, cfg.pagerank_d,
    )
    return cfg
