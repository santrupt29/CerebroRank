"""main.py — CLI entry point for MarvelGraphRank.

Commands
--------
  crawl   Run the ethical hybrid Wikipedia crawler (Stage 1)
  build   Build graph + quality report from crawled CSVs (Stage 2)
  rank    Run PageRank + HITS, validate, save scores.csv (Stage 2)
  report  Generate visualisations and reports (Stage 4 — coming)
  all     Run all stages in sequence (coming)

Usage
-----
  python main.py crawl [--pilot] [--config PATH]
  python main.py build [--config PATH]
  python main.py rank  [--config PATH]
"""

import argparse
import logging
import sys
from pathlib import Path


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def _print_table(title: str, rows: list, headers: list) -> None:
    """Print a simple fixed-width ASCII table."""
    col_w = [max(len(str(headers[i])), max(len(str(r[i])) for r in rows)) + 2
             for i in range(len(headers))]
    sep = "─" * sum(col_w)
    print(f"\n  {title}")
    print("  " + sep)
    print("  " + "".join(f"{str(h):<{col_w[i]}}" for i, h in enumerate(headers)))
    print("  " + sep)
    for row in rows:
        print("  " + "".join(f"{str(v):<{col_w[i]}}" for i, v in enumerate(row)))
    print("  " + sep)


# ══════════════════════════════════════════════════════════════════════════════
# CRAWL
# ══════════════════════════════════════════════════════════════════════════════

def cmd_crawl(args: argparse.Namespace) -> None:
    from core.config import load_config
    from crawler.wiki_crawler import WikiCrawler
    cfg_path = Path(args.config) if args.config else None
    cfg = load_config(cfg_path) if cfg_path else load_config()
    crawler = WikiCrawler(cfg, pilot=args.pilot)
    crawler.crawl()
    crawler.save_outputs()
    width = 60
    print("\n" + "═" * width + "\n  CRAWL SUMMARY\n" + "═" * width)
    for k, v in crawler.stats.items():
        print(f"  {k.replace('_',' ').capitalize():<30} {v}")
    print("═" * width + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# BUILD
# ══════════════════════════════════════════════════════════════════════════════

def cmd_build(args: argparse.Namespace) -> None:
    """Build the graph and save the data-quality report."""
    from core.config import load_config
    from core.graph_builder import load_graph, save_quality_report
    cfg_path = Path(args.config) if args.config else None
    cfg = load_config(cfg_path) if cfg_path else load_config()

    gd = load_graph(cfg)
    save_quality_report(gd.quality_report, cfg)

    print("\n  DATA QUALITY REPORT")
    print("  " + "─" * 40)
    for k, v in gd.quality_report.items():
        print(f"  {k.replace('_',' ').capitalize():<35} {v}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# RANK
# ══════════════════════════════════════════════════════════════════════════════

def cmd_rank(args: argparse.Namespace) -> None:
    """Run PageRank + HITS, validate, save scores.csv, print top-10."""
    import pandas as pd
    from core.algorithms import hits, pagerank
    from core.config import load_config
    from core.graph_builder import load_graph
    from core.validation import save_validation_report, validate_all

    cfg_path = Path(args.config) if args.config else None
    cfg = load_config(cfg_path) if cfg_path else load_config()

    # Build graph
    gd = load_graph(cfg)

    # Run algorithms
    pr = pagerank(
        gd.M, gd.dangling_mask,
        d=cfg.pagerank_d, tol=cfg.pagerank_tol, max_iter=cfg.pagerank_max_iter,
    )
    hits_res = hits(gd.A, tol=cfg.hits_tol, max_iter=cfg.hits_max_iter)

    # Validate against NetworkX
    val = validate_all(gd, pr, hits_res, d=cfg.pagerank_d, tol=cfg.pagerank_tol)
    save_validation_report(val, cfg.validation_path)

    _print_validation_summary(val)

    # Build scores DataFrame
    import numpy as np
    records = [
        {
            "id": idx,
            "title": gd.id_to_title[idx],
            "pagerank": round(float(pr.scores[idx]), 8),
            "authority": round(float(hits_res.authority[idx]), 8),
            "hub": round(float(hits_res.hub[idx]), 8),
        }
        for idx in range(gd.N)
    ]
    scores_df = pd.DataFrame(records)

    # Merge with nodes.csv metadata if available
    node_path = Path(cfg.nodes_path)
    if node_path.exists():
        nodes_df = pd.read_csv(node_path, dtype=str)
        scores_df = nodes_df.merge(
            scores_df[["title", "pagerank", "authority", "hub"]],
            on="title", how="right",
        )

    scores_path = Path(cfg.scores_path)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    scores_df.to_csv(scores_path, index=False)
    print(f"\n  Scores saved → {scores_path}\n")

    # Print top-10 by each metric
    _print_top10(scores_df, "pagerank", "Top-10 by PageRank")
    _print_top10(scores_df, "authority", "Top-10 by HITS Authority")
    _print_top10(scores_df, "hub", "Top-10 by HITS Hub")


def _print_validation_summary(val: dict) -> None:
    pr = val["pagerank"]
    ah = val["hits"]["authority"]
    hh = val["hits"]["hub"]
    print("\n  VALIDATION vs NetworkX")
    print("  " + "─" * 55)
    print(f"  {'Metric':<30} {'PageRank':>10} {'Auth':>8} {'Hub':>8}")
    print("  " + "─" * 55)
    print(f"  {'MSE':<30} {pr['mse']:>10.2e} {ah['mse']:>8.2e} {hh['mse']:>8.2e}")
    print(f"  {'Spearman rho':<30} {pr['spearman_rho']:>10.4f} {ah['spearman_rho']:>8.4f} {hh['spearman_rho']:>8.4f}")
    print(f"  {'Kendall tau':<30} {pr['kendall_tau']:>10.4f} {ah['kendall_tau']:>8.4f} {hh['kendall_tau']:>8.4f}")
    print(f"  {'Top-10 overlap':<30} {pr['top10_overlap']:>10} {ah['top10_overlap']:>8} {hh['top10_overlap']:>8}")
    print()


def _print_top10(df, col: str, heading: str) -> None:
    """Print a ranked table of the top-10 nodes by *col*."""
    import pandas as pd
    top = df.nlargest(10, col)[["title", col]].reset_index(drop=True)
    rows = [(i + 1, row["title"][:55], f"{float(row[col]):.6f}")
            for i, row in top.iterrows()]
    _print_table(heading, rows, ["Rank", "Title", col.capitalize()])


# ══════════════════════════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════════════════════════

def cmd_report(args: argparse.Namespace) -> None:
    """Run Stage 3 evaluation: produce 9 figures, CSV tables, and summary.md."""
    from core.config import load_config
    from core.evaluator import run_evaluation

    cfg_path = Path(args.config) if args.config else None
    cfg = load_config(cfg_path) if cfg_path else load_config()

    print("\n" + "═" * 60)
    print("  RUNNING MARVELGRAPHRANK STAGE 3 EVALUATION & REPORTING")
    print("═" * 60 + "\n")

    manifest = run_evaluation(cfg)

    print("  FIGURES GENERATED (reports/figures/ at 300 DPI):")
    print("  " + "─" * 56)
    for fig in manifest["figures"]:
        print(f"    • {fig}")

    print("\n  UNDERLYING TABLES SAVED (reports/tables/ as CSV):")
    print("  " + "─" * 56)
    for tbl in manifest["tables"]:
        print(f"    • {tbl}")

    print("\n  EXECUTIVE SUMMARY (reports/summary.md):")
    print("  " + "─" * 56)
    print(f"    • {manifest['summary']}")

    print("\n" + "═" * 60)
    print(f"  Stage 3 complete — {len(manifest['generated_files'])} artifacts produced successfully.")
    print("═" * 60 + "\n")


def cmd_all(args: argparse.Namespace) -> None:
    """Run build -> rank -> report sequentially."""
    print("\n" + "═" * 60)
    print("  MARVELGRAPHRANK — FULL PIPELINE (BUILD -> RANK -> REPORT)")
    print("═" * 60 + "\n")
    cmd_build(args)
    cmd_rank(args)
    cmd_report(args)

# ══════════════════════════════════════════════════════════════════════════════
# STUBS
# ══════════════════════════════════════════════════════════════════════════════

def _stub(name: str) -> None:
    print(f"  [{name}] — will be implemented in a future stage.")
# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    _setup_logging()

    parser = argparse.ArgumentParser(
        prog="main.py",
        description="MarvelGraphRank — Marvel Wikipedia link-graph analysis",
    )
    parser.add_argument("--config", metavar="PATH", default=None,
                        help="Path to an alternative config.yaml")
    parser.add_argument("--offline", action="store_true", default=False,
                        help="Operate in offline mode using pre-generated data")
    sub = parser.add_subparsers(dest="command", required=True)

    # crawl
    p_crawl = sub.add_parser("crawl", help="Run the ethical hybrid crawler")
    p_crawl.add_argument("--pilot", action="store_true",
                         help="Crawl ~30 nodes for quick verification")
    p_crawl.set_defaults(func=cmd_crawl)

    # build
    p_build = sub.add_parser("build", help="Build graph + quality report")
    p_build.set_defaults(func=cmd_build)

    # rank
    p_rank = sub.add_parser("rank", help="Run PageRank+HITS, save scores.csv")
    p_rank.set_defaults(func=cmd_rank)

    # report
    p_report = sub.add_parser("report", help="Run Stage 3 evaluation and produce figures, tables, summary.md")
    p_report.set_defaults(func=cmd_report)

    # all
    p_all = sub.add_parser("all", help="Run build -> rank -> report sequentially")
    p_all.add_argument("--offline", action="store_true", default=False,
                       help="Operate in offline mode using pre-generated data")
    p_all.set_defaults(func=cmd_all)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
