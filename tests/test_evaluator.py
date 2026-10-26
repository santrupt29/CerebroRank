"""tests/test_evaluator.py — Unit tests for Stage 3 evaluation and reporting module."""

from pathlib import Path
import pandas as pd
import pytest

from core.algorithms import hits, pagerank
from core.config import Config, load_config
from core.evaluator import (
    generate_interactive_network,
    generate_summary_markdown,
    plot_bias_limitations,
    plot_community_detection,
    plot_convergence_curves,
    plot_damping_sensitivity,
    plot_hubs_authorities_pagerank,
    plot_metric_comparison,
    plot_node_type_analysis,
    plot_personalized_pagerank,
    run_evaluation,
)
from core.graph_builder import GraphData, load_graph
from core.validation import validate_all


@pytest.fixture(scope="module")
def cfg() -> Config:
    return load_config()


@pytest.fixture(scope="module")
def gd(cfg: Config) -> GraphData:
    return load_graph(cfg)


@pytest.fixture(scope="module")
def pr_res(gd: GraphData, cfg: Config):
    return pagerank(gd.M, gd.dangling_mask, d=cfg.pagerank_d, tol=cfg.pagerank_tol, max_iter=cfg.pagerank_max_iter)


@pytest.fixture(scope="module")
def hits_res(gd: GraphData, cfg: Config):
    return hits(gd.A, tol=cfg.hits_tol, max_iter=cfg.hits_max_iter)


@pytest.fixture(scope="module")
def scores_df(cfg: Config) -> pd.DataFrame:
    p = Path(cfg.scores_path)
    if p.exists():
        return pd.read_csv(p)
    pytest.skip("scores.csv not found")


class TestFigure1ConvergenceCurves:
    def test_generates_figure_and_csv(self, pr_res, hits_res, tmp_path: Path):
        fig_path = tmp_path / "01_convergence.png"
        csv_path = tmp_path / "01_convergence.csv"
        out_fig = plot_convergence_curves(pr_res, hits_res, fig_path, csv_path)

        assert out_fig.exists()
        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        assert "iteration" in df.columns
        assert "pagerank_l1_residual" in df.columns
        assert "hits_authority_l1_residual" in df.columns
        assert len(df) == max(len(pr_res.residuals), len(hits_res.convergence))


class TestFigure2DampingSensitivity:
    def test_generates_damping_artifacts(self, gd: GraphData, tmp_path: Path):
        fig_path = tmp_path / "02_damping.png"
        plot_damping_sensitivity(gd, [0.5, 0.7, 0.85, 0.95], 1e-6, 100, fig_path, tmp_path)

        assert fig_path.exists()
        mat_csv = tmp_path / "02_damping_spearman_matrix.csv"
        ranks_csv = tmp_path / "02_damping_top10_ranks.csv"
        iters_csv = tmp_path / "02_damping_iterations.csv"

        assert mat_csv.exists()
        assert ranks_csv.exists()
        assert iters_csv.exists()

        iters_df = pd.read_csv(iters_csv)
        assert len(iters_df) == 4
        # Higher damping factor should require more or equal iterations
        assert iters_df.loc[iters_df["damping_factor"] == 0.95, "iterations_to_converge"].values[0] >= \
               iters_df.loc[iters_df["damping_factor"] == 0.5, "iterations_to_converge"].values[0]


class TestFigure3MetricComparison:
    def test_generates_metric_correlation_tables(self, scores_df: pd.DataFrame, tmp_path: Path):
        fig_path = tmp_path / "03_comparison.png"
        plot_metric_comparison(scores_df, fig_path, tmp_path)

        assert fig_path.exists()
        assert (tmp_path / "03_metric_spearman_correlation.csv").exists()
        assert (tmp_path / "03_metric_kendall_correlation.csv").exists()
        assert (tmp_path / "03_metric_jaccard_overlap.csv").exists()

        jaccard_df = pd.read_csv(tmp_path / "03_metric_jaccard_overlap.csv")
        assert "Jaccard_at_10" in jaccard_df.columns
        assert "Jaccard_at_20" in jaccard_df.columns


class TestFigure4HubsAuthoritiesPageRank:
    def test_generates_scatter_and_archetypes(self, gd: GraphData, scores_df: pd.DataFrame, tmp_path: Path):
        fig_path = tmp_path / "04_hubs_authorities.png"
        out_fig, archetypes = plot_hubs_authorities_pagerank(gd, scores_df, fig_path, tmp_path)

        assert out_fig.exists()
        assert (tmp_path / "04_top15_combined_rankings.csv").exists()
        assert (tmp_path / "04_hub_authority_pagerank_scatter.csv").exists()
        assert (tmp_path / "04_archetype_examples.csv").exists()

        assert "high_hub" in archetypes
        assert "high_auth" in archetypes
        assert "negative_result" in archetypes
        assert len(archetypes["neg_explanation"]) > 20


class TestFigure5PersonalizedPageRank:
    def test_generates_personalized_comparison(self, gd: GraphData, scores_df: pd.DataFrame, tmp_path: Path):
        fig_path = tmp_path / "05_ppr.png"
        table_path = tmp_path / "05_ppr.csv"
        plot_personalized_pagerank(gd, scores_df, 1e-6, 100, fig_path, table_path)

        assert fig_path.exists()
        assert table_path.exists()
        df = pd.read_csv(table_path)
        assert len(df) == 10
        assert "Global_Title" in df.columns
        assert "MCU_Title" in df.columns
        assert "XMen_Title" in df.columns


class TestFigure6CommunityDetection:
    def test_generates_community_table_and_plot(self, gd: GraphData, scores_df: pd.DataFrame, tmp_path: Path):
        fig_path = tmp_path / "06_comm.png"
        table_path = tmp_path / "06_comm.csv"
        _, comm_df = plot_community_detection(gd, scores_df, fig_path, table_path)

        assert fig_path.exists()
        assert table_path.exists()
        assert len(comm_df) > 0
        assert "Size" in comm_df.columns
        assert "Dominant_Type" in comm_df.columns


class TestFigure7NodeTypeAnalysis:
    def test_generates_node_type_table_and_plot(self, scores_df: pd.DataFrame, tmp_path: Path):
        fig_path = tmp_path / "07_node_type.png"
        table_path = tmp_path / "07_node_type.csv"
        plot_node_type_analysis(scores_df, fig_path, table_path)

        assert fig_path.exists()
        assert table_path.exists()
        df = pd.read_csv(table_path)
        assert "node_type" in df.columns
        assert "pagerank_mean" in df.columns


class TestFigure8InteractiveNetwork:
    def test_generates_html_and_tables(self, gd: GraphData, scores_df: pd.DataFrame, tmp_path: Path):
        html_path = tmp_path / "network.html"
        generate_interactive_network(gd, scores_df, 75, "authority", html_path, tmp_path)

        assert html_path.exists()
        assert (tmp_path / "08_network_nodes.csv").exists()
        assert (tmp_path / "08_network_edges.csv").exists()

        html_content = html_path.read_text(encoding="utf-8")
        assert "vis" in html_content.lower()


class TestFigure9BiasLimitations:
    def test_generates_bias_artifacts(self, gd: GraphData, scores_df: pd.DataFrame, tmp_path: Path):
        fig_path = tmp_path / "09_bias.png"
        table_path = tmp_path / "09_bias.csv"
        _, summary = plot_bias_limitations(gd, scores_df, 1e-6, 100, fig_path, table_path)

        assert fig_path.exists()
        assert table_path.exists()
        assert len(summary) > 20
        df = pd.read_csv(table_path)
        assert len(df) == 10


class TestExecutiveSummaryAndPipeline:
    def test_run_evaluation_complete_manifest(self, cfg: Config, gd: GraphData, pr_res, hits_res, scores_df):
        val = validate_all(gd, pr_res, hits_res, d=cfg.pagerank_d, tol=cfg.pagerank_tol)
        manifest = run_evaluation(cfg, gd=gd, pr_res=pr_res, hits_res=hits_res, scores_df=scores_df, val_report=val)

        assert len(manifest["figures"]) == 9
        assert len(manifest["tables"]) == 16
        assert Path(manifest["summary"]).exists()
        for f in manifest["generated_files"]:
            assert Path(f).exists()
