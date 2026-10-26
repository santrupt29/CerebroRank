"""tests/test_app.py — Automated verification for Streamlit dashboard (app.py)."""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

from core.config import load_config
APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture(scope="module")
def app_runner():
    """Run app.py and return the AppTest element tree."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    assert not at.exception, f"App execution raised exception: {at.exception}"
    return at


def test_app_starts_and_renders_seven_tabs(app_runner: AppTest):
    """Verify that all 7 tabs render properly."""
    assert len(app_runner.tabs) == 7
    expected_tab_keywords = [
        "Overview",
        "Rankings",
        "Node Explorer",
        "Network",
        "Algorithm Comparison",
        "Toy Graph",
        "About",
    ]
    for tab, kw in zip(app_runner.tabs, expected_tab_keywords):
        assert kw.lower() in tab.label.lower()


def test_search_works_for_three_example_titles():
    """Verify autocomplete search for at least 3 distinct titles."""
    test_titles = ["Spider-Man", "Iron Man", "Marvel Comics"]

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    assert not at.exception

    for title in test_titles:
        node_selector = at.selectbox[0]
        assert title in node_selector.options, f"{title} not found in selectbox options"
        node_selector.set_value(title).run()
        assert not at.exception, f"Failed when searching for {title}"


def test_personalized_pagerank_button_execution():
    """Verify the 'Personalized PageRank from this node' action."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    assert not at.exception

    ppr_button = at.button[0]
    assert "Personalized PageRank" in ppr_button.label
    ppr_button.click().run()
    assert not at.exception


def test_damping_slider_recomputes_live():
    """Verify that shifting the damping slider recomputes rankings without crashing."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    assert not at.exception

    d_slider = at.sidebar.slider[0]
    assert "Damping" in d_slider.label
    d_slider.set_value(0.70).run()
    assert not at.exception


def test_toy_graph_iteration():
    """Verify that the toy graph playground updates probability vector on step advance."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    assert not at.exception

    step_slider = next(s for s in at.slider if "Step" in s.label)
    assert "Step" in step_slider.label
    step_slider.set_value(3).run()
    assert not at.exception
