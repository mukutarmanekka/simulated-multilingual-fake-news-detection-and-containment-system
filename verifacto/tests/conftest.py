"""Shared pytest fixtures for the verifacto test suite.

Fixtures are added phase by phase: `graph` (Phase 1), `posts` (Phase 2),
`stub` (Phase 4). Everything is session-scoped so suites stay fast and the
whole suite is deterministic.
"""

import pytest

from verifacto.data.posts import generate_posts
from verifacto.network.graph_builder import build_language_graph


@pytest.fixture(scope="session")
def graph():
    return build_language_graph()


@pytest.fixture(scope="session")
def posts(graph):
    return generate_posts(graph)
