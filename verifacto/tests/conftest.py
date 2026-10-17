"""Shared pytest fixtures for the verifacto test suite.

Fixtures are added phase by phase: `graph` in Phase 1, `posts` in Phase 2,
`stub` in Phase 4. Everything is session-scoped so suites stay fast and the
whole suite is deterministic.
"""

import pytest
