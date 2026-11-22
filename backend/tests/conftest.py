"""Shared fixtures for the simulator test-suites (ARCHITECTURE §8).

* ``toy_graph`` — the tiny two-division network from :func:`sim.testing.make_toy_graph`
  (exact ``network.json`` schema); every physics/mask unit test runs on it.
* ``real_graph`` — ``data/network.json`` (WP1). Tests that need the real network skip
  cleanly while the artifact does not exist yet.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from sim.graph import Graph, data_dir, load_graph
from sim.testing import make_toy_graph


def network_path() -> Path:
    return data_dir() / "network.json"


def have_real_network() -> bool:
    return network_path().is_file()


@pytest.fixture(scope="session")
def toy_graph() -> Graph:
    return make_toy_graph()


@pytest.fixture(scope="session")
def real_graph() -> Graph:
    if not have_real_network():
        pytest.skip(f"{network_path()} not built yet (WP1)")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_graph(network_path())


@pytest.fixture(scope="session", params=["toy", "real"])
def any_graph(request: pytest.FixtureRequest) -> Graph:
    """The toy graph and (when present) the real network."""
    if request.param == "toy":
        return request.getfixturevalue("toy_graph")
    return request.getfixturevalue("real_graph")
