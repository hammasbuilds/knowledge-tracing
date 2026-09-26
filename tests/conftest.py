from __future__ import annotations

import pytest
from helpers import TRUE_BKT

from kt.data import Log
from kt.synthetic import simulate_bkt


@pytest.fixture(scope="session")
def synth() -> Log:
    return simulate_bkt(TRUE_BKT, n_students=1200, attempts_per_skill=10, seed=1)


@pytest.fixture(scope="session")
def synth_test() -> Log:
    return simulate_bkt(TRUE_BKT, n_students=400, attempts_per_skill=10, seed=2)
