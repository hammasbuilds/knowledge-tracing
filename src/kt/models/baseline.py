"""History-free baseline: how hard is this item, ignoring who answers it.

A smoothed item correct-rate shrunk towards its skill's rate, which is shrunk
towards the global rate. Any model that cannot beat this is not using the
student's history.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..data import Log


@dataclass
class ItemMean:
    strength: float = 5.0  # pseudo-counts pulling towards the parent rate
    name: str = "ItemMean"
    global_rate: float = 0.5
    skill_rate: np.ndarray = field(default_factory=lambda: np.zeros(0))
    item_rate: np.ndarray = field(default_factory=lambda: np.zeros(0))
    item_seen: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))

    def fit(self, train: Log, val: Log | None = None) -> ItemMean:
        y = train.correct.astype(np.float64)
        self.global_rate = float(y.mean())
        k = self.strength
        s_n = np.bincount(train.skill, minlength=train.n_skills)
        s_c = np.bincount(train.skill, weights=y, minlength=train.n_skills)
        self.skill_rate = (s_c + k * self.global_rate) / (s_n + k)
        i_n = np.bincount(train.item, minlength=train.n_items)
        i_c = np.bincount(train.item, weights=y, minlength=train.n_items)
        # an item's parent rate is that of its lowest-id skill (collapsed logs have one)
        parent = np.full(train.n_items, self.global_rate)
        order = np.lexsort((train.skill, train.item))
        first = np.ones(len(order), dtype=bool)
        first[1:] = train.item[order][1:] != train.item[order][:-1]
        parent[train.item[order][first]] = self.skill_rate[train.skill[order][first]]
        self.item_rate = (i_c + k * parent) / (i_n + k)
        self.item_seen = i_n > 0
        return self

    def predict(self, log: Log, horizon: int = 1) -> np.ndarray:
        out = np.full(len(log), self.global_rate)
        ok_s = log.skill < len(self.skill_rate)
        out[ok_s] = self.skill_rate[log.skill[ok_s]]
        ok_i = log.item < len(self.item_rate)
        seen = np.zeros(len(log), dtype=bool)
        seen[ok_i] = self.item_seen[log.item[ok_i]]
        out[seen] = self.item_rate[log.item[seen]]
        return out
