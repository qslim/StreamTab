"""Rolling labeled table with regime-conditioned negatives."""
from bisect import bisect_right
from collections import deque

import numpy as np


class ContextTable:
    def __init__(self, state, regime, k, g, seed):
        self.state = state
        self.regime = regime
        self.g = g
        self.rows = deque(maxlen=k)
        self.rng = np.random.RandomState(seed)
        self.cutoff = 0

    def _draw_pair(self, start, u, v):
        pairs = self.state.seen_pairs
        pool = len(pairs) - start
        if pool > 0:
            for _ in range(8):
                pair = pairs[start + self.rng.randint(pool)]
                if pair != (u, v):
                    return pair
        return None

    def sample_negative(self, u, v):
        """Inductive falls back to historical, and historical to random."""
        state = self.state
        pair = None
        if self.regime == 'inductive':
            pair = self._draw_pair(bisect_right(state.pair_first_ordinals, self.cutoff), u, v)
        if pair is None and self.regime != 'random':
            pair = self._draw_pair(0, u, v)
        if pair is None:
            pair = u, state.seen_dst[self.rng.randint(len(state.seen_dst))]
        return pair

    def add(self, u, v, t):
        """Append one positive row and its g negative rows."""
        self.rows.append((self.state.features(u, v, t), 1))
        if self.state.seen_dst:  # the first batch has no history to sample from
            for _ in range(self.g):
                nu, nv = self.sample_negative(u, v)
                self.rows.append((self.state.features(nu, nv, t), 0))

    def snapshot(self):
        rows, labels = zip(*self.rows)
        return np.stack(rows), np.array(labels)
