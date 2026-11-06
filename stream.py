"""Interaction history and the 34 candidate features."""
from bisect import bisect_left
from collections import defaultdict, deque

import numpy as np

PAIR_WINDOWS = (1_000, 10_000)
NODE_WINDOWS = (100, 1_000, 10_000)
RECENT_PARTNERS = 16
RECENT_NEIGHBORS = 32
RECENT_PAIR_GAPS = 8
EPSILON = 1e-9
RHO_MAX = 1e6

FEATURE_NAMES = [
    # pair
    'p_dt_last', 'p_cnt', 'p_cnt_w1k', 'p_cnt_w10k', 'p_age', 'p_gap_mean', 'p_due',
    # source
    's_dt_last', 's_cnt', 's_cnt_w100', 's_cnt_w1k', 's_cnt_w10k', 's_deg',
    's_new_rate', 's_gap_mean', 's_gap_std',
    # destination
    'd_dt_last', 'd_cnt', 'd_cnt_w100', 'd_cnt_w1k', 'd_cnt_w10k', 'd_deg',
    'd_new_rate', 'd_gap_mean', 'd_gap_std',
    # popularity
    'd_pop_w1k', 'd_pop_w10k', 'd_pop_trend', 's_act_w1k',
    # neighborhood
    'x_v_in_u_recent', 'x_u_in_v_recent', 'x_cn', 'x_jaccard',
    # global
    'g_gap',
]
N_FEATURES = len(FEATURE_NAMES)


class NodeHistory:
    """History of one node in one role (source or destination)."""

    def __init__(self):
        self.ordinals = []
        self.partners = deque(maxlen=RECENT_PARTNERS)
        self.novel = deque(maxlen=RECENT_PARTNERS)
        self.times = deque(maxlen=RECENT_PARTNERS + 1)
        self.neighbors = deque(maxlen=RECENT_NEIGHBORS)
        self.partner_count = 0

    def add(self, partner, ordinal, t, novel):
        self.ordinals.append(ordinal)
        self.partners.append(partner)
        self.novel.append(float(novel))
        self.times.append(t)
        self.neighbors.append(partner)
        self.partner_count += novel


class StreamState:
    def __init__(self):
        self.n = 0
        self.src = defaultdict(NodeHistory)
        self.dst = defaultdict(NodeHistory)
        self.pair_ordinals = {}
        self.pair_times = {}
        self.seen_pairs = []
        self.pair_first_ordinals = []
        self.seen_dst = []
        self.last_t = np.nan

    def ingest(self, u, v, t):
        pair = (u, v)
        novel = pair not in self.pair_ordinals
        if novel:
            self.pair_ordinals[pair] = []
            self.pair_times[pair] = []
            self.seen_pairs.append(pair)
            self.pair_first_ordinals.append(self.n)
        if v not in self.dst:
            self.seen_dst.append(v)
        self.pair_ordinals[pair].append(self.n)
        self.pair_times[pair].append(t)
        self.src[u].add(v, self.n, t, novel)
        self.dst[v].add(u, self.n, t, novel)
        self.last_t = t
        self.n += 1

    def _count(self, ordinals, window):
        return len(ordinals) - bisect_left(ordinals, self.n - window)

    def _share(self, node, window):
        return self._count(node.ordinals, window) / min(self.n, window)

    def _node_features(self, node, t):
        x = np.full(9, np.nan)
        if node is None:
            return x
        x[0] = np.log1p(t - node.times[-1])
        x[1] = np.log1p(len(node.ordinals))
        x[2] = self._count(node.ordinals, NODE_WINDOWS[0])
        x[3] = np.log1p(self._count(node.ordinals, NODE_WINDOWS[1]))
        x[4] = np.log1p(self._count(node.ordinals, NODE_WINDOWS[2]))
        x[5] = np.log1p(node.partner_count)
        x[6] = sum(node.novel) / len(node.novel)
        if len(node.times) >= 2:
            gaps = np.log1p(np.diff(np.asarray(node.times)))
            x[7] = gaps.mean()
            if len(gaps) >= 2:
                x[8] = gaps.std()
        return x

    def _pair_features(self, u, v, t):
        x = np.full(7, np.nan)
        times = self.pair_times.get((u, v))
        if times is None:
            return x
        ordinals = self.pair_ordinals[(u, v)]
        x[0] = np.log1p(t - times[-1])
        x[1] = np.log1p(len(ordinals))
        x[2] = self._count(ordinals, PAIR_WINDOWS[0])
        x[3] = np.log1p(self._count(ordinals, PAIR_WINDOWS[1]))
        x[4] = np.log1p(t - times[0])
        if len(times) >= 2:
            gap = np.diff(times[-(RECENT_PAIR_GAPS + 1):]).mean()
            x[5] = np.log1p(gap)
            if gap > 0:
                x[6] = min((t - times[-1]) / (gap + EPSILON), RHO_MAX)
        return x

    def _popularity_features(self, source, destination):
        x = np.array([0.0, 0.0, np.nan, 0.0])
        if destination is not None:
            x[0] = self._share(destination, NODE_WINDOWS[1])
            x[1] = self._share(destination, NODE_WINDOWS[2])
            x[2] = np.log((x[0] + EPSILON) / (x[1] + EPSILON))
        if source is not None:
            x[3] = self._share(source, NODE_WINDOWS[1])
        return x

    def _neighborhood_features(self, u, v, source, destination):
        x = np.full(4, np.nan)
        if source is not None:
            x[0] = source.partners.count(v)
        if destination is not None:
            x[1] = destination.partners.count(u)
        if source is not None and destination is not None:
            neighbors_u = set(source.neighbors)
            neighbors_v = set(destination.neighbors)
            if u in self.dst:
                neighbors_u |= set(self.dst[u].neighbors)
            if v in self.src:
                neighbors_v |= set(self.src[v].neighbors)
            common = len(neighbors_u & neighbors_v)
            x[2] = common
            x[3] = common / len(neighbors_u | neighbors_v)
        return x

    def features(self, u, v, t):
        """Describe candidate (u, v) at time t from the current history."""
        source = self.src.get(u)
        destination = self.dst.get(v)
        return np.concatenate([
            self._pair_features(u, v, t),
            self._node_features(source, t),
            self._node_features(destination, t),
            self._popularity_features(source, destination),
            self._neighborhood_features(u, v, source, destination),
            [np.log1p(t - self.last_t)],
        ], dtype=np.float32)
