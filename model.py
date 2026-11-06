"""Frozen TabPFN classifier conditioned on the rolling context."""
import numpy as np
from tabpfn import TabPFNClassifier

from context import ContextTable
from stream import StreamState


class StreamTab:
    def __init__(self, regime, k, g, seed, device, model_path):
        self.state = StreamState()
        self.context = ContextTable(self.state, regime, k, g, seed)
        self.classifier = TabPFNClassifier(device=device, model_path=model_path,
                                           random_state=seed, ignore_pretraining_limits=True)

    def refresh(self):
        """Condition the classifier on the most recent context rows."""
        self.classifier.fit(*self.context.snapshot())

    def features(self, candidates):
        return np.stack([self.state.features(u, v, t) for u, v, t in candidates])

    def score(self, rows):
        return self.classifier.predict_proba(rows)[:, 1]

    def observe(self, batch):
        """Add one batch to the context, then to the history."""
        for event in batch:
            self.context.add(*event)
        for event in batch:
            self.state.ingest(*event)
