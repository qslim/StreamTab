import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from model import StreamTab

BATCH_SIZE = 200


def split_batches(events):
    return [events[i:i + BATCH_SIZE] for i in range(0, len(events), BATCH_SIZE)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='CSV file with columns u, i, ts')
    parser.add_argument('--regime', default='historical',
                        choices=['random', 'historical', 'inductive'])
    parser.add_argument('--k', type=int, default=8192, help='Context size')
    parser.add_argument('--R', type=int, default=4, help='Refresh interval in batches')
    parser.add_argument('--g', type=int, default=1, help='Negatives per context event')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--model_path', default='auto', help='TabPFN checkpoint')
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    times = data.ts.to_numpy(dtype=float)
    events = list(zip(data.u.tolist(), data.i.tolist(), times.tolist()))
    train_end, test_start = np.searchsorted(times, np.quantile(times, [0.70, 0.85]), side='right')

    model = StreamTab(args.regime, args.k, args.g, args.seed, args.device, args.model_path)

    model.context.cutoff = train_end
    for batch in split_batches(events[:test_start]):
        model.observe(batch)
    model.context.cutoff = test_start

    test_batches = split_batches(events[test_start:])
    ap = []
    for start in range(0, len(test_batches), args.R):
        model.refresh()
        cycle = test_batches[start:start + args.R]
        rows = []
        for batch in cycle:
            negatives = [(*model.context.sample_negative(u, v), t) for u, v, t in batch]
            rows.append(model.features(batch + negatives))
            model.observe(batch)
        scores = model.score(np.concatenate(rows))
        offset = 0
        for batch in cycle:
            n = len(batch)
            labels = np.r_[np.ones(n), np.zeros(n)]
            ap.append(average_precision_score(labels, scores[offset:offset + 2 * n]))
            offset += 2 * n
    print(f'AP: {np.mean(ap):.4f}')


if __name__ == '__main__':
    main()
