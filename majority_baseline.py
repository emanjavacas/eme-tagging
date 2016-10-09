

from collections import Counter


class MajorityBaseline(object):
    def __init__(self):
        self.word2tag = None
        self.unknown = 'UNK'

    def predict(self, X):
        if not self.word2tag:
            raise ValueError("Model hasn't been trained yet")
        return [self.word2tag.get(w.lower(), self.unknown) for w in X]

    def fit(self, X, y, **kwargs):
        counts = Counter(zip(X, y))
        results = {}
        for (w, t), c in counts.items():
            if w in results:
                current_max_count = counts.get((w, t), 0)
                results[w] = t if c > current_max_count else results[w]
            else:
                results[w] = t
        self.word2tag = results
        self.classes_ = results.values()
