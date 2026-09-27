"""Reusable, mutation-isolated batches for a small-sample fitting diagnostic."""
from copy import deepcopy


class FixedSampleBatches:
    def __init__(self, samples, batch_size, num_batches=None):
        if not samples or batch_size < 1:
            raise ValueError("Need samples and a positive batch size")
        self.samples = deepcopy(samples)
        self.batch_size = batch_size
        self.num_batches = num_batches or (len(samples) + batch_size - 1) // batch_size

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for i in range(self.num_batches):
            start = (i * self.batch_size) % len(self.samples)
            yield deepcopy(self.samples[start:start + self.batch_size])
