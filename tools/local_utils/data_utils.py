import random
from collections import defaultdict
from torch.utils.data import Sampler

class GroupBalancedBatchSampler(Sampler):
    def __init__(self, groups, batch_groups=8, group_size=4, drop_last=False, shuffle=True):
        self.groups = groups
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.batch_groups = batch_groups
        self.group_size = group_size
        by_gid = defaultdict(list)
        for idx, g in enumerate(groups):
            by_gid[g].append(idx)
        self.pool = list(by_gid.values())

    def __iter__(self):
        pool = self.pool[:]
        if self.shuffle:
            random.shuffle(pool)
            for p in pool:
                random.shuffle(p)
        batch = []
        for g in pool:
            if len(g) < self.group_size: continue
            batch.extend(g[:self.group_size])
            if len(batch) >= self.batch_groups * self.group_size:
                yield batch[:self.batch_groups * self.group_size]
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        usable = sum(1 for g in self.pool if len(g) >= self.group_size)
        per_batch = self.batch_groups
        return max(1, usable // per_batch)
