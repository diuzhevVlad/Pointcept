import math
import os
import numpy as np
import torch.utils.data

import pointcept.utils.comm as comm


class RareClassBatchSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        dataset,
        batch_size,
        rare_classes,
        min_rare=1,
        shuffle=True,
        drop_last=True,
        seed=None,
        require_labels=True,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if min_rare <= 0 or min_rare > batch_size:
            raise ValueError("min_rare must be in [1, batch_size]")
        if not rare_classes:
            raise ValueError("rare_classes must be non-empty")

        self.dataset = dataset
        self.batch_size = batch_size
        self.rare_classes = list(rare_classes)
        self.min_rare = min_rare
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = 0 if seed is None else int(seed)
        self.require_labels = require_labels
        self.epoch = 0

        self._rare_indices, self._common_indices = self._build_indices()
        if len(self._rare_indices) == 0:
            raise ValueError("No samples contain the requested rare classes")

        self.world_size = comm.get_world_size()
        self.rank = comm.get_rank()

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return int(math.ceil(len(self.dataset) / self.batch_size))

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)

        rare_indices = self._shard(self._rare_indices, rng)
        common_indices = self._shard(self._common_indices, rng)

        if len(rare_indices) == 0:
            # Fallback: use global rare indices if sharding produced none.
            rare_indices = self._rare_indices

        num_batches = len(self)
        num_rare_total = num_batches * self.min_rare
        num_common_total = num_batches * (self.batch_size - self.min_rare)

        rare_draw = self._draw(rare_indices, num_rare_total, rng)
        common_draw = self._draw(common_indices, num_common_total, rng)

        for i in range(num_batches):
            batch = []
            start_r = i * self.min_rare
            end_r = start_r + self.min_rare
            batch.extend(rare_draw[start_r:end_r])

            start_c = i * (self.batch_size - self.min_rare)
            end_c = start_c + (self.batch_size - self.min_rare)
            batch.extend(common_draw[start_c:end_c])

            if self.shuffle:
                rng.shuffle(batch)
            yield batch

    def _shard(self, indices, rng):
        if self.world_size <= 1:
            return indices
        if self.shuffle:
            indices = list(indices)
            rng.shuffle(indices)
        shards = np.array_split(indices, self.world_size)
        if len(shards) == 0:
            return []
        return list(shards[self.rank])

    def _draw(self, indices, count, rng):
        if count == 0:
            return []
        if len(indices) == 0:
            return []
        replace = count > len(indices)
        return rng.choice(indices, size=count, replace=replace).tolist()

    def _build_indices(self):
        data_list = getattr(self.dataset, "data_list", None)
        if data_list is None:
            raise ValueError("Dataset must expose data_list for RareClassBatchSampler")
        loop = getattr(self.dataset, "loop", 1)

        learning_map = getattr(self.dataset, "learning_map", None)
        ignore_index = getattr(self.dataset, "ignore_index", -1)
        map_array = None
        if learning_map is not None:
            if isinstance(learning_map, np.ndarray):
                map_array = learning_map
            elif isinstance(learning_map, dict):
                max_key = max(learning_map.keys())
                map_array = np.full(max_key + 1, ignore_index, dtype=np.int32)
                for k, v in learning_map.items():
                    if k <= max_key:
                        map_array[k] = v
            else:
                map_array = None

        rare_base = []
        common_base = []

        for idx, data_path in enumerate(data_list):
            label_path = (
                data_path.replace("velodyne", "labels")
                .replace(".bin", ".label")
            )
            if not os.path.exists(label_path):
                if self.require_labels:
                    common_indices.append(idx)
                continue

            labels = np.fromfile(label_path, dtype=np.int32)
            labels = labels & 0xFFFF
            if map_array is not None and labels.size > 0:
                if labels.max() < map_array.shape[0]:
                    labels = map_array[labels]
                else:
                    map_array = None

            if labels.size > 0 and np.isin(labels, self.rare_classes).any():
                rare_base.append(idx)
            else:
                common_base.append(idx)

        if loop <= 1:
            return rare_base, common_base

        data_len = len(data_list)
        rare_indices = []
        common_indices = []
        for repeat in range(loop):
            offset = repeat * data_len
            rare_indices.extend([offset + i for i in rare_base])
            common_indices.extend([offset + i for i in common_base])
        return rare_indices, common_indices
