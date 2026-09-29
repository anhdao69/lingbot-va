"""Contiguous dense KV with legacy-compatible virtual-slot eviction.

Repeated temporary reads append into spare storage and restore only a Python
length. CUDA indexing/compaction is restricted to structural history changes.
"""

from functools import lru_cache

import numpy as np
import torch


@lru_cache(maxsize=256)
def _legacy_eviction_order(relative_ids, device):
    # Equal-age argsort is deliberately NOT made stable: that would select a
    # different token subset on partial eviction. Share this cold metadata plan
    # across layers; no key/value tensor is cached here.
    ids = torch.tensor(relative_ids, dtype=torch.int64, device=device)
    return tuple(torch.argsort(ids).cpu().tolist())


class DenseFastCache:
    def __init__(self, capacity, batch, heads, dim, device, dtype):
        self.capacity = capacity
        self.k = torch.empty((batch, capacity, heads, dim), device=device, dtype=dtype)
        self.v = torch.empty_like(self.k)
        self.length = 0
        self.slots = []
        self.ids = np.full(capacity, -1, dtype=np.int64)
        self.pred = np.zeros(capacity, dtype=bool)

    def state_bytes(self):
        return (self.k.numel() + self.v.numel()) * self.k.element_size()

    def _compact(self, keep):
        old_slots = self.slots
        keep_slots = [old_slots[i] for i in keep]
        removed = set(old_slots).difference(keep_slots)
        if removed:
            self.ids[list(removed)] = -1
            self.pred[list(removed)] = False
        if keep != list(range(len(keep))):
            index = torch.tensor(keep, dtype=torch.long, device=self.k.device)
            # Separate gathered temporaries avoid overlapping source/destination.
            self.k[:, : len(keep)].copy_(self.k.index_select(1, index))
            self.v[:, : len(keep)].copy_(self.v.index_select(1, index))
        self.slots = keep_slots
        self.length = len(keep)

    def ensure_space(self, tokens):
        if not 0 < tokens <= self.capacity:
            raise ValueError("Current chunk must fit in the dense cache")
        need = self.length + tokens - self.capacity
        if need <= 0:
            return
        used = np.flatnonzero(self.ids >= 0)
        ages = self.ids[used]
        relative = tuple((ages - ages.min()).tolist())
        order = _legacy_eviction_order(relative, str(self.k.device))
        removed = set(used[list(order[:need])].tolist())
        self._compact([i for i, slot in enumerate(self.slots) if slot not in removed])

    def _write(self, key, value):
        end = self.length + key.shape[1]
        self.k[:, self.length : end].copy_(key)
        self.v[:, self.length : end].copy_(value)
        return end

    def append(self, key, value, is_pred):
        self.ensure_space(key.shape[1])
        end = self._write(key, value)
        free = np.flatnonzero(self.ids < 0)[: key.shape[1]]
        self.ids[free] = int(self.ids.max()) + 1
        self.pred[free] = is_pred
        self.slots.extend(free.tolist())
        self.length = end
        return tuple(free.tolist())

    def restore(self, slots):
        # Remove precisely the returned virtual slots, including non-LIFO calls.
        # Original rollback never resurrects entries evicted by insertion.
        removed = set(slots)
        self._compact([i for i, slot in enumerate(self.slots) if slot not in removed])

    def clear_pred(self):
        keep = [i for i, slot in enumerate(self.slots) if not self.pred[slot]]
        if len(keep) != self.length:
            self._compact(keep)

    def attend(self, query, key, value, update_cache, attention):
        if update_cache not in (0, 1, 2):
            raise ValueError("update_cache must be 0, 1 or 2")
        if update_cache == 0:
            self.ensure_space(key.shape[1])
            end = self._write(key, value)
            return attention(query, self.k[:, :end], self.v[:, :end])
        self.append(key, value, is_pred=update_cache == 1)
        return attention(query, self.k[:, : self.length], self.v[:, : self.length])
