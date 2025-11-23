import math
from typing import TypeVar, Optional, Iterator

import torch
from torch.utils.data import Sampler, Dataset
import torch.distributed as dist


T_co = TypeVar('T_co', covariant=True)


class DistributedRandomSampler(Sampler[T_co]):
    r"""Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False, random_ratio=1) -> None:
        self.random_ratio = random_ratio if random_ratio <= 1 else random_ratio / len(dataset)
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and int(len(self.dataset)*self.random_ratio) % self.num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (int(len(self.dataset)*self.random_ratio) - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(int(len(self.dataset)*self.random_ratio) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()[:int(len(self.dataset)*self.random_ratio)]
        else:
            indices = list(range(len(self.dataset)))[:int(len(self.dataset)*self.random_ratio)]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class DualDatasetSampler(Sampler[T_co]):
    """Sampler that samples from two datasets simultaneously.
    
    This sampler ensures that each batch contains samples from both datasets.
    For example, if batch_size=4, it will sample 2 samples from each dataset.
    Supports distributed training with DDP.
    When dataset2 is shorter than dataset1, it will repeat samples from dataset2.
    
    Args:
        dataset1: First dataset
        dataset2: Second dataset
        batch_size: Total batch size (must be even)
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle: Whether to shuffle the data
        seed: Random seed for shuffling
        drop_last: If True, then the sampler will drop the tail of the data
            to make it evenly divisible across the number of replicas.
        random_ratio: Ratio of data to sample. If <= 1, it's treated as a ratio.
            If > 1, it's treated as the number of samples to take.
    """
    
    def __init__(self, dataset1: Dataset, dataset2: Dataset, batch_size: int = 4,
                 num_replicas: Optional[int] = None, rank: Optional[int] = None,
                 shuffle: bool = True, seed: int = 0, drop_last: bool = False,
                 random_ratio: float = 1.0) -> None:
        if batch_size % 2 != 0:
            raise ValueError("batch_size must be even")
            
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )
            
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.batch_size = batch_size
        self.samples_per_dataset = batch_size // 2
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        
        self.input_random_ratio = random_ratio
        len_d1 = len(self.dataset1)
        len_d2 = len(self.dataset2)

        if 0 <= self.input_random_ratio <= 1.0:
            self.length1 = int(len_d1 * self.input_random_ratio)
            self.length2 = int(len_d2 * self.input_random_ratio)
        elif self.input_random_ratio > 1.0:
            # Treat as an absolute number of samples to aim for from each, capped by dataset length
            self.length1 = min(int(self.input_random_ratio), len_d1)
            self.length2 = min(int(self.input_random_ratio), len_d2)
        else: # Negative random_ratio
            raise ValueError("random_ratio must be non-negative.")

        # Check for problematic empty dataset condition for pairing
        if self.length1 > 0 and self.length2 == 0:
            raise ValueError(
                "DualDatasetSampler: dataset1 has items to sample ({} after random_ratio) but dataset2 is effectively empty "
                "(0 after random_ratio). Cannot guarantee 50/50 paired sampling as dataset2 cannot provide samples."
                .format(self.length1)
            )
        
        # self.total_length is the reference for DDP calculations, based on d1's effective item count.
        # It dictates how many pairs will be formed before DDP adjustments.
        self.total_length = self.length1 
        
        # Calculate DDP samples per replica based on d1's effective item count
        if self.drop_last and self.total_length % self.num_replicas != 0:
            # Number of d1 items per replica
            self.num_samples = math.ceil(
                (self.total_length - self.num_replicas) / self.num_replicas
            )
        else:
            # Number of d1 items per replica
            self.num_samples = math.ceil(self.total_length / self.num_replicas)
            if self.num_samples % self.samples_per_dataset != 0:
                self.num_samples = math.ceil(self.num_samples / self.samples_per_dataset) * self.samples_per_dataset
        
        # Total d1 items across all replicas after DDP adjustment
        self.total_size = self.num_samples * self.num_replicas
        
        self.shuffle = shuffle
        self.seed = seed
        
    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices1 = torch.randperm(len(self.dataset1), generator=g).tolist()[:self.length1]
            indices2 = torch.randperm(len(self.dataset2), generator=g).tolist()[:self.length2]
        else:
            indices1 = list(range(self.length1))
            indices2 = list(range(self.length2))

        # Debug prints
        # print(f"DualDatasetSampler Debug:")
        # print(f"  len(dataset1): {len(self.dataset1)}, len(dataset2): {len(self.dataset2)}")
        # print(f"  self.length1: {self.length1}, self.length2: {self.length2}")
        # print(f"  indices1[:5]: {indices1[:5] if indices1 else 'EMPTY'}")
        # print(f"  indices2[:5]: {indices2[:5] if indices2 else 'EMPTY'}")
        # print(f"  samples_per_dataset: {self.samples_per_dataset}")

        # Create pairs of indices instead of interleaved list
        pairs = []
        current_pos_indices2 = 0

        for i_start_d1 in range(0, self.length1, self.samples_per_dataset):
            batch1_segment = indices1[i_start_d1 : i_start_d1 + self.samples_per_dataset]
            num_from_d1_this_step = len(batch1_segment)

            if num_from_d1_this_step == 0:
                break

            batch2_segment_raw = []
            if self.length2 > 0:
                for _ in range(num_from_d1_this_step):
                    batch2_segment_raw.append(indices2[current_pos_indices2])
                    current_pos_indices2 = (current_pos_indices2 + 1) % self.length2
            
            batch2_segment_offset = [idx + len(self.dataset1) for idx in batch2_segment_raw]
            
            # Debug prints for each segment
            # print(f"  Segment {i_start_d1//self.samples_per_dataset}:")
            # print(f"    batch1_segment: {batch1_segment}")
            # print(f"    batch2_segment_raw: {batch2_segment_raw}")
            # print(f"    batch2_segment_offset: {batch2_segment_offset}")
            
            # Create pairs: each pair contains samples_per_dataset items from each dataset
            pair = batch1_segment + batch2_segment_offset
            pairs.append(pair)
        
        # print(f"  Created {len(pairs)} pairs")
        # print(f"  First pair: {pairs[0] if pairs else 'EMPTY'}")

        # Calculate how many pairs needed after DDP adjustment
        # self.total_size is total d1 items across all replicas
        # So we need self.total_size // self.samples_per_dataset pairs total
        total_pairs_needed = self.total_size // self.samples_per_dataset
        
        # Extend pairs if needed
        if len(pairs) < total_pairs_needed:
            repeat_times = math.ceil(total_pairs_needed / len(pairs))
            pairs = (pairs * repeat_times)[:total_pairs_needed]
        elif len(pairs) > total_pairs_needed:
            pairs = pairs[:total_pairs_needed]
        
        # print(f"  After padding/truncation: {len(pairs)} pairs")

        # DDP subsampling on pairs
        pairs_per_replica = self.num_samples // self.samples_per_dataset
        subsampled_pairs = pairs[self.rank * pairs_per_replica:(self.rank + 1) * pairs_per_replica]
        
        # print(f"  Rank {self.rank} gets {len(subsampled_pairs)} pairs")
        # print(f"  pairs_per_replica: {pairs_per_replica}")
        
        # Flatten subsampled pairs into final indices
        subsampled_indices = []
        for pair in subsampled_pairs:
            subsampled_indices.extend(pair)
        
        expected_len_subsampled_per_replica = self.num_samples * 2
        if len(pairs) == 0:
             expected_len_subsampled_per_replica = 0
        
        # print(f"  After DDP subsampling for rank {self.rank}:")
        # print(f"    subsampled_indices[:10]: {subsampled_indices[:10]}")
        # print(f"    len(subsampled_indices): {len(subsampled_indices)}")
        
        assert len(subsampled_indices) == expected_len_subsampled_per_replica, \
            f"DualDatasetSampler Error: len(subsampled_indices)={len(subsampled_indices)} for rank {self.rank} " \
            f"!= expected_len_subsampled_per_replica={expected_len_subsampled_per_replica}"

        return iter(subsampled_indices)
    
    def __len__(self) -> int:
        # self.num_samples is the number of items from dataset1 per replica.
        # Since we pair each d1 item with a d2 item, total items per replica is num_samples * 2.
        return self.num_samples * 2
        
    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler.
        
        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
