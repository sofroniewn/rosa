import warnings
import math
from typing import List, Optional, Sequence, TypeVar, Union

T = TypeVar("T")
from torch import default_generator, randperm
import numpy as np
from torch.utils.data import Dataset, Subset
from torch._utils import _accumulate
from torch import Generator
import itertools


def make_lengths(lengths, total_length):
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: List[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(total_length * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)
        remainder = total_length - sum(subset_lengths)  # type: ignore[arg-type]
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(
                    f"Length of split at index {i} is 0. "
                    f"This might result in an empty dataset."
                )
    # Cannot verify that dataset is Sized
    if sum(lengths) != total_length:  # type: ignore[arg-type]
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!"
        )

    return lengths


def random_split_multi(
    dataset: Dataset[T],
    lengths: Sequence[Union[int, float]],
    shape: Sequence[int],
    generator: Optional[Generator] = default_generator,
) -> List[Subset[T]]:
    """
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    If a list of fractions that sum up to 1 is given,
    the lengths will be computed automatically as
    floor(frac * len(dataset)) for each fraction provided.

    After computing the lengths, if there are any remainders, 1 count will be
    distributed in round-robin fashion to the lengths
    until there are no remainders left.

    Optionally fix the generator for reproducible results, e.g.:

    >>> random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))
    >>> random_split(range(30), [0.3, 0.3, 0.4], generator=torch.Generator(
    ...   ).manual_seed(42))

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths or fractions of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    # Cannot verify that dataset is Sized
    if math.prod(shape) != len(dataset):  # type: ignore[arg-type]
        raise ValueError(
            "Product of shape does not equal the length of the input dataset!"
        )

    # Make sure fractions are converted into lengths
    lengths_multi = [make_lengths(lengths, s) for s in shape]
    # Generate random list of indices matching total length
    indices_multi = [randperm(sum(lengths), generator=generator).tolist() for lengths in lengths_multi]  # type: ignore[call-overload]
    # Split indices according to lengths
    indices_multi_split = [
        [
            indices[offset - length : offset]
            for offset, length in zip(_accumulate(lengths), lengths)
        ]
        for indices, lengths in zip(indices_multi, lengths_multi)
    ]
    # Create full multi-index for each split
    indices_full_multi_split = []
    for i in range(len(lengths)):
        indices_full_multi_split.append(
            [p for p in itertools.product(*[ind[i] for ind in indices_multi_split])]
        )
    # convert full multi index back into linear index
    indices_linear = [
        [np.ravel_multi_index(i, shape) for i in j] for j in indices_full_multi_split
    ]

    datasubsets = [Subset(dataset, indices) for indices in indices_linear]
    for d, indices_full_multi in zip(datasubsets, indices_full_multi_split):
        d.indices_multi = indices_full_multi

    return datasubsets
