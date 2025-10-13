import numpy as np


def count_unique_values(data: np.ndarray) -> list[int]:
    counts = []
    for col_idx in range(data.shape[1]):
        unique_values = set(data[:, col_idx])
        counts.append(len(unique_values))

    return counts


def value_per_attribute(data: np.ndarray) -> list[dict[any, int]]:
    value_counts = []
    for col_idx in range(data.shape[1]):
        col_values = data[:, col_idx]
        counts = {}
        for value in col_values:
            if value in counts:
                counts[value] += 1
            else:
                counts[value] = 1
        value_counts.append(counts)

    return value_counts
