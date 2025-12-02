import numpy as np


def count_unique_values(data: np.ndarray) -> list[int]:
    counts = []
    for col_idx in range(data.shape[1]):
        col_data = data[:, col_idx]
        unique_vals = get_unique_values(col_data)
        counts.append(len(unique_vals))

    return counts


def get_unique_values(data_column: np.ndarray) -> list:
    unique_list = []
    for value in data_column:
        if value not in unique_list:
            unique_list.append(value)
    return unique_list


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
