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


def entropy(data: np.ndarray) -> float:
    """
    Entropy is a measure of impurity or uncertainty in a dataset.
    In decision trees, it quantifies how mixed the class labels are low entropy
    means the data are mostly of one class (pure), while high entropy means the
    data are evenly distributed among different classes (high disorder).
    """
    values_per_attr = value_per_attribute(data)

    class_counts = values_per_attr[-1]
    total = sum(class_counts.values())

    ent = 0.0
    for count in class_counts.values():
        p = count / total
        if p > 0:
            ent -= p * np.log2(p)
    return ent
