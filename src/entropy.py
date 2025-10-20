import numpy as np
from data_stats import value_per_attribute


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


def entropy_per_attribute(data: np.ndarray) -> dict[int, float]:
    n_cols = data.shape[1]
    result = {}
    total_rows = len(data)
    values_per_attr = value_per_attribute(data)

    for attr_idx in range(n_cols - 1):
        attr_value_counts = values_per_attr[attr_idx]
        weighted_entropy = 0.0

        for val in attr_value_counts.keys():
            subset = data[data[:, attr_idx] == val]
            subset_entropy = entropy(subset)
            weight = len(subset) / total_rows
            weighted_entropy += weight * subset_entropy

        result[attr_idx] = weighted_entropy

    return result


def information_gain(data: np.ndarray) -> dict[int, float]:
    base_entropy = entropy(data)
    info = entropy_per_attribute(data)
    return {attr: base_entropy - h for attr, h in info.items()}


def best_attribute_by_gain(data: np.ndarray) -> tuple[int, float]:
    info_gain = information_gain(data)
    best_attr = max(info_gain, key=info_gain.get)
    return best_attr, info_gain[best_attr]
