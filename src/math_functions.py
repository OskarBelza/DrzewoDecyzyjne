import numpy as np
from data_stats import value_per_attribute, get_unique_values


def entropy_from_counts(value_frequency: dict) -> float:
    total_count = sum(value_frequency.values())
    entropy_value = 0.0

    for frequency in value_frequency.values():
        if frequency > 0:
            probability = frequency / total_count
            entropy_value -= probability * np.log2(probability)

    return entropy_value


def entropy(data: np.ndarray) -> float:
    decision_value_frequency = value_per_attribute(data)[-1]
    return entropy_from_counts(decision_value_frequency)


def info_attribute(data: np.ndarray) -> dict[int, float]:
    result = {}
    num_columns = data.shape[1]
    total_rows = len(data)

    for attribute_index in range(num_columns - 1):
        column_values = data[:, attribute_index]
        unique_values = get_unique_values(column_values)

        weighted_entropy = 0.0
        for attribute_value in unique_values:
            subset = data[column_values == attribute_value]
            subset_entropy = entropy(subset)
            weight = len(subset) / total_rows
            weighted_entropy += weight * subset_entropy

        result[attribute_index] = weighted_entropy

    return result


def gain(data: np.ndarray) -> dict[int, float]:
    base_entropy = entropy(data)
    info_values = info_attribute(data)
    return {
        attribute_index: base_entropy - info_value
        for attribute_index, info_value in info_values.items()
    }


def split_info(data: np.ndarray) -> dict[int, float]:
    result = {}
    all_value_frequencies = value_per_attribute(data)
    num_columns = data.shape[1]

    for attribute_index in range(num_columns - 1):
        value_frequency = all_value_frequencies[attribute_index]
        result[attribute_index] = entropy_from_counts(value_frequency)

    return result


def gain_ratio(data: np.ndarray) -> dict[int, float]:
    gain_values = gain(data)
    split_info_values = split_info(data)
    result = {}

    for attribute_index, gain_value in gain_values.items():
        split_value = split_info_values[attribute_index]

        if split_value == 0:
            result[attribute_index] = 0.0
        else:
            result[attribute_index] = gain_value / split_value

    return result
