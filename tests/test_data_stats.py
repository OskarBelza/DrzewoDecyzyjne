import pytest
import numpy as np
from src.data_stats import get_unique_values, count_unique_values, value_per_attribute


@pytest.fixture
def sample_data():
    return np.array([
        ['A', '1', 'Tak'],
        ['A', '2', 'Nie'],
        ['B', '3', 'Tak']
    ])


def test_get_unique_values(sample_data):
    col0 = sample_data[:, 0]
    result = get_unique_values(col0)

    assert result == ['A', 'B']


def test_count_unique_values(sample_data):
    expected = [2, 3, 2]

    assert count_unique_values(sample_data) == expected


def test_value_per_attribute(sample_data):
    result = value_per_attribute(sample_data)

    assert len(result) == 3
    assert isinstance(result[0], dict)


    assert result[0] == {'A': 2, 'B': 1}
    assert result[1] == {'1': 1, '2': 1, '3': 1}
    assert result[2] == {'Tak': 2, 'Nie': 1}
