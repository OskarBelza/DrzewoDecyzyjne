import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.math_functions import (
    entropy_from_counts,
    entropy,
    info_attribute,
    gain,
    split_info,
    gain_ratio
)

@pytest.fixture
def logic_data():
    return np.array([
        ['A', 'X', '1'],
        ['A', 'X', '1'],
        ['B', 'X', '0'],
        ['B', 'X', '0']
    ])


def test_entropy_from_counts():
    counts_chaos = {'tak': 2, 'nie': 2}
    assert entropy_from_counts(counts_chaos) == pytest.approx(1.0)

    counts_order = {'tak': 4, 'nie': 0}
    assert entropy_from_counts(counts_order) == pytest.approx(0.0)

    counts_skewed = {'A': 1, 'B': 3}
    assert entropy_from_counts(counts_skewed) == pytest.approx(0.811, abs=0.001)


def test_entropy(logic_data):
    assert entropy(logic_data) == pytest.approx(1.0)


def test_info_attribute(logic_data):
    result = info_attribute(logic_data)

    assert result[0] == pytest.approx(0.0)
    assert result[1] == pytest.approx(1.0)


def test_gain(logic_data):
    result = gain(logic_data)

    assert result[0] == pytest.approx(1.0)
    assert result[1] == pytest.approx(0.0)


def test_split_info(logic_data):
    result = split_info(logic_data)

    assert result[0] == pytest.approx(1.0)
    assert result[1] == pytest.approx(0.0)


def test_gain_ratio(logic_data):
    result = gain_ratio(logic_data)

    assert result[0] == pytest.approx(1.0)
    assert result[1] == pytest.approx(0.0)
