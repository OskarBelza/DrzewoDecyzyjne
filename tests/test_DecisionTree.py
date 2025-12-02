import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.DecisionTree import DecisionTree


@pytest.fixture
def and_gate_data():
    return np.array([
        ['0', '0', '0'],
        ['0', '1', '0'],
        ['1', '0', '0'],
        ['1', '1', '1']
    ])


def test_decision_tree_fit_predict_perfect(and_gate_data):
    tree = DecisionTree()
    tree.fit(and_gate_data)

    assert tree.predict(np.array(['0', '0'])) == '0'
    assert tree.predict(np.array(['0', '1'])) == '0'
    assert tree.predict(np.array(['1', '0'])) == '0'
    assert tree.predict(np.array(['1', '1'])) == '1'


def test_majority_decision_logic():
    data = np.array([
        ['A', 'TAK'],
        ['A', 'TAK'],
        ['A', 'NIE'],
        ['A', 'TAK']
    ])

    decision = DecisionTree._get_majority_decision(data)
    assert decision == 'TAK'


def test_predict_without_fit_raises_error():
    tree = DecisionTree()
    sample = np.array(['0', '0'])

    with pytest.raises(Exception) as excinfo:
        tree.predict(sample)

    assert "Tree is empty" in str(excinfo.value)


def test_unknown_value_handling(and_gate_data):
    tree = DecisionTree()
    tree.fit(and_gate_data)

    bad_sample = np.array(['5', '0'])
    result = tree.predict(bad_sample)

    assert "Unknown value" in result