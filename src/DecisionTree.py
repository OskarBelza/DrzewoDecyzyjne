import numpy as np
from math_functions import gain_ratio
from data_stats import get_unique_values


class TreeNode:

    def __init__(self, attribute_index=None, decision=None, children=None):
        self.attribute_index = attribute_index
        self.decision = decision
        self.children = children if children else {}
        self.is_leaf = decision is not None


class DecisionTree:

    def __init__(self):
        self.root = None

    def fit(self, data: np.ndarray):
        self.root = self._build_tree_recursive(data)

    def predict(self, sample: np.ndarray) -> any:
        if self.root is None:
            raise Exception("Tree is empty! Build the tree using .fit() method first.")
        return self._predict_recursive(sample, self.root)

    def print_tree(self, attribute_names=None):
        if self.root is None:
            raise Exception("Tree is empty! Build the tree using .fit() method first.")

        self._print_node(self.root, attribute_names, "", True, "Start")


    @staticmethod
    def _get_majority_decision(data):
        decision_column = data[:, -1]
        counts = {}
        for d in decision_column:
            counts[d] = counts.get(d, 0) + 1
        return max(counts, key=counts.get)

    def _build_tree_recursive(self, data: np.ndarray) -> TreeNode:
        decision_column = data[:, -1]
        unique_decisions = get_unique_values(decision_column)

        if len(unique_decisions) == 1:
            return TreeNode(decision=unique_decisions[0])

        ratios = gain_ratio(data)

        if not ratios:
            return TreeNode(decision=self._get_majority_decision(data))

        best_attr_idx = max(ratios, key=ratios.get)

        if ratios[best_attr_idx] == 0:
            return TreeNode(decision=self._get_majority_decision(data))

        node = TreeNode(attribute_index=best_attr_idx)

        col_values = data[:, best_attr_idx]
        unique_vals = get_unique_values(col_values)

        for val in unique_vals:
            subset = data[data[:, best_attr_idx] == val]
            if len(subset) == 0:
                node.children[val] = TreeNode(decision=self._get_majority_decision(data))
            else:
                node.children[val] = self._build_tree_recursive(subset)

        return node

    def _predict_recursive(self, sample, node):
        if node.is_leaf:
            return node.decision

        attr_val = sample[node.attribute_index]

        if attr_val in node.children:
            return self._predict_recursive(sample, node.children[attr_val])
        else:
            return "Unknown value: " + str(attr_val)

    def _print_node(self, node, attr_names, prefix, is_last, description):
        connector = "└── " if is_last else "├── "

        if node.is_leaf:
            content = f"\033[92mDECISON: {node.decision}\033[0m"
        else:
            attr_name = attr_names[node.attribute_index] if attr_names else f"Attribute {node.attribute_index}"
            content = f"\033[94m[ {attr_name}? ]\033[0m"

        if description == "Start":
            print(content)
        else:
            print(f"{prefix}{connector}({description}) -> {content}")

        new_prefix = prefix + ("    " if is_last else "│   ")

        if not node.is_leaf:
            children_keys = sorted(list(node.children.keys()), key=str)

            count = len(children_keys)

            for i, key in enumerate(children_keys):
                is_last_child = (i == count - 1)
                self._print_node(node.children[key], attr_names, new_prefix, is_last_child, key)
