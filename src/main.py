import numpy as np
from load_data import load_data
from data_stats import count_unique_values, value_per_attribute
from DecisionTree import DecisionTree
from math_functions import (
    entropy,
    info_attribute,
    gain,
    split_info,
    gain_ratio,
)

def check_math_functions(data: np.array) -> None:
    print("Data:")
    print(data)
    print("dtype:", data.dtype)
    print()

    print("Unique values per column:")
    print(count_unique_values(data))
    print()

    print("Value counts per column:")
    print(value_per_attribute(data))
    print()

    base_entropy = entropy(data)
    print("Entropy of decision attribute (Info(T)):", base_entropy)
    print()

    info = info_attribute(data)
    print("Info(X,T) per attribute:")
    for attr, val in info.items():
        print(f"  a{attr + 1}: {val}")
    print()

    g = gain(data)
    print("Gain(X,T) per attribute:")
    for attr, val in g.items():
        print(f"  a{attr + 1}: {val}")
    print()

    s = split_info(data)
    print("SplitInfo(X,T) per attribute:")
    for attr, val in s.items():
        print(f"  a{attr + 1}: {val}")
    print()

    gr = gain_ratio(data)
    print("GainRatio(X,T) per attribute:")
    for attr, val in gr.items():
        print(f"  a{attr + 1}: {val}")
    print()

    best_gain_attr = max(g, key=g.get)
    best_gr_attr = max(gr, key=gr.get)

    print("Best attribute by Gain:")
    print(f"  a{best_gain_attr + 1} -> {g[best_gain_attr]}")
    print()

    print("Best attribute by GainRatio:")
    print(f"  a{best_gr_attr + 1} -> {gr[best_gr_attr]}")

def check_decision_tree(data: np.array) -> None:
    tree = DecisionTree()
    tree.fit(data)

    new_car_1 = np.array(['low', 'low', '5more', 'more', 'big', 'high'])
    new_car_2 = np.array(['vhigh', 'vhigh', '2', '2', 'small', 'low'])

    col_names = [
        "buying price",
        "maintenance cost",
        "doors count",
        "persons capacity",
        "luggage boot size",
        "estimated safety"
    ]

    print("\n--- TREE VISUALIZATION ---\n")
    tree.print_tree(attribute_names=col_names)

    print("Perfect car test:", tree.predict(new_car_1))
    print("Worst car test:", tree.predict(new_car_2))

if __name__ == "__main__":
    dataset = load_data("data/car.data", separator=",", headline=False)

    #check_math_functions(dataset)
    check_decision_tree(dataset)
