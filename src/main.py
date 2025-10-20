from load_data import load_data
from data_stats import count_unique_values, value_per_attribute
from entropy import entropy, entropy_per_attribute, information_gain, best_attribute_by_gain

if __name__ == "__main__":
    data = load_data("data/gielda.txt", separator=",", headline=False)
    print("Data:", data)
    print("Data type:", data.dtype)

    print("Number of unique values:", count_unique_values(data))
    print("Values and their counts:", value_per_attribute(data))

    print("Entropy of the decision class:", entropy(data))
    print("Entropy for each attribute:", entropy_per_attribute(data))

    print("Information gain for each attribute:", information_gain(data))
    print("Best attribute according to information gain:", best_attribute_by_gain(data))

