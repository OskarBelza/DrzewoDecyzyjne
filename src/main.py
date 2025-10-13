from load_data import load_data
from data_stats import count_unique_values, value_per_attribute, entropy

if __name__ == "__main__":
    data = load_data("data/gielda.txt", separator=",", headline=False)
    print("Data:", data)
    print("Data type:", data.dtype)

    print("Number of unique values:", count_unique_values(data))
    print("Values and their counts:", value_per_attribute(data))
    print("Dataset entropy:", entropy(data))
