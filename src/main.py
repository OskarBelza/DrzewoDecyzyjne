from load_data import load_data
from data_stats import count_unique_values, value_per_attribute

if __name__ == "__main__":
    data = load_data("data/gielda.txt", separator=",", headline=False)
    print(data)
    print(data.dtype)

    print(count_unique_values(data))
    print(value_per_attribute(data))
