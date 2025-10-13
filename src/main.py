from load_data import load_data

if __name__ == "__main__":
    data = load_data("data/gielda.txt", separator=",", headline=False)
    print(data)
    print(data.dtype)
