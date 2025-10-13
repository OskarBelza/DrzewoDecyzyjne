import re
import numpy as np


def parse_value(value: str):
    if re.fullmatch(r'-?\d+', value):
        return int(value)
    elif re.fullmatch(r'-?\d*\.\d+', value):
        return float(value)
    return value


def load_data(file_path: str, separator: str = ",", headline: bool = False) -> np.ndarray:
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if headline:
        lines = lines[1:]

    data = []

    for line in lines:
        str_values = line.split(separator)
        parsed_values = [parse_value(value) for value in str_values]
        data.append(parsed_values)

    return np.array(data, dtype=object)
