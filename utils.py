

def write_csv(data: dict, file_name: str):


    with open(file_name, 'w') as f:
        for key, value in data.items():
            f.write(f"{key},{value}\n")
    print(f"Data written to {file_name}")


def read_csv(file_name: str) -> dict:
    data = {}
    with open(file_name, 'r') as f:
        for line in f:
            key, value = line.strip().split(',')
            data[int(key)] = float(value)
    return data



if __name__ == "__main__":
    data = {1: 0.5, 2: 0.25, 3: 0.125}
    write_csv(data, 'results/test.csv')
    read_data = read_csv('results/test.csv')
    print(read_data)