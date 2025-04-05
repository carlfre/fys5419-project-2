import csv

def write_to_csv(cols: list[list], headers: list[str], filename: str):
    rows = zip(*cols) # Transpose columns to rows

    with open(filename, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)


def read_csv(filename: str) -> list[list]:
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        rows = [row for row in reader]

    cols = list(zip(*rows)) # Transpose rows to columns
    headers = [col[0] for col in cols]
    cols = [list(map(float, col[1:])) for col in cols]
    return cols, headers



if __name__ == "__main__":
    # write_to_csv([[1, 2, 3], [4, 5, 6], [7, 8, 9]], ['a', 'b', 'c'], 'output/test.csv')
    cols, headers = read_csv('output/lipkin_J_eq_2_new.csv')
    # cols, headers = read_csv('output/simple_2_qubit_new.csv')
    V_over_eps, energies_J_eq_2 = cols

    import matplotlib.pyplot as plt
    plt.plot(V_over_eps, energies_J_eq_2)
    plt.show()