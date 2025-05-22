

import numpy as np


from collections import Counter


def write_csv(data: dict, file_name: str) -> None:
    with open(file_name, 'w') as f:
        for key, value in data.items():
            f.write(f"{key},{value}\n")
    print(f"Data written to {file_name}")


def read_csv(file_name: str) -> dict[int, float]:
    """Reads a CSV file and returns a dictionary with integer keys and float values."""
    data = {}
    with open(file_name, 'r') as f:
        for line in f:
            key, value = line.strip().split(',')
            data[int(key)] = float(value)
    return data


def latex_table(data: list[list], above_headers: list[str] | None = None, left_headers: list[str] | None = None, alignment='c'):
    """
    Converts a nested list into a LaTeX table.

    Parameters:
        data: List of lists (e.g., [[1, 2], [3, 4]])
        alignment: Column alignment ('c', 'l', 'r')

    Returns:
    - A string containing the LaTeX code for the table.
    """
    num_cols = len(data[0])


    if left_headers is None:
        alignment_str = " ".join([alignment] * num_cols)
    else:
        alignment_str = " ".join([alignment] * (num_cols + 1))
    latex_lines = [f"\\begin{{tabular}}{{{alignment_str}}}"]

    latex_lines.append("\\toprule")
    if above_headers is not None:
        header_str = " & ".join(f"{header}" for header in above_headers) + " \\\\ "
        if left_headers is not None:
            header_str = " & " + header_str
        latex_lines.append(header_str)
        latex_lines.append("\\midrule")

    for i, row in enumerate(data):
        # if left_headers is not None:
        row_str = " & ".join(f"${cell}$" for cell in row) + " \\\\ "
        if left_headers is not None:
            latex_lines.append(f"{left_headers[i]} & {row_str}")
        else:
            latex_lines.append(row_str)

    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")

    return "\n".join(latex_lines)

def freq_to_probabilities(frequencies: dict[int, int]) -> dict[int, float]:
    """Converts frequencies to probabilities (relative frequencies)."""
    total = sum(frequencies.values())
    return {k: v / total for k, v in frequencies.items()}


def measure(psi: np.ndarray, n_shots: int) -> dict[str, int]:
    """Make n_shots measurements of the state psi."""

    n_qubits = round(np.log2(psi.shape[0]))
    probability_vector = np.abs(psi) ** 2
    samples = np.random.choice(
        np.arange(probability_vector.shape[0]),
        p=probability_vector,
        size=n_shots,
        replace=True,
    )
    binary_representations = [np.binary_repr(s, n_qubits) for s in samples]
    return dict(Counter(binary_representations))

if __name__ == "__main__":
    # data = {1: 0.5, 2: 0.25, 3: 0.125}
    # write_csv(data, 'results/test.csv')
    # read_data = read_csv('results/test.csv')
    # print(read_data)
    above_headers = ["Column 1", "Column 2", "Column 3"]
    left_headers = ["Row 1", "Row 2", "Row 3"]
    data = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    table = latex_table(data, above_headers=above_headers, left_headers=left_headers)
    print(table)


def are_distributions_close(frequencies1: dict[str, int], frequencies2: dict[str, int], tol: float = 0.05) -> bool:
    """Checks if the l^∞ distance between two distributions is less than tol."""

    dist1 = freq_to_probabilities(frequencies1)
    dist2 = freq_to_probabilities(frequencies2)

    if not set(dist1.keys()) == set(dist2.keys()):
        for key in dist1.keys():
            if key not in dist2:
                dist2[key] = 0
        for key in dist2.keys():
            if key not in dist1:
                dist1[key] = 0

    for key in dist1.keys():
        if abs(dist1[key] - dist2[key]) > tol: # Checking absolute difference
            return False

    return True


def random_state(n_qubits: int) -> np.ndarray:
    """Generates a random n_qubit state."""
    psi = np.random.rand(2**n_qubits) + 1j * np.random.rand(2**n_qubits)
    psi = psi / np.linalg.norm(psi)
    return psi

def sorted_dict(d: dict) -> dict:
    """Sorts dictionary by values, in descending order """
    return dict(sorted(d.items(), key=lambda item: -item[1]))

