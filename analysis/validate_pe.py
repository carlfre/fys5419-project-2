from time import time

import numpy as np
import matplotlib.pyplot as plt

from qiskit_implementations import (
    inverse_qft_qiskit,
    phase_estimation_qiskit,
    qft_qiskit,
)
from phase_estimation import phase_estimation
from utils import freq_to_probabilities, latex_table

plt.rcParams.update({
    'axes.labelsize': 16,
    'legend.fontsize': 14,
    'font.size': 12,  # base font size
})


def fraction_to_binary(num: float) -> str:
    """Convert a fraction to its binary representation."""

    assert 0 < num < 1, "Number must be between 0 and 1"

    binary = "0."
    while num > 0 and len(binary) < 32:
        num *= 2
        if num >= 1:
            binary += "1"
            num -= 1
        else:
            binary += "0"
    return binary


def phase_estimation_histogram_1_over_7():
    ratio = 1 / 7
    t = 4  # Digits of precision in phase estimation


    U = np.diag([1, 1, 1, np.exp(2 * np.pi * 1j * ratio)])  # CU1(2 * np.pi * ratio)
    u = np.array([0, 0, 0, 1])

    estimated_binaries_owncode = phase_estimation(U, u, t, n_shots=1024)
    estimated_binaries_qiskit = phase_estimation_qiskit(U, u, t, n_shots=1024)

    estimated_binaries_owncode = freq_to_probabilities(estimated_binaries_owncode)
    estimated_binaries_qiskit = freq_to_probabilities(estimated_binaries_qiskit)

    xticks = [f"0.{np.binary_repr(i, width=t)}" for i in range(0, 2**t)]
    x = np.arange(len(xticks))

    # Get the frequencies in correct order (0.0000 to 0.1111)
    frequencies_owncode = [
        estimated_binaries_owncode.get(bin_val, 0) for bin_val in xticks
    ]
    frequencies_qiskit = [
        estimated_binaries_qiskit.get(bin_val, 0) for bin_val in xticks
    ]

    # Plot
    plt.bar(x - 0.2, frequencies_owncode, width=0.4, color="blue", label="Our code")
    plt.bar(x + 0.2, frequencies_qiskit, width=0.4, color="red", label="Qiskit")
    plt.xticks(x, xticks, rotation=45)
    plt.xlabel("Estimated Phase (binary)")
    plt.ylabel("Probability")
    plt.tight_layout()
    plt.legend()
    plt.savefig("plots/phase_estimation_1_over_7.pdf")
    # plt.show()

def phase_estimation_1_over_7_as_function_of_qubit_count():
    t_max = 10

    ratio = 1 / 7
    U = np.diag([1, 1, 1, np.exp(2 * np.pi * 1j * ratio)])  # CU1(2 * np.pi * ratio)
    u = np.array([0, 0, 0, 1])
    true_binary = fraction_to_binary(ratio)
    print(true_binary)

    owncode_estimates = []
    qiskit_estimates = []

    for t in range(1, t_max+1):
        estimated_binaries_owncode = phase_estimation(U, u, t, n_shots=1024)
        estimated_binaries_qiskit = phase_estimation_qiskit(U, u, t, n_shots=1024)
        estimated_binary = max(estimated_binaries_owncode, key=estimated_binaries_owncode.get)
        estimated_binary_qiskit = max(estimated_binaries_qiskit, key=estimated_binaries_qiskit.get)
        print(f"t={t}, estimated binary (own code): {estimated_binary}, estimated binary (qiskit): {estimated_binary_qiskit}")
        owncode_estimates.append(estimated_binary)
        qiskit_estimates.append(estimated_binary_qiskit)

    data = list(zip(owncode_estimates, qiskit_estimates))
    columns = ["Our code", "Qiskit"]
    rows = [f"{t}" for t in range(1, t_max+1)]
    table = latex_table(data, columns, rows, alignment="l")
    print(table)

    



def timing_phase_estimation():
    t = 17
    ratio = 1 / 7
    U = np.diag([1, 1, 1, np.exp(1j *2 * np.pi * ratio)])
    u = np.array([0, 0, 0, 1])


    start = time()
    phase_binary = phase_estimation(U, u, t, n_shots=1)
    # phase_binary = phase_estimation_qiskit(U, u, t, n_shots=1)
    estimated_phase = max(phase_binary, key=phase_binary.get)
    print(f"Estimated Phase (binary fraction): {estimated_phase}")
    print("time", time() - start)





if __name__ == "__main__":
    # phase_estimation_histogram_1_over_7()
    # phase_estimation_1_over_7_as_function_of_qubit_count()

    timing_phase_estimation()