from typing import Literal

import numpy as np
import matplotlib.pyplot as plt

from qiskit_implementations import inverse_qft_qiskit, qft_qiskit
from utils import measure, are_distributions_close, latex_table
from qft import qft, inverse_qft
from linear_algebra import dft
from utils import random_state, measure, freq_to_probabilities
from quantum_ops import multi_kron

plt.rcParams.update({
    'axes.labelsize': 16,
    'legend.fontsize': 14,
    'font.size': 12,  # base font size
})




def validate_qft_for_one_state(psi: np.ndarray, n_shots: int = 4096, tol=0.01, run_type = Literal["qft", "iqft"]) -> bool:
    """Validates the QFT/inverse QFT by checking that the distributions of measurements are close, for methods:
    - own code implementation of QFT
    - DFT matrix applied to the state
    - Qiskit implementation of QFT
    or similarly for inverse QFT. 

    Closeness is defined by l^∞ distance between empirical distributions being less than tol.
    """
    dft_mat = dft(len(psi))

    if run_type == "qft":
        owncode_measurements = measure(qft(psi.copy()), n_shots)
        qiskit_measurements = qft_qiskit(psi.copy(), n_shots)
        dft_measurements = measure(dft_mat @ psi.copy(), n_shots)
    elif run_type == "iqft":
        owncode_measurements = measure(inverse_qft(psi.copy()), n_shots)
        qiskit_measurements = inverse_qft_qiskit(psi.copy(), n_shots)
        dft_measurements = measure(np.linalg.pinv(dft_mat) @ psi.copy(), n_shots)    
    else:
        raise ValueError("Invalid run_type. Choose 'qft' or 'iqft'.")

    # Check if the distributions are close
    owncode_qiskit_close = are_distributions_close(owncode_measurements, qiskit_measurements, tol)
    owncode_dft_close = are_distributions_close(owncode_measurements, dft_measurements, tol)
    qiskit_dft_close = are_distributions_close(qiskit_measurements, dft_measurements, tol)
    return owncode_qiskit_close and owncode_dft_close and qiskit_dft_close


def run_validation_loop(n_qubit_list: list[int], run_type: Literal["qft", "iqft"], n_shots: int = 2**15, n_repetitions: int = 10, tol: float = 0.01):
    """Run validation for qft/inverse qft and return validation rates."""
    validation_rates = []

    for n_qubits in n_qubit_list:
        n_validations = 0
        for i in range(n_repetitions):
            psi = random_state(n_qubits)
            if validate_qft_for_one_state(psi, n_shots=n_shots, run_type=run_type, tol=tol):
                n_validations += 1
        print(f"QFT ({n_qubits} qubits): {n_validations}/{n_repetitions} successful validations.")
        validation_rate = n_validations / n_repetitions
        validation_rates.append(validation_rate)
    
    return validation_rates


def random_validation(run_type: Literal["qft", "iqft"] = "qft"):
    """Run validation for qft/inverse qft and generate table with results."""
    np.random.seed(42)
    n_qubits_list = [1, 2, 3, 4, 5]
    n_shots_list = [2**12, 2**14, 2**16]
    table_data = []
    for n_shots in n_shots_list:
        validation_rates = run_validation_loop(n_qubits_list, run_type=run_type, n_shots=n_shots, n_repetitions=100)
        table_data.append(validation_rates)

    n_qubits_list = [f"${n}$" for n in n_qubits_list]
    n_shots_list = [f"${n}$" for n in n_shots_list]
    table = latex_table(table_data, above_headers=n_qubits_list, left_headers=n_shots_list)
    print(table)


def apply_to_ket0_and_plot_hist(n_qubits: int, run_type: Literal["qft", "iqft"] = "qft"):
    n_shots = 4096
    dft_mat = dft(2**n_qubits)

    ket0 = np.array([1, 0])
    ket_list = [ket0] * n_qubits
    psi = multi_kron(*ket_list, type="numpy")



    if run_type == "qft":
        measurements_owncode = measure(qft(psi.copy()), n_shots)
        measurements_qiskit = qft_qiskit(psi.copy(), n_shots)
        measurements_dftmat = measure(dft_mat @ psi.copy(), n_shots)
    elif run_type == "iqft":
        measurements_owncode = measure(inverse_qft(psi.copy()), n_shots)
        measurements_qiskit = inverse_qft_qiskit(psi.copy(), n_shots)
        measurements_dftmat = measure(np.linalg.pinv(dft_mat) @ psi.copy(), n_shots)

    # Convert to probabilities
    measurements_owncode = freq_to_probabilities(measurements_owncode)
    measurements_qiskit = freq_to_probabilities(measurements_qiskit)
    measurements_dftmat = freq_to_probabilities(measurements_dftmat)

    # Plot histograms
    xticks = [f"{np.binary_repr(i, width=n_qubits)}" for i in range(0, 2**n_qubits)]
    x = np.arange(len(xticks))
    plt.bar(x - 0.2, list(measurements_owncode.values()), width=0.2, label="Own code", color="blue")
    plt.bar(x, list(measurements_qiskit.values()), width=0.2, label="Qiskit", color="red")
    plt.bar(x + 0.2, list(measurements_dftmat.values()), width=0.2, label="DFT matrix", color="green")
    plt.xticks(x, xticks)
    plt.xlabel("Measurement outcomes (binary)")
    plt.ylabel("Probabilities")
    plt.legend()
    plt.savefig(f"plots/{run_type}_of_ket0_nqubits={n_qubits}.pdf")
    plt.close()



def verify_unitarity(n_qubits: int) -> bool:
    np.random.seed(42)
    psi = random_state(n_qubits)
    
    psi_prime = inverse_qft(qft(psi.copy()))

    

    n_shots = 2**20
    tol = 0.01


    psi_measurements = measure(psi, n_shots)
    psi_prime_measurements = measure(psi_prime, n_shots)
    psi_measurements = freq_to_probabilities(psi_measurements)
    psi_prime_measurements = freq_to_probabilities(psi_prime_measurements)

    print(are_distributions_close(psi_measurements.copy(), psi_prime_measurements.copy(), tol))

    
    xticks = [f"{np.binary_repr(i, width=n_qubits)}" for i in range(0, 2**n_qubits)]
    x = np.arange(len(xticks))
    plt.bar(x-0.2, [psi_measurements[xi] for xi in xticks], width=0.2, label=r"$|\psi>$", color="blue")
    plt.bar(x+0.2, [psi_prime_measurements[xi] for xi in xticks], width=0.2, label=r"$QFT^\dagger QFT |\psi>$", color="red")
    plt.xticks(x, xticks)
    plt.xlabel("Measurement outcomes (binary)")
    plt.ylabel("Probabilities")
    plt.legend()
    plt.savefig(f"plots/unitarity_check_n_qubits={n_qubits}.pdf")
    plt.show()
    plt.close()
    

    










        



if __name__ == "__main__":
    verify_unitarity(3)
    
    # apply_to_ket0_and_plot_hist(3, run_type="qft")
    # apply_to_ket0_and_plot_hist(3, run_type="iqft")

    # main("qft")
    # random_validation("iqft")
