from qiskit import QuantumCircuit
import numpy as np
from qiskit_aer import AerSimulator
from qiskit.result import Result
from qiskit.exceptions import QiskitError
import cmath
from qiskit.quantum_info import Statevector  # Import Statevector


def run_and_get_statevector(circuit: QuantumCircuit) -> Result:
    """Simulates the circuit and returns the result."""
    simulator = AerSimulator(method='statevector')
    # Add save_statevector to the circuit
    circuit.save_statevector()
    job = simulator.run(circuit, shots=1)  # shots=1 to get a single statevector
    result = job.result()
    return result


def print_formatted_statevector(statevector: Statevector, num_qubits: int):
    """Prints the statevector in the desired format."""

    for i in range(2**num_qubits):
        binary_state = bin(i)[2:].zfill(num_qubits)  # Binary representation
        amplitude = statevector[i]  # Access amplitude using Statevector indexing
        probability = np.abs(amplitude) ** 2
        phase = cmath.phase(amplitude)

        print(
            f"|{binary_state}> ({i}): ampl: {amplitude:.4f} prob: {probability:.4f} Phase: {phase:.2f}"
        )


# Generalized QFT
def qft_nq(n):
    qc = QuantumCircuit(n, name=f'QFT_{n}')
    for j in range(n):
        qc.h(j)  # Hadamard on qubit j
        for k in range(j + 1, n):
            angle = np.pi / (2 ** (k - j))  # Phase for R_{k-j}
            qc.cp(angle, j, k)
    # Reverse the order with swaps
    for j in range(n // 2):
        qc.swap(j, n - 1 - j)
    return qc


# Generalized Inverse QFT
def iqft_nq(n):
    qc = QuantumCircuit(n, name=f'IQFT_{n}')
    # Reverse the swaps
    for j in range(n // 2):
        qc.swap(j, n - 1 - j)
    # Apply gates in reverse order with conjugate phases
    for j in range(n - 1, -1, -1):
        for k in range(n - 1, j, -1):
            angle = -np.pi / (2 ** (k - j))  # Negative phase for inverse
            qc.cp(angle, j, k)
        qc.h(j)  # Hadamard on qubit j
    return qc


if __name__ == '__main__':
    # Test for n = 4 as an example
    n = 4
    qc_qft = qft_nq(n)
    qc_iqft = iqft_nq(n)

    print(f"{n}-Qubit QFT:")
    print(qc_qft)
    print("Statevector:")
    try:
        result = run_and_get_statevector(qc_qft)
        statevector = result.get_statevector()
        print_formatted_statevector(statevector, n)
    except QiskitError as e:
        print(f"  Error: {e}")

    print(f"\n{n}-Qubit Inverse QFT:")
    print(qc_iqft)
    print("Statevector:")
    try:
        result = run_and_get_statevector(qc_iqft)
        statevector = result.get_statevector()
        print_formatted_statevector(statevector, n)
    except QiskitError as e:
        print(f"  Error: {e}")

    