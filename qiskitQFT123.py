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


def qft_1q():
    qc = QuantumCircuit(1, name='QFT_1')
    qc.h(0)  # Apply Hadamard to qubit 0
    return qc


def iqft_1q():
    qc = QuantumCircuit(1, name='IQFT_1')
    qc.h(0)  # Hadamard is its own inverse
    return qc


def qft_2q():
    qc = QuantumCircuit(2, name='QFT_2')
    qc.h(0)  # Hadamard on qubit 0
    qc.cp(np.pi / 2, 0, 1)  # Controlled R2 (phase = π/2) from q0 to q1
    qc.h(1)  # Hadamard on qubit 1
    qc.swap(0, 1)  # Swap to match QFT ordering
    return qc


def iqft_2q():
    qc = QuantumCircuit(2, name='IQFT_2')
    qc.swap(0, 1)  # Reverse the swap
    qc.h(1)  # Hadamard on qubit 1
    qc.cp(-np.pi / 2, 0, 1)  # Controlled R2† (phase = -π/2) from q0 to q1
    qc.h(0)  # Hadamard on qubit 0
    return qc


def qft_3q():
    qc = QuantumCircuit(3, name='QFT_3')
    # Qubit 0
    qc.h(0)
    qc.cp(np.pi / 2, 0, 1)  # R2 from q0 to q1
    qc.cp(np.pi / 4, 0, 2)  # R3 from q0 to q2
    # Qubit 1
    qc.h(1)
    qc.cp(np.pi / 2, 1, 2)  # R2 from q1 to q2
    # Qubit 2
    qc.h(2)
    # Swaps for ordering (reverse all qubits)
    qc.swap(0, 2)
    return qc


def iqft_3q():
    qc = QuantumCircuit(3, name='IQFT_3')
    # Reverse swaps
    qc.swap(0, 2)
    # Qubit 2
    qc.h(2)
    # Qubit 1
    qc.cp(-np.pi / 2, 1, 2)  # R2† from q1 to q2
    qc.cp(-np.pi / 4, 0, 2)  # R3† from q0 to q2
    qc.h(1)
    # Qubit 0
    qc.cp(-np.pi / 2, 0, 1)  # R2† from q0 to q1
    qc.h(0)
    return qc


def print_formatted_statevector(statevector: Statevector, num_qubits: int):
    """Prints the statevector in the desired format."""

    for i in range(2**num_qubits):
        binary_state = bin(i)[2:].zfill(num_qubits)  # Binary representation
        amplitude = statevector[i]  # Access amplitude using Statevector indexing
        probability = np.abs(amplitude) ** 2
        phase = cmath.phase(amplitude)

        print(
            f"|{binary_state}> ({i}): ampl: {amplitude:.3f} prob: {probability:.3f} Phase: {phase:.2f}"
        )


if __name__ == '__main__':
    # Test 1-Qubit QFT and Inverse QFT
    qc_qft_1 = qft_1q()
    qc_iqft_1 = iqft_1q()

    print("1-Qubit QFT:")
    print(qc_qft_1)
    print("Statevector:")
    try:
        result = run_and_get_statevector(qc_qft_1)
        statevector = result.get_statevector()
        print_formatted_statevector(statevector, 1)
    except QiskitError as e:
        print(f"  Error: {e}")

    print("\n1-Qubit Inverse QFT:")
    print(qc_iqft_1)
    print("Statevector:")
    try:
        result = run_and_get_statevector(qc_iqft_1)
        statevector = result.get_statevector()
        print_formatted_statevector(statevector, 1)
    except QiskitError as e:
        print(f"  Error: {e}")

    # Test 2-Qubit QFT and Inverse QFT
    qc_qft_2 = qft_2q()
    qc_iqft_2 = iqft_2q()

    print("\n2-Qubit QFT:")
    print(qc_qft_2)
    print("Statevector:")
    try:
        result = run_and_get_statevector(qc_qft_2)
        statevector = result.get_statevector()
        print_formatted_statevector(statevector, 2)
    except QiskitError as e:
        print(f"  Error: {e}")

    print("\n2-Qubit Inverse QFT:")
    print(qc_iqft_2)
    print("Statevector:")
    try:
        result = run_and_get_statevector(qc_iqft_2)
        statevector = result.get_statevector()
        print_formatted_statevector(statevector, 2)
    except QiskitError as e:
        print(f"  Error: {e}")

    # Test 3-Qubit QFT and Inverse QFT
    qc_qft_3 = qft_3q()
    qc_iqft_3 = iqft_3q()

    print("\n3-Qubit QFT:")
    print(qc_qft_3)
    print("Statevector:")
    try:
        result = run_and_get_statevector(qc_qft_3)
        statevector = result.get_statevector()
        print_formatted_statevector(statevector, 3)
    except QiskitError as e:
        print(f"  Error: {e}")

    print("\n3-Qubit Inverse QFT:")
    print(qc_iqft_3)
    print("Statevector:")
    try:
        result = run_and_get_statevector(qc_iqft_3)
        statevector = result.get_statevector()
        print_formatted_statevector(statevector, 3)
    except QiskitError as e:
        print(f"  Error: {e}")