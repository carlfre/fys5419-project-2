from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.result import Result
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Statevector  
from qiskit.circuit.library import QFT
import numpy as np
import cmath


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
            f"|{binary_state}> ({i}): ampl: {amplitude:.8f} prob: {probability:.4f} Phase: {phase:.2f}"
        )


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
    #qc.swap(0, 1)  # Swap to match QFT ordering
    return qc


def iqft_2q():
    qc = QuantumCircuit(2, name='IQFT_2')
    #qc.swap(0, 1)  # Reverse the swap
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
    #qc.swap(0, 2)
    return qc


def iqft_3q():
    qc = QuantumCircuit(3, name='IQFT_3')
    # Reverse swaps
    #qc.swap(0, 2)
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

def create_qft_circuits_dict():
    circuits = {
        "qc_qft_1":  (QFT(num_qubits = 1, do_swaps=False, inverse=False), "1-Qubit QFT", 1),
        "qc_iqft_1": (QFT(num_qubits = 1, do_swaps=False, inverse=True), "1-Qubit IQFT", 1),
        "qc_qft_2":  (QFT(num_qubits = 2, do_swaps=False, inverse=False), "2-Qubit QFT", 2),
        "qc_iqft_2": (QFT(num_qubits = 2, do_swaps=False, inverse=True), "2-Qubit IQFT", 2),
        "qc_qft_3":  (QFT(num_qubits = 3, do_swaps=False, inverse=False), "3-Qubit QFT", 3),
        "qc_iqft_3": (QFT(num_qubits = 3, do_swaps=False, inverse=True), "3-Qubit IQFT", 3)
    }
    return circuits

if __name__ == '__main__':
    # Test 1-Qubit QFT and Inverse QFT
    qc_qft_1 = qft_1q()
    qc_iqft_1 = iqft_1q()
     # Test 2-Qubit QFT and Inverse QFT
    qc_qft_2 = qft_2q()
    qc_iqft_2 = iqft_2q()
    # Test 3-Qubit QFT and Inverse QFT
    qc_qft_3 = qft_3q()
    qc_iqft_3 = iqft_3q()

    names = ["1-Qubit QFT", "1-Qubit IQFT", "2-Qubit QFT", "2-Qubit IQFT", "3-Qubit QFT", "3-Qubit IQFT"]
    listqftandiqft = [qc_qft_1, qc_iqft_1, qc_qft_2, qc_iqft_2, qc_qft_3, qc_iqft_3]
    numqubits = [1,1,2,2,3,3]


    for i, j, k in zip(names, listqftandiqft, numqubits):
        print(i)          # Use the dictionary value (e.g., "1-Qubit QFT")
        print(j)          # Prints the key (e.g., qc_qft_1 object)
        print("Statevector:")
        try:
            result = run_and_get_statevector(j)
            statevector = result.get_statevector()
            print_formatted_statevector(statevector, k)
            print()
        except QiskitError as e:
            print(f"  Error: {e}")


#----------Qiskit's inbuilt QFT and IQFT-----------#
    # Create circuits
    circuits_dict = create_qft_circuits_dict()

    # Loop with decomposition
    for name, (circuit, label, j) in circuits_dict.items():
        print(f"{label}:")
        print("Statevector:")
        try:
            # Decompose the QFT circuit into basic gates
            decomposed_circuit = circuit.decompose()
            result = run_and_get_statevector(decomposed_circuit)
            statevector = result.get_statevector()
            print_formatted_statevector(statevector, j)
            print()
        except QiskitError as e:
            print(f"  Error: {e}")