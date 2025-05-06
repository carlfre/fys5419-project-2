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
            f"|{binary_state}> ({i}): ampl: {amplitude:.4f} prob: {probability:.4f} Phase: {phase:.2f}"
        )

def check_unitarity(circuit: QuantumCircuit, tol=1e-10) -> bool:
    """Checks if the circuit's unitary matrix satisfies U†U = I."""
    # Create a copy of the circuit to avoid modifying the original
    circuit_copy = circuit.copy()
    simulator = AerSimulator(method='unitary')
    # Add save_unitary to the copied circuit
    circuit_copy.save_unitary()
    job = simulator.run(circuit_copy)
    result = job.result()
    try:
        unitary = result.get_unitary()
        # Convert Operator to NumPy array to avoid deprecation warning
        unitary_array = np.asarray(unitary)
        # Compute conjugate transpose (dagger)
        unitary_dagger = unitary_array.conj().T
        # Compute product U†U
        product = np.dot(unitary_dagger, unitary_array)
        # Check if product is close to identity
        identity = np.eye(2**circuit.num_qubits)
        if np.linalg.norm(product - identity) < tol:
            print("Good, the matrix is unitary!")
        elif np.linalg.norm(product - identity) > tol:
            print("Not Good, the matrix is not unitary!")
        elif np.linalg.norm(product - identity) == tol:
            print("The matrix is straight on unitary tolerence!")
        return 
    except QiskitError as e:
        print(f"Unitarity check error: {e}")
        return False

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

def create_qft_circuits_dict(n):
    circuits = {
        f"qc_qft_{n}":  (QFT(num_qubits = n, do_swaps=True, inverse=False), f"{n}-Qubit QFT", n),
        f"qc_iqft_{n}": (QFT(num_qubits = n, do_swaps=True, inverse=True), f"{n}-Qubit IQFT", n)
    }
    return circuits

if __name__ == '__main__':
    # Test for n = 4 as an example
    n = 6
    qc_qft = qft_nq(n)
    qc_iqft = iqft_nq(n)

    names = [f"{n}-Qubit QFT", f"{n}-Qubit IQFT"]
    listqftandiqft = [qc_qft,qc_iqft]
    numqubits = [n,n]

    for i, j, k in zip(names, listqftandiqft, numqubits):
        print(i)          # Use the dictionary value (e.g., "1-Qubit QFT")
        print(j)          # Prints the key (e.g., qc_qft_1 object)
        # Check unitarity
        is_unitary = check_unitarity(j)
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
    circuits_dict = create_qft_circuits_dict(n)

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
    