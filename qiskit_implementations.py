import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT, PhaseEstimation, UnitaryGate
from qiskit_aer import AerSimulator

from utils import sorted_dict
import scipy.sparse as sp


def qft_qiskit(psi: np.ndarray, n_shots: int) -> dict[int, int]:
    """QFT implementation from Qiskit"""

    num_qubits = round(np.log2(psi.shape[0]))


    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.initialize(psi, range(num_qubits))
    qc.append(QFT(num_qubits), range(num_qubits))
    qc.measure(range(num_qubits), range(num_qubits))

    simulator = AerSimulator()
    compiled_circuit = transpile(qc, simulator)
    job = simulator.run(compiled_circuit, shots=n_shots)
    result = job.result()
    counts = result.get_counts()
    return dict(counts)


def inverse_qft_qiskit(psi: np.ndarray, n_shots: int) -> dict[str, int]:
    """Inverse QFT implementation from Qiskit"""
    num_qubits = round(np.log2(psi.shape[0]))

    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.initialize(psi, range(num_qubits))

    qc.append(QFT(num_qubits).inverse(), range(num_qubits))
    qc.measure(range(num_qubits), range(num_qubits))

    simulator = AerSimulator()
    compiled_circuit = transpile(qc, simulator)
    job = simulator.run(compiled_circuit, shots=n_shots)
    result = job.result()
    counts = result.get_counts()
    return dict(counts)


def phase_estimation_qiskit(U: QuantumCircuit | np.ndarray | sp.spmatrix, u: np.ndarray, t: int, n_shots: int) -> dict[str, int]:
    """Phase estimation implementation from Qiskit."""
    if isinstance(U, sp.spmatrix):
        U = U.todense()
    if isinstance(U, np.ndarray):
        U = UnitaryGate(U)

    num_qubits_U = U.num_qubits
    n_qubits_total = num_qubits_U + t

    qc = QuantumCircuit(n_qubits_total, t)
    qc.initialize(u, range(t, n_qubits_total))
    
    pe = PhaseEstimation(t, U)
    qc.append(pe, qc.qubits)
    qc.measure(range(t), range(t))

    simulator = AerSimulator()
    compiled_circuit = transpile(qc, simulator)
    job = simulator.run(compiled_circuit, shots=n_shots)
    result = job.result()
    counts = result.get_counts()
    return sorted_dict({"0." + k[::-1]: v for k, v in counts.items()})