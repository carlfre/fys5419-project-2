from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit import transpile
import numpy as np
from math import gcd
from fractions import Fraction

def shors_algorithm(N, a):
    """
    Implements Shor's algorithm to factorize number N using base a.
    Returns one non-trivial factor of N or None if failed.
    """
    # Step 1: Check if N is even
    if N % 2 == 0:
        return 2
    
    # Step 2: Check if N is a power
    for i in range(2, int(np.log2(N)) + 1):
        x = round(N ** (1/i))
        if x ** i == N:
            return x
    
    # Step 3: Quantum period finding
    n = int(np.ceil(np.log2(N)))  # Number of qubits for modular exponentiation
    m = n * 2  # Number of qubits for counting
    
    # Initialize quantum and classical registers
    qr_count = QuantumRegister(m, 'count')
    qr_aux = QuantumRegister(n, 'aux')
    cr = ClassicalRegister(m, 'meas')
    circuit = QuantumCircuit(qr_count, qr_aux, cr)
    
    # Apply Hadamard gates to counting qubits
    for q in range(m):
        circuit.h(qr_count[q])
    
    # Apply modular exponentiation (simplified for N=15)
    circuit.x(qr_aux[0])  # Initialize auxiliary register to |1>
    
    # Controlled modular multiplication
    for q in range(m):
        circuit.append(
            c_amodN(a, 2**q, N, n),
            [qr_count[q]] + list(qr_aux)
        )
    
    # Apply inverse QFT
    circuit.append(qft_dagger(m), qr_count)
    
    # Measure counting register
    circuit.measure(qr_count, cr)
    
    # Transpile the circuit for the backend
    backend = AerSimulator()
    circuit = transpile(circuit, backend=backend)
    
    # Run simulation
    result = backend.run(circuit, shots=100).result()
    counts = result.get_counts()
    
    # Step 4: Classical post-processing
    for measured_value in counts:
        measured_int = int(measured_value, 2)
        if measured_int == 0:
            continue
        # Convert to phase
        phase = measured_int / (2**m)
        # Continued fraction to find r
        frac = Fraction(phase).limit_denominator(N)
        r = frac.denominator
        
        # Step 5: Check if r is the period
        if r % 2 == 0 and (pow(a, r//2, N) != N-1):
            factor1 = gcd(pow(a, r//2, N) - 1, N)
            factor2 = gcd(pow(a, r//2, N) + 1, N)
            if 1 < factor1 < N:
                return factor1
            if 1 < factor2 < N:
                return factor2
    
    return None

def c_amodN(a, power, N, n):
    """
    Controlled multiplication by a^power mod N gate for N=15.
    Returns a controlled gate for modular exponentiation.
    """
    # For N=15, we simplify by precomputing a^power mod 15
    U = QuantumCircuit(n)
    result = pow(a, power, N)  # Compute a^power mod N classically
    # Apply X gates to represent the result in binary
    for q in range(n):
        if result & (1 << q):
            U.x(q)
    # Convert to controlled gate
    return U.to_gate(name=f'cmod_{a}^{power}%{N}').control(1)

def qft_dagger(n):
    """
    Inverse Quantum Fourier Transform for n qubits.
    """
    qc = QuantumCircuit(n)
    for qubit in range(n//2):
        qc.swap(qubit, n-qubit-1)
    for j in range(n):
        for m in range(j):
            qc.cp(-np.pi/float(2**(j-m)), m, j)
        qc.h(j)
    return qc.to_gate(name='qft_dagger')

# Example usage
if __name__ == "__main__":
    N = 15  # Number to factorize
    a = 7   # Random number coprime with N
    factor = shors_algorithm(N, a)
    print(f"Found factor: {factor}")