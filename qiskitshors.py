from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT
from qiskit_aer import AerSimulator # Using Aer simulator for counts
from qiskit.circuit.exceptions import CircuitError # Import CircuitError
import math
import random
import fractions
import time

# --- Helper Functions (gcd, c_amodN_15) remain the same ---
def gcd(a, b):
    """Compute the greatest common divisor of a and b."""
    while b:
        a, b = b, a % b
    return a

def c_amodN_15(a, power):
    """
    Creates a controlled-U gate for C-U_{a^power} mod 15.
    Hardcoded for N=15 based on Qiskit Textbook examples.
    Requires 4 target qubits.
    """
    N = 15
    n_target = 4 # N=15 needs 4 qubits
    a_pow = pow(a, power, N)

    U = QuantumCircuit(n_target, name=f"U_{a}^{power}mod{N}")

    # Circuits for N=15, a=7, 11, 13, 14 (and derived powers)
    if a_pow == 7:
        U.swap(2, 3); U.swap(1, 2); U.swap(0, 1)
        for q in range(n_target): U.x(q)
    elif a_pow == 11:
        U.swap(0, 3); U.swap(1, 2)
        for q in range(n_target): U.x(q)
    elif a_pow == 13:
        U.swap(1, 3); U.swap(0, 2)
        for q in range(n_target): U.x(q)
    elif a_pow == 14: # a=2^1 -> a_pow=2; a=2^2 -> a_pow=4; a=2^3 -> a_pow=8; a=2^4 -> a_pow=1
        U.swap(0, 1); U.swap(2, 3)
        for q in range(n_target): U.x(q)
    elif a_pow == 1: # Identity
        pass
    elif a_pow == 4:
        U.swap(0, 2); U.swap(1, 3)
    elif a_pow == 2:
        U.swap(0,1); U.swap(1,2); U.swap(2,3)
    elif a_pow == 8:
        U.swap(2,3); U.swap(1,2); U.swap(0,1)
    else:
         raise NotImplementedError(f"Modular exponentiation for a^power={a_pow} mod 15 not implemented.")

    C_U = U.to_gate().control(1)
    C_U.name = f"C-U({a}^{power}mod{N})"
    return C_U

# --- Main Shor's Algorithm Steps ---

def run_shors(N=15, shots=2048):
    """Runs Shor's algorithm to factor N."""

    print(f"--- Attempting to factor N={N} ---")
    start_time = time.time()

    # *** AMENDMENT: Check if N is supported by the hardcoded function ***
    if N!= 15:
        print(f"Error: This implementation's quantum circuit (c_amodN_15) is hardcoded only for N=15.")
        print(f"Cannot factor N={N} with this specific code.")
        print("Implementing Shor's for general N requires a complex general modular exponentiation circuit.")
        return None
    # *********************************************************************

    # 1. Classical Pre-processing
    if N % 2 == 0:
        print(f"Factor found classically: 2 and {N//2}")
        return 2, N // 2

    a = random.randint(2, N - 1)
    print(f"Trying base a = {a}")
    g = gcd(a, N)
    if g > 1:
        print(f"Factor found classically: {g} and {N//g}")
        return g, N // g
    print(f"Base a={a} is coprime to N={N}. Proceeding to quantum part.")

    # 2. Quantum Period Finding
    n_target = math.ceil(math.log2(N)) # Workspace qubits (will be 4 since N=15)
    t = 2 * n_target                   # Counting qubits (will be 8)
    total_qubits = t + n_target

    # Create circuit
    qc = QuantumCircuit(total_qubits, t, name="Shor Period Finding")

    # Initialize workspace to |1> (qubit t is LSB of workspace)
    qc.x(t)

    # Hadamard gates on counting register (qubits 0 to t-1)
    qc.h(range(t))
    qc.barrier()

    # Controlled Modular Exponentiation
    for k in range(t):
        control_qubit = k
        target_qubits = list(range(t, total_qubits))
        power_of_2 = 2**k
        # We know N=15 here, so c_amodN_15 is appropriate
        controlled_unitary = c_amodN_15(a, power_of_2)
        # The append should now work because n_target will be 4,
        # matching the 5-qubit expectation (1 control + 4 target)
        try:
            qc.append(controlled_unitary, [control_qubit] + target_qubits)
        except CircuitError as e:
             # This catch is now less likely for N=15, but good practice
             print(f"\nCircuitError during append (should not happen for N=15): {e}")
             print("There might be an issue with qubit indexing or the c_amodN_15 function.")
             return None
        except NotImplementedError as e:
             # Catch if c_amodN_15 doesn't support the specific a_pow value
             print(f"\nError: {e}")
             print(f"The base a={a} resulted in a power {pow(a, power_of_2, N)} not handled by c_amodN_15.")
             print("Try running again with a different random 'a'.")
             return None # Indicate failure for this run

    qc.barrier()

    # Inverse QFT on counting register
    iqft_gate = QFT(t, inverse=True, do_swaps=True, name='IQFT') # Use swaps for easier interpretation
    qc.append(iqft_gate, range(t))
    qc.barrier()

    # Measure counting register
    qc.measure(range(t), range(t))

    # print(qc.draw(output='text', fold=-1)) # Uncomment to see the circuit

    # 3. Execute Circuit
    print("Executing quantum circuit...")
    simulator = AerSimulator()
    compiled_circuit = transpile(qc, simulator)
    job = simulator.run(compiled_circuit, shots=shots)
    result = job.result()
    counts = result.get_counts(qc)
    print(f"Measurement results (counts): {counts}")

    # 4. Classical Post-processing: Find Period 'r'
    candidate_periods = {}
    for output_str, count in counts.items():
        # Ensure output_str has the correct length 't' before converting
        if len(output_str)!= t:
            print(f"Warning: Skipping unexpected measurement result format: '{output_str}'")
            continue
        m = int(output_str, 2) # Measured integer
        if m == 0: continue

        phase = m / (2**t)
        try:
            frac = fractions.Fraction(phase).limit_denominator(N) # Continued fractions
        except (ZeroDivisionError, OverflowError):
             print(f"Warning: Could not compute fraction for phase {phase}. Skipping outcome {output_str}.")
             continue

        r = frac.denominator
        if r > 0:
            candidate_periods[r] = candidate_periods.get(r, 0) + count

    if not candidate_periods:
        print("Period finding failed (no valid period candidates found from measurements).")
        # This is where a message like "No common denominators" might fit conceptually,
        # although the exact wording comes from the continued fraction failing to find
        # a suitable denominator 'r'.
        return None

    # Find most likely period
    r = max(candidate_periods, key=candidate_periods.get)
    print(f"Most likely period candidate: r = {r}")

    # 5. Calculate Factors
    # Perform validation: a^r mod N == 1?
    if pow(a, r, N)!= 1:
        print(f"Validation failed: a^r mod N!= 1 ({pow(a, r, N)}!= 1). Period candidate r={r} is likely incorrect.")
        print("This might be due to measurement noise or insufficient shots.")
        return None # Indicate failure for this run

    if r % 2!= 0:
        print(f"Period r={r} is odd. Factorization attempt failed for this 'a'.")
        return None # Indicate failure for this run

    x = pow(a, r // 2, N)
    if x == (N - 1):
        print(f"Period r={r} leads to trivial factors (x = -1 mod N). Factorization attempt failed for this 'a'.")
        return None # Indicate failure for this run

    p = gcd(x - 1, N)
    q = gcd(x + 1, N)

    if p!= 1 and p!= N:
        print(f"Success! Factors found: {p} and {q}")
        end_time = time.time()
        print(f"Total time: {end_time - start_time:.2f} seconds.")
        return p, q
    else:
        # Handle the case where gcd gives trivial factors (less common if r is correct)
        print(f"Factorization attempt failed (found trivial factors p={p}, q={q} from r={r}).")
        return None # Indicate failure for this run


# --- Run the Algorithm ---
if __name__ == "__main__":
    # --- Test N=15 (Should work) ---
    print("\nTesting N=15...")
    factors_15 = run_shors(N=15)
    if factors_15:
        print(f"Final Factors for N=15: {factors_15}")
    else:
        print("Could not find factors for N=15 with this run. Try running again.")

    # --- Test N=21 (Should print error and stop) ---
    print("\nTesting N=21...")
    factors_21 = run_shors(N=21)
    if factors_21:
         print(f"Final Factors for N=21: {factors_21}") # Should not reach here
    else:
         print("Exited gracefully for N=21 as expected.")

    # --- Test N=25 (Should print error and stop) ---
    print("\nTesting N=25...")
    factors_25 = run_shors(N=25)
    if factors_25:
         print(f"Final Factors for N=25: {factors_25}") # Should not reach here
    else:
         print("Exited gracefully for N=25 as expected.")

    # --- Test N=30 (Should factor classically) ---
    print("\nTesting N=30...")
    factors_30 = run_shors(N=30)
    if factors_30:
         print(f"Final Factors for N=30: {factors_30}")
    else:
         print("Could not find factors for N=30 (classical factoring should have worked).") # Should not reach here