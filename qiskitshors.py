import math
import random
from fractions import Fraction
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.providers.basic_provider import BasicSimulator # Basic simulator
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Operator

# --- Classical Helper Functions ---

def gcd(a, b):
    """Computes the greatest common divisor of a and b."""
    return math.gcd(a, b)

def power(a, d, n):
    """Computes (a^d) % n efficiently."""
    res = 1
    a = a % n
    while d > 0:
        if d % 2 == 1:
            res = (res * a) % n
        a = (a * a) % n
        d //= 2
    return res

# --- Classical Pre-processing Functions ---

def miller_rabin_test(d, n):
    """Single round of Miller-Rabin primality test."""
    a = 2 + random.randint(1, n - 4) # Witness between 2 and n-2
    x = power(a, d, n)
    if x == 1 or x == n - 1:
        return True
    while d!= n - 1:
        x = (x * x) % n
        d *= 2
        if x == 1:
            return False
        if x == n - 1:
            return True
    return False

def is_prime(n, k=40):
    """Probabilistic primality test using Miller-Rabin."""
    if n <= 1 or n == 4: return False
    if n <= 3: return True
    if n % 2 == 0: return False # Handle even numbers

    d = n - 1
    while d % 2 == 0:
        d //= 2

    for _ in range(k):
        if not miller_rabin_test(d, n):
            return False
    return True

def integer_nth_root(n, k):
    """Finds integer k-th root using binary search (simplified)."""
    if n < 0: return -1, False # Not handled here
    if n == 0: return 0, True
    if n == 1: return 1, True
    if k == 1: return n, True

    # Estimate high bound more tightly if possible
    high = 1 << (n.bit_length() // k + 1)
    low = 0
    while low < high:
        mid = (low + high) // 2
        if mid == 0: # Avoid 0**k issues if low becomes 0
             low = 1
             continue
        try:
            p = mid**k
        except OverflowError:
            p = float('inf')

        if p == n:
            return mid, True
        elif p < n:
            low = mid + 1
        else:
            high = mid
    # After loop, low is the smallest integer s.t. low**k >= n
    # Check if (low-1)**k == n
    root = low - 1
    if root > 0:
        try:
            if root**k == n:
                return root, True
        except OverflowError:
             pass # Ignore overflow on final check
    return -1, False # No exact integer root found


def is_perfect_power(n):
    """Checks if n = b^e for integers b>1, e>1."""
    if n <= 3: return None
    limit = int(math.log2(n)) + 1 # Max possible exponent
    for k in range(2, limit):
        root, is_exact = integer_nth_root(n, k)
        if is_exact:
            return (root, k) # Return base and exponent
    return None

def shor_classical_preproc(N):
    """Performs classical checks before running quantum part."""
    if N <= 1:
        print(f"Input N={N} must be greater than 1.")
        return None, "InvalidInput"
    if N % 2 == 0:
        print(f"N={N} is even. Factors found classically: (2, {N//2})")
        return (2, N // 2), "EvenFactorFound"
    if is_prime(N):
        print(f"N={N} is prime. No non-trivial factors.")
        return None, "Prime"
    power_check = is_perfect_power(N)
    if power_check:
        base, exp = power_check
        print(f"N={N} is a perfect power: {base}^{exp}. Factors derived classically.")
        # Could add logic to find prime factors from base if needed
        return (base, N // base), "PerfectPowerFactorFound" # Example return

    print(f"N={N} is composite and not a perfect power. Proceeding to quantum period finding.")
    return None, "Proceed"

# --- Quantum Period Finding ---

def c_amodN_unitary(a, power_of_a, N, n_work):
    """
    Creates the controlled unitary gate for (x * a**power_of_a) mod N
    using the Operator class.
    NOTE: This is specific to the chosen 'a' and N.
          It constructs the full unitary matrix, which is inefficient for large N.
    Args:
        a (int): The base number.
        power_of_a (int): The power to which 'a' is raised (e.g., 2**j).
        N (int): The number to factor.
        n_work (int): The number of qubits in the work register (ceil(log2(N))).
    Returns:
        Gate: The controlled Qiskit Gate.
    """
    num_states = 2**n_work
    op_matrix = np.zeros((num_states, num_states), dtype=complex)
    
    # Calculate a^(power_of_a) mod N
    a_pow = pow(a, power_of_a, N)

    # Define the permutation matrix for U|y> = |ay mod N>
    for y in range(num_states):
        if y < N: # Only apply operation if y < N
            target_y = (y * a_pow) % N
        else: # Map states >= N to themselves (identity)
            target_y = y
        op_matrix[target_y, y] = 1 # op_matrix[output, input]

    # Create the Unitary gate
    U = Operator(op_matrix)
    gate = QuantumCircuit(n_work, name=f"U_{a}^{power_of_a}_mod{N}")
    gate.unitary(U, range(n_work))
    
    # Create the controlled version
    c_U = gate.control(1) # 1 control qubit
    return c_U


def qft_dagger(n):
    """n-qubit inverse QFT on the first n qubits in circuit."""
    qc = QuantumCircuit(n)
    # Don't forget the Swaps! (if needed, or handle classically)
    # Qiskit's QFT.inverse() handles this internally or via do_swaps
    # Implementing manually for illustration (matching textbook):
    for qubit in range(n//2):
        qc.swap(qubit, n-qubit-1)
    for j in range(n):
        for m in range(j):
            qc.cp(-math.pi/float(2**(j-m)), m, j)
        qc.h(j)
    qc.name = "IQFT"
    return qc

def quantum_period_finder(N, a):
    """Constructs and runs the quantum period finding circuit for N."""
    
    n_work = math.ceil(math.log2(N)) # Work qubits: n = ceil(log2(N))
    t = 2 * n_work           # Counting qubits: t = 2n (recommended)

    print(f"Factoring N={N} with a={a}.")
    print(f"Using {t} counting qubits and {n_work} work qubits.")

    qc = QuantumCircuit(t + n_work, t) # Circuit with classical register for counting qubits

    # Initialize counting qubits in superposition
    for q in range(t):
        qc.h(q)

    # Initialize work register to |1> (e.g., |0...01>)
    qc.x(t + n_work - 1) # Set last work qubit to 1

    qc.barrier()

    # Apply controlled modular exponentiation: U^x|1> = |a^x mod N>
    for q in range(t): # Iterate over counting qubits (controls)
        # Apply controlled U^(a^(2^q)) mod N
        exponent = 2**q
        try:
            # Use the unitary-based function
            controlled_U = c_amodN_unitary(a, exponent, N, n_work)
            # Apply: control qubit is q, target qubits are t to t+n_work-1
            qc.append(controlled_U, [q] + list(range(t, t + n_work)))
        except Exception as e:
             print(f"Error creating/appending controlled unitary for N={N}, a={a}, exponent=2^{q}: {e}")
             return None # Indicate failure

    qc.barrier()

    # Apply inverse QFT to counting register
    # Using Qiskit's built-in IQFT is recommended for general use:
    # from qiskit.circuit.library import QFT
    # iqft_gate = QFT(num_qubits=t, inverse=True, do_swaps=False).to_gate()
    # iqft_gate.name = "IQFT"
    # qc.append(iqft_gate, range(t))
    # Using manual implementation from textbook for consistency:
    qc.append(qft_dagger(t), range(t))

    qc.barrier()

    # Measure counting qubits
    qc.measure(range(t), range(t))

    # --- Simulation ---
    # Use BasicSimulator for simplicity, or AerSimulator/Sampler for performance
    print("Simulating quantum circuit...")
    # Ensure BasicSimulator is available or switch to Aer
    try:
        sim_backend = BasicSimulator()
    except ImportError:
        print("BasicSimulator not found, trying Aer simulator.")
        sim_backend = AerSimulator()

    # Transpile for the simulator
    qc_transpiled = transpile(qc, sim_backend)
    
    # Optional: Print circuit depth and gate counts
    print("Circuit depth:", qc_transpiled.depth())
    print("Gate counts:", qc_transpiled.count_ops())
        
    job = sim_backend.run(qc_transpiled, shots=1024)
    result = job.result()
    counts = result.get_counts(qc) # Get counts using the original circuit object
    print("Simulation complete.")

    # Optional: Plot histogram
    # from qiskit.visualization import plot_histogram
    # plot_histogram(counts)
    # plt.show()

    return counts, t # Return counts and number of counting qubits 't'

# --- Classical Post-processing ---

def shor_classical_postproc(counts, N, a, t):
    """Processes measurement results to find the period and factors."""
    if counts is None:
        return None, "QuantumFailure"

    measured_values = []# Store {'phase': phase, 'decimal': decimal, 'count': count}
    total_shots = sum(counts.values())

    # Process measurement results (remember QFT output is bit-reversed from definition)
    for output_bin, count in counts.items():
        # Reverse the binary string because QFT output is often reversed
        # Depending on QFT implementation (with/without swaps), adjust if needed.
        # The manual qft_dagger above includes swaps, so no reversal needed here.
        # If using Qiskit's QFT(..., do_swaps=False), use output_bin[::-1]
        decimal_value = int(output_bin, 2)
        phase = decimal_value / (2**t)
        measured_values.append({'phase': phase, 'decimal': decimal_value, 'count': count})

    # Sort by counts descending to process most likely results first
    measured_values.sort(key=lambda x: x['count'], reverse=True)

    print(f"\nClassical Post-processing for N={N}, a={a}")
    print(f"Total shots: {total_shots}")
    print(f"Number of unique measurement outcomes: {len(measured_values)}")
    print(f"Top measurement outcomes (phase = decimal / 2^{t}):")
    for val in measured_values[:min(5, len(measured_values))]: # Print top 5 or fewer
        print(f"  Decimal: {val['decimal']:<5d} (Binary: {val['decimal']:0{t}b}), Phase: {val['phase']:.5f}, Counts: {val['count']}")

    # Try to find period 'r' using continued fractions
    found_period = False
    for item in measured_values:
        phase = item['phase']
        decimal = item['decimal']
        count = item['count']

        # Phase 0 gives no period information
        if phase == 0:
            continue

        print(f"\nProcessing measurement: Decimal={decimal}, Phase={phase:.5f}, Counts={count}")

        # Use continued fractions to find candidate period r
        frac = Fraction(phase).limit_denominator(N) # Key step! Limit denominator to N
        r_candidate = frac.denominator

        print(f"  Continued fraction approximation: {frac} -> Candidate period r={r_candidate}")

        if r_candidate == 0 or r_candidate >= N: # Period must be 0 < r < N
            print(f"  Candidate period r={r_candidate} is invalid (must be 0 < r < N).")
            continue

        # Validate the candidate period: Check if a^r_candidate = 1 (mod N)
        if power(a, r_candidate, N) == 1:
            print(f"  Period candidate r={r_candidate} VALIDATED (a^r == 1 mod N).")
            r = r_candidate
            found_period = True
            # --- Factor finding logic ---
            # Check if period is even
            if r % 2!= 0:
                print(f"  Period r={r} is odd. Cannot use this period to find factors.")
                # We found the period, but it's odd. Might try other measurements,
                # but often indicates we need a different 'a'. Let's stop processing this 'a'.
                return None, "OddPeriodFound"

            # Check non-triviality condition: a^(r/2)!= -1 (mod N)
            x = power(a, r // 2, N)
            if x == N - 1:
                print(f"  Period r={r} leads to trivial factors (a^(r/2) == -1 mod N).")
                # Try next measurement outcome for this 'a'
                continue # Continue processing other measurements for this 'a'

            # Calculate factors: gcd(a^(r/2) +/- 1, N)
            p = gcd(x + 1, N)
            q = gcd(x - 1, N)

            print(f"  Potential factors from r={r}: p=gcd({x}+1, {N})={p}, q=gcd({x}-1, {N})={q}")

            # Check if factors are non-trivial
            if p!= 1 and p!= N:
                print(f"Non-trivial factors found: ({p}, {N//p})")
                return (p, N // p), "FactorsFound"
            elif q!= 1 and q!= N:
                 print(f"Non-trivial factors found: ({q}, {N//q})")
                 return (q, N // q), "FactorsFound"
            else:
                 print(f"  Period r={r} led to trivial factors (p or q is 1 or N).")
                 # Continue processing other measurements for this 'a'
                 continue
            # --- End Factor finding logic ---
        else:
             print(f"  Period candidate r={r_candidate} failed validation (a^r!= 1 mod N).")
             # Continue processing other measurements for this 'a'

    # If loop finishes without finding factors
    if found_period:
         print("Found a valid period, but it led to trivial factors or was odd.")
         return None, "TrivialFactorsOrOddPeriod"
    else:
         print("No valid period found from any measurement outcome for this 'a'.")
         return None, "NoValidPeriodFound"


# --- Main Orchestration Function ---

def factor_shor(N, max_attempts=5):
    """Main function to factor N using Shor's algorithm."""
    print(f"--- Attempting to factor N={N} ---")

    factors, status = shor_classical_preproc(N)
    if status!= "Proceed":
        print(f"Pre-processing status: {status}")
        return factors # Return result from pre-processing

    # If pre-processing passed, proceed to quantum part
    attempts = 0
    possible_factors = None
    while attempts < max_attempts:
        attempts += 1
        print(f"\nAttempt {attempts}/{max_attempts}:")
        # Choose random base 'a' coprime to N
        while True:
            a = random.randint(2, N - 1)
            g = gcd(a, N)
            if g == 1:
                print(f"Trying base a={a} (coprime to N={N})...")
                break
            else:
                # Found factor classically during 'a' selection!
                print(f"Factor found classically during 'a' selection: gcd({a}, {N}) = {g}")
                return (g, N // g)

        # Run quantum period finding
        counts, t = quantum_period_finder(N, a) # Get counts and 't'

        # Post-process results
        if counts is not None:
            possible_factors, status = shor_classical_postproc(counts, N, a, t)
            print(f"Post-processing status for a={a}: {status}")
            if status == "FactorsFound":
                return possible_factors
            # Other statuses (OddPeriodFound, TrivialFactorsOrOddPeriod, NoValidPeriodFound)
            # mean this 'a' didn't work, so we continue to the next attempt.
        else:
            print(f"Quantum simulation failed for a={a}. Trying again.")
            status = "QuantumFailure" # Ensure status is set if counts is None

        # Loop continues to try a new 'a' if factors not found

    print(f"\nFailed to find factors after {max_attempts} attempts.")
    return None

# --- Example Usage ---
if __name__ == "__main__":
    # Test 
    N=21
    factors = factor_shor(N, max_attempts=10) # Increase attempts if needed
    if factors:
        print(f"\nSuccessfully factored {N} into: {factors}")
    else:
        print(f"\nCould not factor {N} with the given attempts.")

