import math
import random
from fractions import Fraction
import numpy as np
import matplotlib.pyplot as plt
import time
from qiskit import QuantumCircuit, transpile
from qiskit.providers.basic_provider import BasicSimulator # Basic simulator
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Operator
# Import standard gates explicitly if needed for decomposition target, though transpile usually handles this
from qiskit.circuit.library import standard_gates

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

def get_divisors(n):
    """Finds all positive divisors of an integer n."""
    if n <= 0:
        return []
    divisors = []
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            divisors.append(i)
            if i * i != n:
                divisors.append(n // i)
    divisors.sort()
    return divisors


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

    # Using math.isqrt for Python 3.8+ is more efficient and robust for k=2
    if k == 2 and hasattr(math, 'isqrt'):
         root = math.isqrt(n)
         if root is not None and root * root == n: # Check square precisely
              return root, True
         return -1, False # Not a perfect square if k=2

    # Generic binary search for k > 2 or older Python versions
    # Use integer division to avoid floats and potential precision issues
    low = 1 # Start from 1
    # Estimate an upper bound: If mid**k > n, then any value > mid is also > n.
    # A safe upper bound is n itself, but we can be more efficient.
    # If n = b^k, then b = n^(1/k). log_2(b) = log_2(n) / k.
    # b approximately 2^(log2(n)/k). Let's use 2^(ceil(log2(n)/k)) or similar
    # A simple loose upper bound is n. A tighter one could be 2 * (n // 2) if n>1
    # A safer high that avoids overflow for reasonable k: high = 2
    # while high**k < n: high *= 2. But this can overflow.
    # A more robust high: For n >= 2, log_k(n) is the exponent. k^exp = n.
    # exp = log(n) / log(k). The root is approximately exp(log(n)/k).
    # A high value could be int(n**(1.0/k)) + 2 if k > 0, but float(n**(1.0/k)) can have precision issues.
    # Let's try a safer binary search range.
    # If mid^k > n, then mid > n^(1/k). The root must be <= mid-1.
    # If mid^k < n, then mid < n^(1/k). The root must be >= mid+1.
    # If n=1000, k=3, root=10. high=1000. log2(1000) approx 10. high = 2**10 = 1024.
    # Let's use min(n, 2**(math.ceil(math.log2(n) / k) + 2)) as a safe upper bound.
    # Or even simpler: low=1, high=n.
    high = n
    while low <= high: # Use <= for potential exact match
        mid = (low + high) // 2
        if mid == 0: # Avoid 0**k issues, especially for k > 1
             low = 1
             continue
        try:
            p = mid**k
        except OverflowError:
            p = float('inf') # Treat overflow as larger than n

        if p == n:
            return mid, True
        elif p < n:
            low = mid + 1
        else: # p > n or OverflowError
            high = mid - 1

    # After the loop, if an exact root exists, it was returned.
    # If the loop finishes, no exact integer root was found in [1, n].
    return -1, False


def is_perfect_power(n):
    """Checks if n = b^e for integers b>1, e>1."""
    if n <= 3: return None
    # Maximum possible exponent 'e' is log2(n) for base b=2.
    # We need to check bases b >= 2.
    # The highest possible exponent is floor(log2(n)).
    # We only need to check exponents k from 2 up to floor(log2(n)).
    limit_k = int(math.log2(n))

    # Iterate through possible exponents k starting from 2
    for k in range(2, limit_k + 1):
        # For a given exponent k, find the potential base 'root'
        root, is_exact = integer_nth_root(n, k)
        if is_exact and root > 1: # Ensure base is > 1
            # If an exact integer k-th root was found and the root is > 1,
            # then n is a perfect power: root^k
            return (root, k) # Return base and exponent
    # If the loop finishes without finding such a root and exponent, n is not a perfect power > 1^e
    return None

def shor_classical_preproc(N):
    """Performs classical checks before running quantum part."""
    if not isinstance(N, int) or N <= 1:
        print(f"Input N={N} must be an integer greater than 1.")
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
        # To be rigorous, the factors of N are the prime factors of the base.
        # For simplicity here, we'll just return the base and N/base.
        # If the base is prime, these are the prime factors.
        # If the base is composite, we'd need to factor the base.
        # Assuming N is composite and not prime (checked above), if it's a perfect power b^e,
        # then b must be a factor, and N/b is another.
        return (base, N // base), "PerfectPowerFactorFound" # Example return, base might be composite

    print(f"N={N} is composite and not a perfect power. Proceeding to quantum period finding.")
    return None, "Proceed"

# --- Quantum Period Finding ---

def c_amodN_unitary(a, power_of_a, N, n_work):
    """
    Creates the controlled unitary gate for (x * a**power_of_a) mod N
    using the Operator class.
    NOTE: This is specific to the chosen 'a' and N.
          It constructs the full unitary matrix, which is inefficient for large N.
          This unitary *will* be decomposed by transpile.
    Args:
        a (int): The base number.
        power_of_a (int): The power to which 'a' is raised (e.g., 2**j).
        N (int): The number to factor.
        n_work (int): The number of qubits in the work register (ceil(log2(N))).
    Returns:
        Gate: The controlled Qiskit Gate.
    """
    # Operator size is 2^n_work x 2^n_work
    num_states = 2**n_work
    op_matrix = np.zeros((num_states, num_states), dtype=complex)

    # Calculate a^(power_of_a) mod N
    a_pow = pow(a, power_of_a, N)

    # Define the permutation matrix for U|y> = |ay mod N>
    # The work register represents states from 0 to 2^n_work - 1.
    # The modular multiplication operation is only relevant for states y < N.
    # States y >= N are mapped to themselves to make the operator unitary
    # without affecting the computation for states < N.
    for y in range(num_states):
        # If the current state y is less than N, apply the modular multiplication.
        if y < N:
            target_y = (y * a_pow) % N
        else:
             # If the current state y is >= N, map it to itself.
             target_y = y
        # Set the matrix element for the transition from state y to state target_y
        op_matrix[target_y, y] = 1 # op_matrix[output, input]

    # Create the Unitary gate from the matrix
    U = Operator(op_matrix)
    # Create a QuantumCircuit containing just this unitary gate on n_work qubits
    gate_name = f"U_{a}^{power_of_a}_mod{N}"
    # Create a shorter name if the power is a power of 2
    log2_power = int(math.log2(power_of_a)) if (power_of_a > 0 and (power_of_a & (power_of_a - 1) == 0)) else None
    if log2_power is not None:
         gate_name = f"U_{a}^(2^{log2_power})_mod{N}"
    # Truncate long names if necessary
    if len(gate_name) > 50: # Arbitrary limit
        gate_name = gate_name[:47] + "..."


    gate = QuantumCircuit(n_work, name=gate_name)
    # Apply the unitary operator to the work register qubits (0 to n_work-1 relative to the gate)
    # The unitary method automatically uses the operator's dimension (2^n_work)
    # and expects a list of n_work qubits.
    gate.unitary(U, range(n_work))

    # Create the controlled version of this gate with 1 control qubit
    c_U = gate.control(1)
    return c_U


def qft_dagger(n):
    """n-qubit inverse QFT on the first n qubits in circuit."""
    qc = QuantumCircuit(n)
    # Manual implementation of IQFT with swaps
    # Apply swaps first
    for qubit in range(n//2):
        qc.swap(qubit, n-qubit-1)
    # Then apply H and controlled rotations
    for j in range(n):
        # Apply controlled phase gates R_k = exp(i * 2*pi / 2^k)
        # The angle for CQFT is -pi/2^(j-m) for the j-th qubit controlled by m-th qubit
        # Here, j is the target qubit (from 0 to n-1), m is the control qubit (from 0 to j-1)
        for m in range(j):
             # The angle for CPhase(phi) |control>|target> is phi
             # For IQFT, the angle applied by qubit m on qubit j is -2*pi / 2^(j-m)
             angle = -2 * math.pi / (2**(j - m))
             # The control qubit is m, target is j
             qc.cp(angle, m, j) # C-Phase gate
        # Apply Hadamard to qubit j
        qc.h(j)
    qc.name = "IQFT"
    return qc

def quantum_period_finder(N, a):
    """
    Constructs and runs a SINGLE quantum period finding circuit instance for N and a.
    Returns counts and t, or None on failure.
    """

    # Determine the number of qubits needed.
    # n: number of work qubits. n = ceil(log2(N)). Stores values up to N-1.
    n_work = math.ceil(math.log2(N))
    # t: number of counting qubits. t is chosen large enough for sufficient precision.
    # A common heuristic is t = 2n. This ensures the probability of measuring k such that
    # k/2^t is close to m/r is high enough for many m.
    t = 2 * n_work

    # Safety check: Ensure N and a are valid inputs for period finding (N>1, a coprime to N, 1 < a < N)
    # This check is also done in the main loop, but included here as a function-level safety.
    if not isinstance(N, int) or N <= 1 or not isinstance(a, int) or a < 2 or a >= N or gcd(a, N) != 1:
         print(f"Internal error: Invalid N={N} or a={a} passed to quantum_period_finder.")
         return None, None # Indicate failure

    # The quantum circuit size grows with t + n_work.
    # The Unitary gate is 2^n_work x 2^n_work. Simulating this directly is the bottleneck.
    # Let's set a limit on the number of work qubits for simulation feasibility.
    max_work_qubits_for_unitary = 6 # Corresponds to N-1 up to 2^6 - 1 = 63
    if n_work > max_work_qubits_for_unitary:
        print(f"N={N} requires {n_work} work qubits ({t+n_work} total). Simulation with explicit unitary matrix is too large.")
        print(f"Limit is {max_work_qubits_for_unitary} work qubits (N <= {2**max_work_qubits_for_unitary - 1}).")
        return None, None # Indicate simulation impossibility with this method


    # Create the quantum circuit
    # We need t qubits for the counting register and n_work qubits for the work register.
    # Total qubits = t + n_work.
    # We also need t classical bits to measure the counting register.
    qc = QuantumCircuit(t + n_work, t)

    # Step 1: Initialize counting qubits in superposition (|h>^t)
    # Apply Hadamard gates to qubits 0 to t-1 (the counting register)
    qc.h(range(t))

    # Step 2: Initialize work register to |1>
    # The work register qubits are indexed from t to t+n_work-1.
    # Apply an X gate to the last qubit of the work register (qubit t + n_work - 1)
    # This assumes the state |1> is represented by |0...01> in n_work qubits.
    qc.x(t + n_work - 1)

    qc.barrier() # Optional: Add a barrier for visualization

    # Step 3: Apply controlled modular exponentiation (U_{a^{2^j}} mod N)
    # Iterate over the counting qubits q from 0 to t-1.
    # The control qubit for the j-th gate (using 0-based indexing for q) is qubit q.
    # The exponent for 'a' in the unitary is 2^q.
    # The target qubits are the work register (qubits t to t+n_work-1).
    for q in range(t):
        # The exponent for 'a' is 2^q based on the position of the control qubit.
        exponent_power_of_2 = 2**q
        try:
            # Create the controlled unitary gate for a^(2^q) mod N.
            # This gate is controlled by 1 qubit and acts on n_work target qubits.
            controlled_U = c_amodN_unitary(a, exponent_power_of_2, N, n_work)
            # Append the controlled gate to the circuit.
            # The first qubit in the list provided to append() is the control qubit.
            # The remaining qubits are the target qubits for the unitary.
            qc.append(controlled_U, [q] + list(range(t, t + n_work)))
        except Exception as e:
             print(f"Error creating/appending controlled unitary for N={N}, a={a}, exponent=2^{q}: {e}")
             # If creating/appending a unitary fails (e.g., due to Operator limits or other issues),
             # the circuit for this 'a' is invalid. Return None to indicate failure.
             return None, None # Indicate failure

    qc.barrier() # Optional: Add a barrier

    # Step 4: Apply inverse QFT to counting register (qubits 0 to t-1)
    # Create the IQFT circuit for 't' qubits.
    iqft_gate = qft_dagger(t)
    # Append the IQFT gate to the circuit, acting on the counting qubits (0 to t-1).
    qc.append(iqft_gate, range(t))

    qc.barrier() # Optional: Add a barrier

    # Step 5: Measure counting qubits (0 to t-1)
    # Measure the counting qubits into the classical register (0 to t-1).
    qc.measure(range(t), range(t))

    # --- Simulation ---
    # print("Simulating quantum circuit...") # Muted for fewer prints per run
    try:
        # Attempt to use AerSimulator for better performance and features
        sim_backend = AerSimulator()
    except ImportError:
        # If AerSimulator is not available, fall back to the BasicSimulator
        # print("AerSimulator not found, falling back to BasicSimulator.") # Muted
        try:
            sim_backend = BasicSimulator()
        except ImportError:
            # If neither is available, inform the user and return None
            print("BasicSimulator also not found. Please install qiskit-aer or basic-provider.")
            return None, None # Indicate simulation failure

    # Transpile the circuit for the simulator backend.
    # This decomposes the custom unitary gates and the IQFT into the simulator's native gates.
    try:
        qc_transpiled = transpile(qc, sim_backend)
        # print("Circuit successfully transpiled.") # Muted
    except Exception as e:
        print(f"Error during circuit transpilation for N={N}, a={a}: {e}")
        # Return None if transpilation fails
        return None, None # Indicate transpilation failure

    # Run the transpiled circuit on the simulator.
    try:
        # Use a fixed number of shots, e.g., 1024 or 2048, for statistical sampling.
        job = sim_backend.run(qc_transpiled, shots=1024) # Increased shots for potentially better statistics
        result = job.result()
        # Get the measurement counts from the simulation result.
        # Use the transpiled circuit object to get counts, as its gates might be different.
        counts = result.get_counts(qc_transpiled)
        # print("Simulation complete for one run.") # Muted
        # Return the counts and the number of counting qubits 't'.
        return counts, t
    except Exception as e:
        print(f"Error during quantum simulation execution for N={N}, a={a}: {e}")
        # Return None if simulation execution fails
        return None, None # Indicate simulation execution failure


# --- Classical Post-processing ---

def shor_classical_postproc(counts, N, a, t):
    """
    Processes measurement results from a SINGLE quantum run
    to find period candidates and potential factors.
    Collects all (a, r_candidate) pairs evaluated from this run's outcomes
    AND a list of (a, validated_r) pairs where r was validated.

    Args:
        counts (dict): Measurement counts from the quantum circuit run.
        N (int): The number being factored.
        a (int): The base used in modular exponentiation for this run.
        t (int): Number of counting qubits.

    Returns:
        tuple: (factors, status, ar_pairs_evaluated_this_run, validated_ar_pairs_this_run)
               factors is a tuple (p, q) if non-trivial factors found, otherwise None.
               status is a string indicating the outcome ("FactorsFound", "OddPeriodFound", etc.) for this run.
               ar_pairs_evaluated_this_run is a list of (a, r_candidate) tuples for all
               r_candidates evaluated from measurement outcomes in this run.
               validated_ar_pairs_this_run is a list of (a, validated_r) tuples where
               validated_r is a period found and validated (a^r == 1 mod N).
    """
    # List to store all (a, r_candidate) pairs evaluated in this post-processing run
    ar_pairs_evaluated_this_run = []
    # List to store only validated (a, r) pairs
    validated_ar_pairs_this_run = []

    # If counts is None, it means the quantum simulation failed.
    if counts is None:
        # Return default values including empty lists if counts is None
        return None, "QuantumFailure", ar_pairs_evaluated_this_run, validated_ar_pairs_this_run

    # List to store processed measurement outcomes (phase, decimal, count)
    measured_values = []

    # Process each measured outcome (binary string) and its count
    for output_bin, count in counts.items():
        # Convert the binary string outcome to an integer (decimal)
        # The binary string from Qiskit measurement is MSB first by default.
        decimal_value = int(output_bin, 2)
        # Calculate the phase estimate: k / 2^t
        # The measured value 'k' corresponds to the decimal_value.
        phase = decimal_value / (2**t)
        # Store the processed outcome
        measured_values.append({'phase': phase, 'decimal': decimal_value, 'count': count})

    # Sort the outcomes by frequency (count) in descending order
    # This allows us to process the most likely results first.
    measured_values.sort(key=lambda x: x['count'], reverse=True)

    # print(f"Processing {len(measured_values)} distinct measurement outcomes for a={a}...") # Muted

    # Iterate through the sorted measured values to find period candidates
    found_valid_period_this_run = False # Flag to track if ANY valid period was found in this run's outcomes
    for item in measured_values:
        phase = item['phase']
        decimal = item['decimal']
        count = item['count']

        # Use continued fractions to find a candidate period r from the phase estimate (k/2^t approx m/r)
        # Fraction(phase).limit_denominator(limit) finds the best rational approximation k'/r'
        # where r' <= limit. A typical limit is N-1.
        if phase == 0:
            # A phase of 0 usually corresponds to k=0. This implies m=0.
            # k/2^t = 0/r. Continued fractions might give 0/1. r=1 is a trivial period.
            r_candidate = 1 # Trivial period candidate
            # Store the (a, r_candidate) pair derived from this outcome
            ar_pairs_evaluated_this_run.append((a, r_candidate))
            # print(f"  Outcome {decimal} (phase {phase:.6f}, count {count}): r_candidate={r_candidate} (trivial).") # Print for visibility
            continue # Skip validation for the trivial period r=1

        try:
            # Limit the denominator search up to N-1. The true period r must be less than N.
            # The continued fraction expansion of phase will give approximations m/r_candidate.
            frac = Fraction(phase).limit_denominator(N - 1)
            r_candidate = frac.denominator
            # k_candidate = frac.numerator # The numerator m is also found but not directly used for factoring here

        except Exception as e:
             # Handle potential errors during continued fraction computation
             # print(f"  Outcome {decimal} (phase {phase:.6f}, count {count}): Error processing with continued fractions: {e}") # Print error
             # Store the (a, r_candidate) pair using a placeholder (-1) to indicate CF failure for this outcome
             ar_pairs_evaluated_this_run.append((a, -1)) # Use -1 as indicator for CF failure or invalid result
             continue # Skip to processing the next measurement outcome

        # Store the (a, r_candidate) pair derived from this measurement outcome
        ar_pairs_evaluated_this_run.append((a, r_candidate))

        # Check if the candidate period r_candidate is valid for factoring.
        # It must be a positive integer less than N. r_candidate >= N implies the CF failed to find r < N.
        # r_candidate == 0 should not happen with limit_denominator >= 1, but safety check.
        if r_candidate <= 0 or r_candidate >= N:
            # print(f"  Outcome {decimal} (phase {phase:.6f}, count {count}): Candidate period r={r_candidate} invalid (out of range).") # Muted
            continue # This candidate is invalid, try the next measurement outcome

        # Validate the candidate period: Check if a^r_candidate = 1 (mod N)
        # This is the crucial check from Shor's algorithm post-processing.
        # The true order r must satisfy this. Any multiple of the order also satisfies this.
        if power(a, r_candidate, N) == 1:
            # print(f"  Outcome {decimal} (phase {phase:.6f}, count {count}): Candidate period r={r_candidate} VALIDATED (a^r == 1 mod N).") # Print validation success
            r = r_candidate # This r_candidate is a valid period
            found_valid_period_this_run = True # Set the flag: at least one valid period found in this run

            # Add this validated period to the list of validated pairs for this run
            validated_ar_pairs_this_run.append((a, r))

            # --- Factor finding logic ---
            # Check the conditions for finding non-trivial factors from a period r:
            # 1. r must be even.
            # 2. a^(r/2) mod N must not be N-1 (which is equivalent to a^(r/2) == -1 mod N).

            # Check if the validated period r is even
            if r % 2!= 0:
                # print(f"  Validated period r={r} is odd. Cannot find factors this way.") # Muted
                # An odd period is valid but doesn't directly give factors using this step.
                continue # Try the next measurement outcome

            # Calculate a^(r/2) mod N
            x = power(a, r // 2, N)

            # Check the non-triviality condition: a^(r/2) mod N != N-1
            if x == N - 1:
                # print(f"  Validated period r={r} leads to trivial factors (a^(r/2) == -1 mod N).") # Muted
                # This condition yields gcd(a^(r/2)+1, N) = N and gcd(a^(r/2)-1, N) = 1, or vice-versa.
                continue # Try the next measurement outcome

            # If both conditions are met (r is even and x != N-1), calculate potential factors:
            # Factor 1: gcd(a^(r/2) + 1, N)
            # Factor 2: gcd(a^(r/2) - 1, N)
            p = gcd(x + 1, N)
            q = gcd(x - 1, N)

            # print(f"  Potential factors from validated period r={r}, base a={a}: p={p}, q={q}") # Muted

            # Check if either calculated factor is non-trivial (i.e., not 1 or N)
            # If p is a non-trivial factor, N/p is the other. If q is, N/q is the other.
            if (p!= 1 and p!= N) or (q!= 1 and q!= N):
                 # Non-trivial factors found!
                 # Determine which factor is non-trivial and return (factor, N/factor).
                 # If p is non-trivial, use p. Otherwise, q must be non-trivial (given the conditions).
                 factors_found_this_run = (p, N // p) if (p!= 1 and p!= N) else (q, N // q)
                 print(f"Non-trivial factors found in post-processing (a={a}, r={r}): {factors_found_this_run}")
                 # We found factors from this measurement outcome in this run.
                 # Return the factors, status, all collected pairs, and validated pairs found SO FAR in this run.
                 # We can stop processing further outcomes for this 'a' as we have factored N.
                 return factors_found_this_run, "FactorsFound", ar_pairs_evaluated_this_run, validated_ar_pairs_this_run
            else:
                 # The period r was valid and even, but led to trivial factors.
                 # This can happen if gcd(a^(r/2)+1, N) or gcd(a^(r/2)-1, N) results in 1 or N.
                 # This is not a failure, just means this specific period didn't yield factors.
                 # print(f"  Validated period r={r} led to trivial factors (p or q is 1 or N).") # Muted
                 continue # Continue processing other measurements in this run
            # --- End Factor finding logic ---

        # else: # power(a, r_candidate, N) != 1
             # The r_candidate from continued fractions did NOT pass the validation check.
             # It is not a period (or not a multiple of the order).
             # print(f"  Candidate period r={r_candidate} failed validation.") # Muted
             # Continue processing other measurement outcomes in this run.

    # If the loop finishes iterating through all measured outcomes for this 'a'
    # without finding non-trivial factors:
    if found_valid_period_this_run:
         # Valid period(s) were found (and validated), but they were either odd or led to trivial factors.
         print(f"Finished processing outcomes for a={a}. Found valid period(s), but they led to trivial factors or were odd.")
         # Return None factors, status, all collected pairs, and validated pairs for this 'a' attempt.
         return None, "TrivialFactorsOrOddPeriod", ar_pairs_evaluated_this_run, validated_ar_pairs_this_run
    else:
         # No period candidate from any measurement outcome passed the validation check.
         print(f"Finished processing outcomes for a={a}. No valid period found from any measurement outcome.")
         # Return None factors, status, all collected pairs, and an empty validated pairs list for this 'a' attempt.
         return None, "NoValidPeriodFound", ar_pairs_evaluated_this_run, validated_ar_pairs_this_run


# --- Main Orchestration Function (Simplified) ---

def factor_shor_precheck_only(N):
    """
    Performs initial classical checks for Shor's algorithm.
    The quantum/post-processing loop is handled externally.

    Args:
        N (int): The number to factor.

    Returns:
        tuple: (factors, status)
               factors is a tuple (p, q) if found classically, otherwise None.
               status is a string indicating the outcome ("EvenFactorFound", "Prime", etc. or "Proceed").
    """
    return shor_classical_preproc(N)


# --- Example Usage (Modified for User Input and Multiple Runs) ---
if __name__ == "__main__":
    # --- Get User Input ---
    while True:
        try:
            n_input_str = input("Enter the number N to factor (integer > 1): ")
            N_to_factor = int(n_input_str)
            if N_to_factor > 1:
                break
            else:
                print("N must be greater than 1. Please try again.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

    while True:
        a_input_str = input(f"Enter a specific base 'a' (an integer between 2 and {N_to_factor -1}), or press Enter for random 'a' each run: ")
        if a_input_str.strip() == "":
            specific_a_value = None
            break # User wants random 'a' for each run
        try:
            a_val = int(a_input_str)
            # Basic validation before proceeding
            if a_val >= 2 and a_val < N_to_factor:
                 # Check if the specific 'a' is coprime. If not, inform the user and let them choose again.
                 g = gcd(a_val, N_to_factor)
                 if g != 1:
                      print(f"The chosen base a={a_val} shares a factor with N={N_to_factor}. gcd({a_val}, {N_to_factor}) = {g}. You found a factor classically! ({g}, {N_to_factor // g})")
                      # If factor found here, exit cleanly
                      print("Exiting as factor was found classically during 'a' selection.")
                      exit() # Exit the script
                 else:
                      specific_a_value = a_val
                      break
            else:
                print(f"Invalid 'a' value. Please enter an integer between 2 and {N_to_factor - 1}.")
        except ValueError:
            print("Invalid input. Please enter an integer or press Enter.")

    while True:
         try:
             runs_input_str = input("How many quantum runs/attempts should be performed? (Enter an integer >= 1): ")
             num_total_runs = int(runs_input_str)
             if num_total_runs >= 1:
                  break
             else:
                  print("Number of runs must be at least 1. Please try again.")
         except ValueError:
              print("Invalid input. Please enter an integer.")


    # --- No need for the 'plot_all_r_candidates' flag anymore based on the new requirement ---
    # plot_all_r_candidates = True # Default
    # while True:
    #     plot_choice = input("Plot ALL candidate r values (including non-validated/failures) in histogram? (yes/no): ").lower().strip()
    #     if plot_choice in ['yes', 'y']:
    #         plot_all_r_candidates = True
    #         break
    #     elif plot_choice in ['no', 'n']:
    #         plot_all_r_candidates = False
    #         break
    #     else:
    #         print("Invalid input. Please enter 'yes' or 'no'.")
    # --- End Removed Input ---


    print("-" * 30)

    start = time.time()

    # Perform initial classical pre-checks
    factors_classical, status_classical = factor_shor_precheck_only(N_to_factor)

    # If classical checks found factors or N is not factorable by Shor (e.g., prime)
    if status_classical!= "Proceed":
        print(f"Factoring halted by classical pre-processing: {status_classical}")
        # Return classical factors if found, and an empty data array.
        end = time.time()
        print(f"Total execution time: {end - start:.8f} seconds.")
        # Plotting data will be empty in this case, handled later.
        collected_ar_pairs = np.array([]) # Ensure empty array is defined for all collected
        collected_validated_ar_pairs = np.array([]) # Ensure empty array for validated r
        final_factors = factors_classical

    else: # Classical pre-processing allows proceeding to quantum
        all_ar_pairs_list = [] # Initialize list to collect ALL (a, r_candidate) pairs across all runs
        all_validated_ar_pairs_list = [] # Initialize list to collect ONLY validated (a, r) pairs
        found_factors_quantum = None # Variable to store factors if found in any quantum run

        print(f"Proceeding to quantum period finding for N={N_to_factor} with {num_total_runs} run(s).")

        # --- Main Loop for Quantum Runs ---
        for run_count in range(1, num_total_runs + 1):
            # If factors were already found, stop running further quantum simulations.
            # This check is important if we are doing random 'a' runs and one yields factors early.
            # If specific_a_value is not None, we run all requested runs for that 'a',
            # even if factors are found quantumly, to collect data for that 'a'.
            # If specific_a_value is None (random 'a'), and factors are found (either classically in 'a' pick
            # or quantumly), we stop remaining quantum runs as N is factored.
            if found_factors_quantum is not None and specific_a_value is None:
                 print(f"\nFactors already found in a previous run. Stopping remaining runs.")
                 break # Exit the 'for run_count' loop


            print(f"\n--- Run {run_count}/{num_total_runs} ---")

            # Determine the base 'a' for this run
            a_for_run = None # Initialize for this run
            current_run_found_classical_factor = None # Flag for this specific run's a selection

            if specific_a_value is not None:
                a_for_run = specific_a_value
                # We already validated coprime in the input loop for specific_a
                print(f"Using specified base a={a_for_run} for this run.")

            else:
                # Choose a random base 'a' coprime to N for this run
                 while True:
                     a_for_run = random.randint(2, N_to_factor - 1)
                     g = gcd(a_for_run, N_to_factor)
                     if g == 1:
                         print(f"Using random base a={a_for_run} (coprime to N={N_to_factor}) for this run.")
                         break # Found a suitable 'a' for this random run
                     else:
                         # Found factor classically during 'a' selection in THIS run!
                         print(f"Factor found classically during 'a' selection for run {run_count}: gcd({a_for_run}, {N_to_factor}) = {g}")
                         current_run_found_classical_factor = (g, N_to_factor // g) # Store factors for THIS run
                         # Stop all further runs as factors are found.
                         # The break condition at the top of the loop will catch this in the next iteration.
                         found_factors_quantum = current_run_found_classical_factor # Store this as the found factors
                         print(f"Factors found. Stopping remaining runs.")
                         break # Exit the 'while True' for 'a' selection for this run


            # --- Decide whether to run quantum for THIS run ---
            # Run quantum if 'a' was successfully selected (either specific or random coprime)
            # AND factors weren't found *just now* classically for this run's 'a'.
            if a_for_run is not None and (current_run_found_classical_factor is None):

                 # Run the quantum period finding circuit for this 'a'
                 counts, t = quantum_period_finder(N_to_factor, a_for_run)

                 # Post-process the results from this single quantum run
                 # shor_classical_postproc returns 4 values
                 if counts is not None:
                     factors_this_run, status_this_run, ar_pairs_in_this_run, validated_ar_pairs_in_this_run = shor_classical_postproc(counts, N_to_factor, a_for_run, t)

                     # Add the collected (a, r_candidate) pairs from this run to the main list (for potential use)
                     all_ar_pairs_list.extend(ar_pairs_in_this_run)
                     # Add the validated (a, r) pairs from this run to the main list (for finding smallest period)
                     all_validated_ar_pairs_list.extend(validated_ar_pairs_in_this_run)

                     print(f"Collected {len(ar_pairs_in_this_run)} total candidate points and {len(validated_ar_pairs_in_this_run)} validated period(s) from post-processing this run.") # Print counts

                     # Check if factors were found in this run's post-processing
                     if factors_this_run:
                          found_factors_quantum = factors_this_run # Store the factors found (will be the last set found)
                          # If specific_a_value is None, we break the loop in the next iteration.
                          # If specific_a_value is set, we continue running to collect more data for plotting.
                          if specific_a_value is not None:
                                print(f"Factors found in run {run_count}. Data collected for a={a_for_run}. Continuing remaining runs as requested.")
                          # else (random 'a'): the loop break condition will handle stopping

                 else: # counts is None, quantum simulation failed for this run
                     print(f"Quantum simulation failed for a={a_for_run} in run {run_count}. No data collected for this run.")

            # If factors were found classically for the 'a' selected in this run
            elif current_run_found_classical_factor is not None:
                 # found_factors_quantum is already updated in the inner while loop.
                 # The outer loop break condition will handle stopping if specific_a_value is None.
                 pass # Do nothing else in this run


        # --- End of Main Loop ---
        end = time.time()
        # Report how many runs actually completed their loop iteration
        print(f"\nFinished {run_count} loop iteration(s).")
        # Count unique 'a' values for which quantum simulation was attempted and produced data
        unique_a_values_with_data = np.unique(np.array(all_ar_pairs_list)[:, 0]) if all_ar_pairs_list else []
        actual_quantum_runs_completed_count = len(unique_a_values_with_data)

        print(f"Completed quantum simulation and post-processing for {actual_quantum_runs_completed_count} unique base(s) 'a'.")
        print(f"Total execution time: {end - start:.8f} seconds.")

        # The final factors are the ones found (either classically or quantumly)
        final_factors = found_factors_quantum


        # Convert the collected data lists to NumPy arrays for plotting
        collected_ar_pairs = np.array(all_ar_pairs_list) if all_ar_pairs_list else np.array([])
        collected_validated_ar_pairs = np.array(all_validated_ar_pairs_list) if all_validated_ar_pairs_list else np.array([])


    # --- Report Final Result ---
    print("\n--- Final Result ---")
    if final_factors:
        print(f"Successfully factored {N_to_factor} into: {final_factors}")
    else:
        print(f"Did not find non-trivial factors for {N_to_factor} after the attempted runs.")


    # --- Generate the specific histogram requested ---
    # 1. Find the smallest validated period
    smallest_validated_period = None
    if collected_validated_ar_pairs.shape[0] > 0:
         positive_validated_periods = collected_validated_ar_pairs[collected_validated_ar_pairs[:, 1] > 0, 1]
         if positive_validated_periods.shape[0] > 0:
             smallest_validated_period = int(positive_validated_periods.min())
             print(f"\nSmallest positive validated period found across runs: {smallest_validated_period}")
         else:
             print("\nNo positive validated periods were collected to determine the smallest.")


    # 2. If smallest validated period found, get its divisors and plot the specific histogram
    if smallest_validated_period is not None:
         divisors_of_smallest_period = get_divisors(smallest_validated_period)
         # The target values for the histogram are the smallest period and its divisors
         target_r_values = sorted(list(set([smallest_validated_period] + divisors_of_smallest_period))) # Ensure unique and sorted

         print(f"Target values for histogram (Smallest Validated Period and its divisors): {target_r_values}")

         # Filter the ALL candidate periods (collected_ar_pairs) to only include the target values
         # We use the second column (r_candidate) and check if it's in the target_r_values list
         filtered_r_candidates = collected_ar_pairs[np.isin(collected_ar_pairs[:, 1], target_r_values), 1]

         if filtered_r_candidates.shape[0] > 0:
              plt.figure(figsize=(10, 6))

              # Determine bins centered around the target values
              # We need a bin for each target value. Bins should span from value - 0.5 to value + 0.5
              # We can create discrete bins manually for exactly the target values.
              # Example: for [2, 3, 6], bins could be [1.5, 2.5, 3.5, 6.5]
              # The arange below creates edges [1.5, 2.5, 3.5, 4.5, 5.5, 6.5] if target_r_values was [2,3,6]
              # The bins will be centered on integers.
              # Add 0.5 margin on min/max for bins
              min_target = min(target_r_values)
              max_target = max(target_r_values)
              # Create bin edges that center the bars correctly on the integer values
              # Edges: min_target - 0.5, min_target + 0.5, ..., max_target + 0.5
              bins = np.arange(min_target - 0.5, max_target + 1.5, 1)


              plt.hist(filtered_r_candidates, bins=bins, alpha=0.8, edgecolor='black', align='mid', rwidth=0.8)
              plt.xlabel("Period 'r' (Smallest Validated Period and its Divisors)")
              plt.ylabel("Frequency (Count of Candidate Period Occurrences)")
              title_a_source = f"({actual_quantum_runs_completed_count} runs with random 'a')" if specific_a_value is None else f"({num_total_runs} runs with a={specific_a_value})"
              plt.title(f"Frequency of Target Period Values as Candidates for N={N_to_factor} {title_a_source}")

              # Set x-axis ticks to show the target values
              plt.xticks(target_r_values)

              plt.grid(axis='y', alpha=0.75)
              plt.tight_layout()
              plt.show()
         else:
              print("\nNo candidate periods matched the target values (Smallest Validated Period and its divisors). No specific histogram generated.")

    else:
        print("\nCannot generate the specific histogram because no positive validated period was found.")


    # --- REMOVED: Previous scatter plot and general histogram code ---
    # if collected_ar_pairs.shape[0] > 0:
    #    ... (scatter plot code removed) ...
    #
    #    ... (general histogram code removed) ...
    # --- End REMOVED ---
