from number_theory import gcd, find_order_classical, simplify

import matplotlib.pyplot as plt
import numpy as np

from utils import read_csv

def freq_to_probabilities(frequencies: dict[int, int]) -> dict[int, float]:
    """Converts frequencies to probabilities."""
    total = sum(frequencies.values())
    return {k: v / total for k, v in frequencies.items()}


def expected_estimated_orders(r: int, returns_probabilities: bool = False) -> dict[int, int] | dict[int, float]:
    """Given an order r, returns the number of value of s for which we estimate the order to be r_prime.
    
    Ie. we might miss the order r, and find an estimate r_prime which is a factor of r.
    
    Args:
        r (int): Order of the system.
        returns_probabilities (bool): If True, returns probabilities instead of frequencies."""

    estimated_orders = {}

    for s in range(0, r):
        _, r_prime = simplify(s, r)

        if r_prime not in estimated_orders:
            estimated_orders[r_prime] = 0

        estimated_orders[r_prime] += 1

    if returns_probabilities:
        return freq_to_probabilities(estimated_orders)
    else:
        return estimated_orders


def plot_qm_order_estimates(a: int, N: int, n_shots: int) -> None:
    """Plots the empirical distribution of the estimated order r', versus the theoretical distribution.
    
    Args:
        a (int): number to compute order of.
        N (int): number to factorize.
        n_shots (int): number of samples taken in order finding algorithm.

    Returns:
        None
    """
    r = find_order_classical(a, N)
    true_order_to_freq = expected_estimated_orders(r, returns_probabilities=True)
    true_order_values = np.array(list(true_order_to_freq.keys()))
    true_order_frequencies = np.array(list(true_order_to_freq.values()))


    estimated_distribution = read_csv(f"results/estimated_order_distribution_a={a}_N={N}_shots={n_shots}.csv")  
    estimated_distribution = freq_to_probabilities(estimated_distribution)
    estimated_distribution_keys = np.array(list(estimated_distribution.keys()))
    estimated_distribution_values = estimated_distribution.values()

    width = 0.4

    plt.figure(figsize=(5, 15/4))

    plt.bar(true_order_values - width/2, true_order_frequencies, color='blue', width=0.4, label='Theoretical distribution')
    plt.bar(estimated_distribution_keys + width/2, estimated_distribution_values, color='red', width=0.4, label=f'Empirical distribution')

    plt.xticks(list(range(min(true_order_values), max(true_order_values) + 1)))
    plt.xlabel('r\'')
    plt.ylabel('Relative frequency')
    plt.legend()
    plt.savefig(f'plots/est_order_dist_a={a}_N={N}_shots={n_shots}.pdf')
    plt.close()
    print(f"Plot saved to plots/est_order_dist_a={a}_N={N}_shots={n_shots}.pdf")


def verify_orders(N: int, n_shots: int = 10_000) -> None:
    """Verify that the estimated orders r' match the orders we expect to see, for all a coprime to N.
     
    Specifically, if the order is r, we should see all factors of r, with a certain frequency,
    which is given by expected_estimated_orders(r).

    We check that the empirical distribution matches the expected distribution up to 5% relative error.

    Args:
        N (int): Number to factorize.
    """
    tolerance = 0.05 # Verify agreement up to 5% relative error

    for a in range(2, N):
        if gcd(a, N) != 1:
            continue
        
        r = find_order_classical(a, N)
        print(f"Testing a={a}, N={N}, r={r}")
        estimated_distribution = read_csv(f"results/estimated_order_distribution_a={a}_N={N}_shots={n_shots}.csv")
        estimated_distribution = freq_to_probabilities(estimated_distribution)
        true_order_to_freq = expected_estimated_orders(r, returns_probabilities=True)

        if set(estimated_distribution.keys()) != set(true_order_to_freq.keys()):
            print(f"Error: Estimated orders do not match true orders for a={a}, N={N}.")
            continue

        for r_prime in estimated_distribution.keys():
            est_freq = estimated_distribution[r_prime]
            true_freq = true_order_to_freq[r_prime]

            if abs(est_freq - true_freq) / true_freq > tolerance:
                print(f"Error: Frequencies do not match for a={a}, N={N}, r_prime={r_prime}.")
                break
            else:
                print(f"Success: Frequencies match for a={a}, N={N}, r_prime={r_prime}.")





if __name__ == "__main__":
    N = 15
    a = 8
    n_shots = 10_000
    plot_qm_order_estimates(a, N, n_shots)
    verify_orders(15, n_shots=n_shots)
