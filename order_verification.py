from number_theory import gcd, find_order_classical

import matplotlib.pyplot as plt

from utils import read_csv

def simplify(numerator: int, denominator: int) -> tuple[int, int]:
    """Simplifies a fraction as much as possible."""

    # if numerator == 0:
    #     return 0, 1
    
    # print(numerator, denominator)
    gcd_val = gcd(numerator, denominator)
    return numerator // gcd_val, denominator // gcd_val


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
        _, est_r = simplify(s, r)

        if est_r not in estimated_orders:
            estimated_orders[est_r] = 0

        estimated_orders[est_r] += 1

    if returns_probabilities:
        return freq_to_probabilities(estimated_orders)
    else:
        return estimated_orders


def plot_qm_order_estimates(a: int, N: int, n_shots: int):
    r = find_order_classical(a, N)
    true_order_to_freq = expected_estimated_orders(r, returns_probabilities=True)
    true_order_values = list(true_order_to_freq.keys())
    true_order_frequencies = list(true_order_to_freq.values())


    estimated_distribution = read_csv(f"results/estimated_order_distribution_a={a}_N={N}_shots={n_shots}.csv")  
    estimated_distribution = freq_to_probabilities(estimated_distribution)

    plt.figure(figsize=(5, 15/4))

    plt.bar(true_order_values, true_order_frequencies, alpha=0.2, color='blue', width=0.5, label='True Distribution')
    plt.bar(estimated_distribution.keys(), estimated_distribution.values(), alpha=0.2, color='red', width=0.5, label=f'Empirical Distribution')

    plt.xticks(list(range(min(true_order_values), max(true_order_values) + 1)))
    plt.xlabel('Order')
    plt.ylabel('Frequency')
    # plt.title(f'Distribution of Estimated Orders for a={a}, N={N}')
    plt.legend()
    plt.savefig(f'plots/est_order_dist_a={a}_N={N}_shots={n_shots}.pdf')
    # plt.show()
    plt.close()
    print(f"Plot saved to plots/est_order_dist_a={a}_N={N}_shots={n_shots}.pdf")


def verify_orders(N: int, n_shots: int = 10_000):
    """Verify that the estimated orders match the orders we expect to see, for all a coprime to N.
     
    Specifically, if the true order is r, we should see all factors of r, with a certain frequency,
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
    a = 7
    n_shots = 10_000
    # expected_orders = expected_estimated_orders(, returns_probabilities=True)
    # print("Expected orders:", expected_orders)
    plot_qm_order_estimates(a, N, n_shots)
    verify_orders(15, n_shots=n_shots)
