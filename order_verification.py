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


def plot_qm_order_estimates(a: int, N: int, runs: int):
    r = find_order_classical(a, N)
    true_order_to_freq = expected_estimated_orders(r, returns_probabilities=True)
    order_values = list(true_order_to_freq.keys())
    order_frequencies = list(true_order_to_freq.values())


    estimated_distribution = read_csv(f"results/estimated_order_distribution_a={a}_N={N}_runs={runs}.csv")  
    estimated_distribution = freq_to_probabilities(estimated_distribution)

    plt.bar(order_values, order_frequencies, alpha=0.2, color='black')
    plt.bar(estimated_distribution.keys(), estimated_distribution.values(), alpha=0.5, color='orange')
    plt.xlabel('Order')
    plt.ylabel('Frequency')
    plt.title(f'Estimated Order Distribution for a={a}, N={N}')
    # plt.savefig(f'plots/est_order_dist_a={a}_N={N}_runs={runs}.png')
    plt.show()




if __name__ == "__main__":
    N = 15
    a = 8
    runs = 200
    # expected_orders = expected_estimated_orders(, returns_probabilities=True)
    # print("Expected orders:", expected_orders)
    plot_qm_order_estimates(a, N, runs)
