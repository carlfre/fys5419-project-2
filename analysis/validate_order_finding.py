from time import time

import matplotlib.pyplot as plt
import numpy as np

from number_theory import gcd, find_order_classical, simplify
from shors import find_order_qm
from utils import freq_to_probabilities, read_csv, write_csv, are_distributions_close

plt.rcParams.update({
    'axes.labelsize': 16,
    'legend.fontsize': 14,
    'font.size': 12,  # base font size
})


def compute_estimated_order_distribution(N: int, a: int, n_shots: int, use_qiskit: bool) -> dict[int, int]:
    """
    Compute the distribution of estimated orders for a given number of runs.
    """

    orders = find_order_qm(a, N, n_shots=n_shots, use_qiskit=use_qiskit)

    run_type = "qiskit" if use_qiskit else "owncode"
    filename = f"results/estimated_order_distribution_a={a}_N={N}_shots={n_shots}_{run_type}.csv"
    write_csv(orders, filename)
    print(f"Estimated order distribution saved to {filename}")
    return orders


def expected_estimated_orders(
    r: int, returns_probabilities: bool = False
) -> dict[int, int] | dict[int, float]:
    """Given an order r, returns the number of value of s for which we estimate the order to be r_prime.

    Ie. we might miss the order r, and find an estimate r_prime which is a factor of r.

    Args:
        r (int): Order of the system.
        returns_probabilities (bool): If True, returns probabilities instead of frequencies.
    """

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

    estimated_distribution = read_csv(
        f"results/estimated_order_distribution_a={a}_N={N}_shots={n_shots}_owncode.csv"
    )
    estimated_distribution = freq_to_probabilities(estimated_distribution)
    estimated_distribution_keys = np.array(list(estimated_distribution.keys()))
    estimated_distribution_values = estimated_distribution.values()

    estimated_distribution_qiskit = read_csv(
        f"results/estimated_order_distribution_a={a}_N={N}_shots={n_shots}_qiskit.csv"
    )
    estimated_distribution_qiskit = freq_to_probabilities(estimated_distribution_qiskit)
    estimated_distribution_keys_qiskit = np.array(list(estimated_distribution_qiskit.keys()))
    estimated_distribution_values_qiskit = estimated_distribution_qiskit.values()

    width = 0.2

    # plt.figure(figsize=(5, 15 / 4))

    plt.bar(
        estimated_distribution_keys - width,
        estimated_distribution_values,
        color="Blue",
        width=width,
        label=f"Our code",
    )
    plt.bar(
        estimated_distribution_keys_qiskit,
        estimated_distribution_values_qiskit,
        color="red",
        width=width,
        label="Qiskit",
    )
    plt.bar(
        true_order_values + width,
        true_order_frequencies,
        color="green",
        width=width,
        label="Theoretical distribution",
    )

    plt.xticks(list(range(min(true_order_values), max(true_order_values) + 1)))
    plt.xlabel("r'")
    plt.ylabel("Relative frequency")
    plt.legend()
    plt.savefig(f"plots/est_order_dist_a={a}_N={N}_shots={n_shots}.pdf")
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
    tolerance = 0.05  # Verify agreement up to 5% absolute error (more precisely, l^infty distance)

    for a in range(2, N):
        if gcd(a, N) != 1:
            continue

        r = find_order_classical(a, N)
        print(f"Testing a={a}, N={N}, r={r}")
        estimated_distribution = read_csv(
            f"results/estimated_order_distribution_a={a}_N={N}_shots={n_shots}_owncode.csv"
        )
        estimated_distribution = freq_to_probabilities(estimated_distribution)
        estimate_distribution_qiskit = read_csv(
            f"results/estimated_order_distribution_a={a}_N={N}_shots={n_shots}_qiskit.csv"
        )
        estimated_distribution_qiskit = freq_to_probabilities(estimate_distribution_qiskit)
        true_order_to_freq = expected_estimated_orders(r, returns_probabilities=True)

        if set(estimated_distribution.keys()) != set(true_order_to_freq.keys()):
            print(f"Error: Estimated orders do not match true orders for a={a}, N={N}.")
            continue

        is_close: bool = are_distributions_close(estimated_distribution, true_order_to_freq, tol=tolerance)
        is_close_qiskit: bool = are_distributions_close(estimated_distribution_qiskit, true_order_to_freq, tol=tolerance)
        print("Our code close to theoretical:", is_close)
        print("Qiskit close to theoretical:", is_close_qiskit)



def run_order_computations(use_qiskit: bool):
    n_shots = 1000
    N = 15
    for a in range(2, N):
        if gcd(a, N) != 1:
            continue
        start = time()
        order_dist = compute_estimated_order_distribution(N, a, n_shots, use_qiskit)
        print(order_dist)
        print("time:", time() - start)


if __name__ == "__main__":
    # run_order_computations(use_qiskit=False)
    # run_order_computations(use_qiskit=True)

    N = 15
    a = 8
    n_shots = 1_000
    plot_qm_order_estimates(a, N, n_shots)
    verify_orders(15, n_shots=n_shots)





