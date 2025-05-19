from number_theory import gcd

import matplotlib.pyplot as plt

def simplify(numerator: int, denominator: int) -> tuple[int, int]:
    """Simplifies a fraction as much as possible."""

    # if numerator == 0:
    #     return 0, 1
    
    # print(numerator, denominator)
    gcd_val = gcd(numerator, denominator)
    return numerator // gcd_val, denominator // gcd_val




def expected_estimated_orders(r: int, returns_probabilities: bool = False) -> dict[int, int] | dict[int, float]:

    estimated_orders = {}

    for s in range(0, r):
        _, est_r = simplify(s, r)

        if est_r not in estimated_orders:
            estimated_orders[est_r] = 0

        estimated_orders[est_r] += 1

    if returns_probabilities:
        return {k: v / r for k, v in estimated_orders.items()}
    return estimated_orders


def plot_qm_order_estimates(a: int, N: int):
    true_order_to_freq = expected_estimated_orders(a, returns_probabilities=True)
    order_values = list(true_order_to_freq.keys())
    order_frequencies = list(true_order_to_freq.values())

    plt.bar(order_values, order_frequencies)
    plt.xlabel('Order')
    plt.ylabel('Frequency')
    plt.title(f'Estimated Order Distribution for a={a}, N={N}')
    plt.savefig(f'plots/est_order_dist_a={a}_N={N}.png')




if __name__ == "__main__":

    r = 8
    expected_orders = expected_estimated_orders(r, returns_probabilities=True)
    print("Expected orders:", expected_orders)
