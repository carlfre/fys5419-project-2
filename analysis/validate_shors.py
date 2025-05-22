import numpy as np
from shors import shors
from time import time
from utils import latex_table


def is_nontrivial_factor(x: int, N: int) -> bool:
    return (N%x == 0) and (x != 1) and (x != N)

def run_shors_15():
    shors(15, n_shots_phase_estimation=1024)

def run_shors_21():
    shors(21, n_shots_phase_estimation=1024)


def shors_timing_test(N: int = 15, n_shots: int = 1024, n_timing_runs=10, use_qiskit: bool = False):
    times = []
    successes = []
    for i in range(n_timing_runs):
        start = time()
        factor = shors(N, n_shots_phase_estimation=n_shots, use_qiskit=use_qiskit)
        end = time()
        times.append(end - start)
        successes.append(is_nontrivial_factor(factor, N))

    print("Average time taken:", np.mean(times))
    print("Success rate:", np.mean(successes))
    return times, successes


def run_timing_tests_1():
    N = 15
    n_timing_runs = 20
    times_owncode_shots_1, s1 = shors_timing_test(N, 1, n_timing_runs, False)
    times_qiskit_shots_1, s2 = shors_timing_test(N, 1, n_timing_runs, True)
    times_owncode_shots_20, s3 = shors_timing_test(N, 20, n_timing_runs, False)
    times_qiskit_shots_20, s4 = shors_timing_test(N, 20, n_timing_runs, True)


    time_owncode_shots_1 = np.mean(times_owncode_shots_1)
    time_qiskit_shots_1 = np.mean(times_qiskit_shots_1)
    time_owncode_shots_20 = np.mean(times_owncode_shots_20)
    time_qiskit_shots_20 = np.mean(times_qiskit_shots_20)

    data = [[time_owncode_shots_1, time_qiskit_shots_1],
            [time_owncode_shots_20, time_qiskit_shots_20]]
    table = latex_table(data, above_headers=["Our code", "Qiskit"], left_headers=["1 shot", "20 shots"])

    print(np.mean(s1), np.mean(s2), np.mean(s3), np.mean(s4))
    print(table)


def run_timing_tests_2():
    N = 21
    n_timing_runs = 5
    times_owncode, _ = shors_timing_test(N, 1, n_timing_runs, False)
    times_qiskit, _ = shors_timing_test(N, 1, n_timing_runs, True)

    data = list(zip(times_owncode, times_qiskit))
    table = latex_table(data, above_headers = ["Our code", "Qiskit"])
    print(table)




if __name__ == "__main__":
    # run_timing_tests_1()
    run_timing_tests_2()
