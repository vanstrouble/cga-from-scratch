import numpy as np
import matplotlib.pyplot as plt

from itertools import count
from multiprocessing import Pool, cpu_count


def fitness_func(chromosome):
    return np.sum(chromosome)


def compete_and_update(probabilities, pop_size):
    # Generate 2 individuals from probabilities
    a = (np.random.random(size=len(probabilities)) < probabilities).astype(np.int32)
    b = (np.random.random(size=len(probabilities)) < probabilities).astype(np.int32)

    # Compete 2 individuals
    winner, loser = (
        (a, b)
        if fitness_func(a) > fitness_func(b)
        else (b, a)
    )

    # Update probability vector
    new_probabilities = probabilities.copy()
    for j in range(len(probabilities)):
        if winner[j] != loser[j]:
            if winner[j] == 1:
                new_probabilities[j] += 1 / pop_size
            else:
                new_probabilities[j] -= 1 / pop_size

    return np.around(new_probabilities, 2)


def plot_history(history, counter):
    x_vals, y_vals = zip(*history)
    plt.plot(x_vals, y_vals, marker='o', linestyle='-', color='b', label='Aptitud')
    plt.title("Evolución hasta converger")
    plt.xlabel("Épocas")
    plt.ylabel("Aptitud")
    plt.grid(True)
    plt.legend(f"Iterations: {counter}")
    plt.show()


class CGA:
    def __init__(self, pop_size=50, chrome_size=4) -> None:
        self.pop_size = pop_size
        self.chrome_size = chrome_size
        self.probabilities = np.full(chrome_size, 0.5, np.float32)
        self.best = 0
        self.iter_num = count()

    def run(self, num_cores=cpu_count(), plotting=True):
        if plotting is True:
            history = []

        sum_probabilities = np.sum(self.probabilities)
        pool = Pool(num_cores)

        with Pool(num_cores) as pool:
            while sum_probabilities != self.chrome_size and sum_probabilities != 0:
                for i in range(self.chrome_size):
                    if self.probabilities[i] > 0 and self.probabilities[i] < 1:
                        new_probs = pool.apply(
                            compete_and_update,
                            (self.probabilities, self.pop_size),
                        )
                        self.probabilities = new_probs
                    else:
                        self.probabilities[i] = 1.0

                    if plotting is True:
                        history.append((next(self.iter_num), np.round(fitness_func(self.probabilities), 2)))

                sum_probabilities = np.sum(self.probabilities)

        if plotting is True:
            plot_history(history, self.iter_num)

        return self.probabilities


if __name__ == "__main__":
    cga = CGA(50, 4)
    best_solution = cga.run()
    print(f"Solución: {best_solution}")

    # values = 10
    # for i in range(1, values+1):
    #     print(f"i: {i}")
    #     cga = CGA(50, 4)
    #     best_solution = cga.run(plotting=True)
    #     print(f"Solution: {best_solution}")
    #     print("-------------------------------------------------")
