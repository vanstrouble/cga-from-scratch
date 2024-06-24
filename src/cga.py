import numpy as np
# import matplotlib.pyplot as plt
from itertools import count
from multiprocessing import Pool, cpu_count


def fitness_func(chromosome):
    return np.sum(chromosome)


def compete_and_update(probabilities, pop_size):
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


class CGA:
    def __init__(self, pop_size=50, chrome_size=4) -> None:
        self.pop_size = pop_size
        self.chrome_size = chrome_size
        self.probabilities = np.full(chrome_size, 0.5, np.float32)
        self.best = 0
        self.iter_num = count()

    def run(self, num_cores=cpu_count()):
        # x_vals = []
        # y_vals = []

        sum_probabilities = np.sum(self.probabilities)
        pool = Pool(num_cores)

        while sum_probabilities != self.chrome_size and sum_probabilities != 0:
            for i in range(self.chrome_size):
                if self.probabilities[i] > 0 and self.probabilities[i] < 1:
                    new_probs = pool.apply(compete_and_update, (self.probabilities, self.pop_size))
                    self.probabilities = new_probs
                else:
                    self.probabilities[i] = 1.0

                # # Graph history
                # x_vals.append(next(self.iter_num))
                # y_vals.append(np.round(fitness_func(self.probabilities), 2))

            sum_probabilities = np.sum(self.probabilities)

        pool.close()
        pool.join()

        # # Graph the history
        # plt.plot(x_vals, y_vals)
        # plt.title("Evolución hasta converger")
        # plt.xlabel("Épocas")
        # plt.ylabel("Aptitud")
        # plt.show()
        # print(f"Épocas: {self.iter_num}")

        return self.probabilities


if __name__ == "__main__":
    best_solution = CGA(50, 4).run()
    print(f"Solución: {best_solution}")
