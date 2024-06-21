import numpy as np
import matplotlib.pyplot as plt
from itertools import count
from numba import jit


class CGA:
    def __init__(self, pop_size=50, chrome_size=4) -> None:
        self.pop_size = pop_size
        self.chrome_size = chrome_size
        self.probabilities = np.full(chrome_size, 0.5, np.float16)
        self.best = 0
        self.iter_num = count()

    def fitness_func(self, chromosome: np.ndarray[np.int32]) -> np.int32:
        # return sum(bit for bit in chromosome)
        return np.sum(chromosome)

    def run(self):
        x_vals = []
        y_vals = []

        sum_probabilities = np.sum(self.probabilities)
        while sum_probabilities != self.chrome_size and sum_probabilities != 0:
            for i in range(self.chrome_size):
                if self.probabilities[i] > 0 and self.probabilities[i] < 1:

                    # Generate individuals
                    a = (np.random.random(size=len(self.probabilities)) < self.probabilities).astype(int)
                    b = (np.random.random(size=len(self.probabilities)) < self.probabilities).astype(int)

                    # Compete 2 individuals
                    winner, loser = (
                        (a, b)
                        if self.fitness_func(a) > self.fitness_func(b)
                        else (b, a)
                    )

                    # Update probability vector
                    for j in range(self.chrome_size):
                        if winner[j] != loser[j]:
                            if winner[j] == 1:
                                self.probabilities[j] += 1 / self.pop_size
                            else:
                                self.probabilities[j] -= 1 / self.pop_size
                    self.probabilities = np.around(self.probabilities, 2)
                else:
                    self.probabilities[i] = 1.0

                # Graph history
                x_vals.append(next(self.iter_num))
                y_vals.append(np.round(self.fitness_func(self.probabilities), 2))

            sum_probabilities = np.sum(self.probabilities)

        # Graph the history
        plt.plot(x_vals, y_vals)
        plt.title("Evolución hasta converger")
        plt.xlabel("Épocas")
        plt.ylabel("Aptitud")
        plt.show()
        print(f"Épocas: {self.iter_num}")

        return self.probabilities


if __name__ == "__main__":
    best_solution = CGA(50, 4).run()
    print(f"Solución: {best_solution}")
