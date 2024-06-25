import numpy as np
import seaborn as sns
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

    sns.set_context("notebook", rc={"lines.linewidth": 2.5})
    sns.set_style("dark")
    palette = sns.color_palette("rocket")
    color = palette[2]

    sns.lineplot(x=x_vals, y=y_vals, linestyle='-', label='Fitness', color=color)

    plt.title("Evolution until convergerce")
    plt.xlabel("Epochs")
    plt.ylabel("Fitness")

    plt.legend(title=f"Iterations: {next(counter)}" if hasattr(counter, '__next__') else f"Iterations: {counter}")
    plt.show()


class CGA:
    def __init__(self, pop_size=50, chrome_size=4) -> None:
        self.pop_size = pop_size
        self.chrome_size = chrome_size
        self.probabilities = np.full(chrome_size, 0.5, np.float64)
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

                    next(self.iter_num)

                sum_probabilities = np.sum(self.probabilities)

        if plotting is True:
            plot_history(history, self.iter_num)

        return self.probabilities, self.iter_num


if __name__ == "__main__":
    # Random population and chromosome size
    population = np.random.randint(1, 100)
    chromosome_size = np.random.randint(2, 10)

    cga = CGA(population, chromosome_size)
    best_solution, it = cga.run()
    print(f"\nPopulation: {population}, Chromosome Size: {chromosome_size}")
    print(f"Solution: {best_solution}, Iterations: {it}")

    # Board example population and chromosome size
    # cga = CGA(50, 4)
    # best_solution, _ = cga.run()
    # print(f"Solution: {best_solution}")

    # Range population and chromosome size
    # population = 100
    # chromosome_size = 15

    # for i in range(1, population+1):
    #     print(f"\nPopulation: {i}")
    #     cga = CGA(population, chromosome_size)
    #     best_solution, it = cga.run(plotting=False)
    #     print(f"Solution: {best_solution}, Iterations: {it}")
