import tqdm

from GeneticAlgorithm import GeneticAlgorithm
import numpy as np

import matplotlib.pyplot as plt


def fitness_function(gene_size):
    def function(gene: np.ndarray):
        maximum = gene_size * (gene_size - 1) / 2
        fitness = maximum

        for i in range(gene_size):
            for j in range(i):
                if gene[i] == gene[j] or abs(i - j) == abs(gene[i] - gene[j]):
                    fitness -= 1

        return fitness / maximum * 10

    return function


def main():
    fitness_10_queen = fitness_function(10)
    genetic_algorithm = GeneticAlgorithm(gene_count=1000, mutation_rate=0.1, gene_code_length=10, lower_bound=1,
                                         upper_bound=10, fitness_function=fitness_10_queen)

    scores = []
    max_length = 500
    for _ in tqdm.tqdm(range(500)):
        result, data = genetic_algorithm.step()

        if result:
            print(f'Early stopping:\nbest gene: {data[0]}, score: {data[1]}')
            break

        scores.extend(data)

        if len(scores) > max_length:
            scores = scores[-max_length:]

        plt.clf()
        plt.scatter(range(len(scores)), scores, linewidths=.25)
        plt.pause(0.01)
    else:
        gene, score = genetic_algorithm.get_best_gene()
        print(f'best gene: {gene}, score: {score}')

    plt.show()


if __name__ == '__main__':
    main()
