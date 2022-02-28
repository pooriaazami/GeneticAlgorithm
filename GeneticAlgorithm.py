import random

import numpy as np
from scipy.special import softmax


class GeneticAlgorithm:
    def __init__(self, gene_count,
                 mutation_rate,
                 gene_code_length,
                 lower_bound,
                 upper_bound,
                 fitness_function):

        self.__mutation_rate = mutation_rate
        self.__fitness_function = fitness_function
        self.__gene_count = gene_count
        self.__gene_length = gene_code_length
        self.__upper_bound = upper_bound
        self.__lower_bound = lower_bound

        self.__genes = self.__generate_random_breed()

    def __get_fitness_scores(self):
        fitness_scores = np.zeros(self.__gene_count)

        for index, gene in enumerate(self.__genes):
            fitness_scores[index] = self.__fitness_function(gene)

        probabilities = softmax(fitness_scores)

        return probabilities, fitness_scores

    def __select_parents(self):
        probabilities, fitness_scores = self.__get_fitness_scores()

        indices = np.random.choice(a=self.__gene_count, p=probabilities, size=self.__gene_count)
        parents = self.__genes[indices]

        return parents, fitness_scores

    def __generate_new_breed(self, parents):
        new_breed = []
        for i in range(self.__gene_count // 2):
            first_parent = parents[2 * i]
            second_parent = parents[2 * i + 1]

            lim = random.randint(1, self.__gene_length - 1)
            first_child = np.hstack((first_parent[:lim], second_parent[lim:]))
            second_child = np.hstack((second_parent[:lim], first_parent[lim:]))

            new_breed.append(first_child)
            new_breed.append(second_child)

        new_breed = np.array(new_breed)

        self.__genes = new_breed

    def __mutate(self):
        mask = np.random.random((self.__gene_count, self.__gene_length)) <= self.__mutation_rate
        mutation_values = self.__generate_random_breed()

        self.__genes = self.__genes * (1 - mask) + mutation_values * mask

    def step(self, maximum_score=10):
        parents, fitness_scores = self.__select_parents()
        best_gene_index = np.argmax(fitness_scores)

        if fitness_scores[best_gene_index] == maximum_score:
            return True, (self.__genes[best_gene_index], fitness_scores[best_gene_index])

        self.__generate_new_breed(parents)
        self.__mutate()

        return False, fitness_scores

    def get_best_gene(self):
        probabilities, fitness_scores = self.__get_fitness_scores()
        best_gene_index = np.argmax(fitness_scores)
        best_gene = self.__genes[best_gene_index]

        return best_gene, self.__fitness_function(best_gene)

    def log(self):
        print(self.__genes)

    def __generate_random_breed(self):
        return np.random.randint(low=self.__lower_bound, high=self.__upper_bound + 1,
                                 size=(self.__gene_count, self.__gene_length))
