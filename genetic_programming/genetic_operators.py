'''
File: genetic_operators.py
The purpose of this file is to define selection, mutation, and crossover
'''
from settings import TOURNAMENT_SIZE
from settings import MUTATION_PROB
from constraints import feature_types
from deap import gp
from fitness import dummy_fitness
from genetic_programming.tree_validation import validate_tree, get_type
import random
from genetic_programming.primitives import primitive_set

def tournament_selection(population, TOURNAMENT_SIZE):
    selection_completed_pop = []

    for i in range(len(population)):
        contenders = random.sample(population, TOURNAMENT_SIZE)
        contender_fitnesses = [(tree, dummy_fitness(tree)) for tree in contenders]
        #check tuples (tree, fitness) by fitness [1] and gets max, then max tuple at index [0] is passed as best_tree
        best_tree = max(contender_fitnesses, key=lambda x: x[1])[0] 

        # validity check during selection
        if validate_tree(best_tree, feature_types):
            selection_completed_pop.append(best_tree)

    return selection_completed_pop

def crossover(selection_completed_pop, N_size):
    crossover_completed_pop = []

    while len(crossover_completed_pop) < N_size:
        parent_1 = random.choice(selection_completed_pop)
        parent_2 = random.choice(selection_completed_pop)
        while parent_1 is parent_2:
            parent_2 = random.choice(selection_completed_pop)

        parent_1_copy = parent_1.copy()
        parent_2_copy = parent_2.copy()

        index_1 = random.randrange(len(parent_1))
        index_2 = random.randrange(len(parent_2))
        slice_1 = parent_1.searchSubtree(index_1)
        slice_2 = parent_2.searchSubtree(index_2)

        child_1 = parent_1_copy[:slice_1.start] + parent_2_copy[slice_2] + parent_1_copy[slice_1.stop:]
        child_2 = parent_2_copy[:slice_2.start] + parent_1_copy[slice_1] + parent_2_copy[slice_2.stop:]

        # validity check after crossover
        candidate_1 = gp.PrimitiveTree(child_1)
        candidate_2 = gp.PrimitiveTree(child_2)

        if validate_tree(candidate_1, feature_types):
            crossover_completed_pop.append(candidate_1)
        if validate_tree(candidate_2, feature_types):
            crossover_completed_pop.append(candidate_2)

    return crossover_completed_pop[:N_size]  # truncate if you overshoot

def mutation(crossover_completed_pop):
    mutated_pop = []

    for child in crossover_completed_pop:
        mutated_child, = gp.mutUniform(child, expr=gp.genFull, pset=primitive_set)

        # validity check after mutation
        if validate_tree(mutated_child, feature_types):
            mutated_pop.append(mutated_child)

    return mutated_pop