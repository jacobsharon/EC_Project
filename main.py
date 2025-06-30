from genetic_programming.primitives import primitive_set
from settings import POPULATION
from deap import gp
from genetic_programming.initialize import initialize_pop
import matplotlib.pyplot as plt
import networkx as nx
import pygraphviz as pgv
from explain.tree_visualization import visualize_tree

def main():
    #1. initialize a random population
    gen1 = initialize_pop(POPULATION)
    for i, tree in enumerate(gen1):
        print(f"Tree {i+1}:")
        print(tree)
        print("-" * 40)


if __name__ == "__main__":
    main()