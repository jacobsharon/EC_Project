from genetic_programming.primitives import primitive_set
from deap import gp
import matplotlib.pyplot as plt
import networkx as nx
import pygraphviz as pgv
from explain.tree_visualization import visualize_tree

def main():
    expr = gp.genFull(primitive_set, min_=3, max_=8)
    tree = gp.PrimitiveTree(expr)
    visualize_tree(tree)

if __name__ == "__main__":
    main()