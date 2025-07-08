'''
File: visualize_tree.py
The purpose of this file is to use graphviz to visualize the trees produced by the algorithm. 
'''
import graphviz
from deap import gp

# #######################################################
# 1. Save symbolic PrimitiveTree as a PNG visualization #
# #######################################################
def save_best_tree_as_image(individual, out_path):
    """
    Saves the given DEAP individual tree as a PNG image.

    Parameters:
        individual (gp.PrimitiveTree): The tree to visualize.
        out_path (str): Path (without extension) where the image should be saved.
    """

    # 1.1 Generate graph components: nodes, edges, and labels from the GP tree
    nodes, edges, labels = gp.graph(individual)

    # 1.2 Initialize a Graphviz directed graph object
    dot = graphviz.Digraph(format="png")

    # 1.3 Add labeled nodes to the graph
    for node in nodes:
        dot.node(str(node), str(labels[node]))

    # 1.4 Connect nodes with directed edges to represent parent-child relationships
    for edge in edges:
        dot.edge(str(edge[0]), str(edge[1]))

    # 1.5 Render the graph to PNG format and remove intermediate files
    dot.render(out_path, cleanup=True)

    # 1.6 Print confirmation message with saved file path
    print(f"Tree saved to: {out_path}.png")