'''
File: visualize_tree.py
The purpose of this file is to use graphviz to visualize the trees produced by the algorithm. 
'''
import graphviz
from deap import gp

def save_best_tree_as_image(individual, out_path):
    """
    Saves the given DEAP individual tree as a PNG image.

    Parameters:
        individual (gp.PrimitiveTree): The tree to visualize.
        out_path (str): Path (without extension) where the image should be saved.
                        Example: 'results/tree_visualizations/best_tree'
    """
    nodes, edges, labels = gp.graph(individual)
    dot = graphviz.Digraph(format="png")
    
    for node in nodes:
        dot.node(str(node), str(labels[node]))
    for edge in edges:
        dot.edge(str(edge[0]), str(edge[1]))
    
    dot.render(out_path, cleanup=True)
    print(f"Tree saved to: {out_path}.png")