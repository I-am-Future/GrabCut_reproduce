import numpy as np
import maxflow

g = maxflow.Graph[float]()
nodeids = g.add_grid_nodes((5, 5))
structure = np.array([[0, 0, 0],
                      [0, 0, 1],
                      [1, 1, 1]])


g.add_grid_edges(nodeids, 2, structure=structure, symmetric=True)
print(nodeids)
print(structure)
print(g)