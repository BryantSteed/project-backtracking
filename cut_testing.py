from tsp_solve_backtracking import backtracking_bssf
from utils import generate_network, Timer
import numpy as np

location, edges = generate_network(5, 11, 0.0)

solutions = backtracking_bssf(edges, Timer())
final_solution = solutions[-1]
print(np.array(edges))
print(final_solution)
print(solutions[-1])