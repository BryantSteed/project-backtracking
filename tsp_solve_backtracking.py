import math
import random
import numpy as np

from utils import Tour, SolutionStats, Timer, score_tour, Solver
from cuttree import CutTree


def random_tour(edges: list[list[float]], timer: Timer) -> list[SolutionStats]:
    stats = []
    n_nodes_expanded = 0
    n_nodes_pruned = 0
    cut_tree = CutTree(len(edges))

    while True:
        if timer.time_out():
            return stats

        tour = random.sample(list(range(len(edges))), len(edges))
        n_nodes_expanded += 1

        cost = score_tour(tour, edges)
        if math.isinf(cost):
            n_nodes_pruned += 1
            cut_tree.cut(tour)
            continue

        if stats and cost > stats[-1].score:
            n_nodes_pruned += 1
            cut_tree.cut(tour)
            continue

        stats.append(SolutionStats(
            tour=tour,
            score=cost,
            time=timer.time(),
            max_queue_size=1,
            n_nodes_expanded=n_nodes_expanded,
            n_nodes_pruned=n_nodes_pruned,
            n_leaves_covered=cut_tree.n_leaves_cut(),
            fraction_leaves_covered=cut_tree.fraction_leaves_covered()
        ))

    if not stats:
        return [SolutionStats(
            [],
            math.inf,
            timer.time(),
            1,
            n_nodes_expanded,
            n_nodes_pruned,
            cut_tree.n_leaves_cut(),
            cut_tree.fraction_leaves_covered()
        )]


def greedy_tour(edges: list[list[float]], timer: Timer) -> list[SolutionStats]:
    num_nodes = len(edges[0])
    stats = []
    global_best_score = math.inf
    for start_node in range(num_nodes):
        path = [start_node]
        visited = set([start_node])
        current_node = start_node
        if timer.time_out():
            return stats
        process_start_node_greedy(edges, num_nodes, path, visited, current_node)
        global_best_score = add_stat(edges, timer, stats, global_best_score, path)
    return stats

def process_start_node_greedy(edges, num_nodes, path, visited, current_node):
    while len(visited) < num_nodes:
        valid_nodes = set()
        for node in range(num_nodes):
            if node not in visited and not math.isinf(edges[current_node][node]):
                valid_nodes.add(node)
        if not valid_nodes:
            break
        curr_min = math.inf
        for node in valid_nodes:
            if edges[current_node][node] < curr_min:
                curr_min = edges[current_node][node]
                best_node = node
        path.append(best_node)
        visited.add(best_node)
        current_node = best_node

def add_stat(edges, timer, stats, global_best_score, path):
    if len(path) != len(edges):
        return global_best_score
    cost = score_tour(path, edges)
    if cost < global_best_score:
        global_best_score = cost
        stat: SolutionStats = SolutionStats(tour=path,
                                               score=cost,
                                               time=timer.time(),
                                               max_queue_size=0,
                                               n_nodes_expanded=0,
                                               n_nodes_pruned=0,
                                               n_leaves_covered=0,
                                               fraction_leaves_covered=0.0)
        stats.append(stat)
    return global_best_score


def backtracking(edges: list[list[float]], timer: Timer) -> list[SolutionStats]:
    return []

def backtracking_bssf(edges: list[list[float]], timer: Timer) -> list[SolutionStats]:
    return []

