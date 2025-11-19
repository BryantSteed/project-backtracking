import math
import random
import numpy as np

from utils import Tour, SolutionStats, Timer, score_tour, Solver
from cuttree import CutTree
from typing import Tuple


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
        global_best_score = add_stat_greedy(edges, timer, stats, global_best_score, path)
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

def add_stat_greedy(edges, timer, stats, global_best_score, path):
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
    stats = []
    stack = [[0]]
    global_best_score = math.inf
    while stack and not timer.time_out():
        path = stack.pop()
        child_paths = expand_path(edges, path)
        for child_path in child_paths:
            global_best_score = vet_child_path_backtracking(edges, stats, stack, 
                                                            child_path, global_best_score, timer)
    return stats

def vet_child_path_backtracking(edges, stats, stack, child_path, global_best_score, timer) -> int:
    if len(child_path) == len(edges):
        cost = score_tour(child_path, edges)
        if cost < global_best_score:
            global_best_score = cost
            stat: SolutionStats = SolutionStats(tour=child_path,
                                               score=cost,
                                               time=timer.time(),
                                               max_queue_size=0,
                                               n_nodes_expanded=0,
                                               n_nodes_pruned=0,
                                               n_leaves_covered=0,
                                               fraction_leaves_covered=0.0)
            stats.append(stat)
    else:
        stack.append(child_path)
    return global_best_score

def expand_path(edges: list[list[float]], path: Tour) -> list[Tour]:
    child_paths = []
    for node in range(len(edges)):
        if node not in path and not math.isinf(edges[path[-1]][node]):
            new_path = path + [node]
            child_paths.append(new_path)
    return child_paths

def backtracking_bssf(edges: list[list[float]], timer: Timer) -> list[SolutionStats]:
    global_best_score = greedy_tour(edges, timer)[0].score
    max_stack_size = 1
    total_partial_states = 1
    total_partial_states_pruned = 0
    cut_tree = CutTree(len(edges))
    stats = []
    stack = [[0]]
    while stack and not timer.time_out():
        path = stack.pop()
        child_paths = expand_path(edges, path)
        total_partial_states += len(child_paths)
        for child_path in child_paths:
            global_best_score, max_stack_size, total_partial_states_pruned, total_partial_states \
                                 = vet_child_path_bssf(edges, stats, stack, 
                                                       child_path, global_best_score, 
                                                       timer, max_stack_size, total_partial_states_pruned, 
                                                       total_partial_states, cut_tree)
    return stats

def vet_child_path_bssf(edges, stats, stack, child_path, 
                        global_best_score, timer, max_stack_size, 
                        total_partial_states_pruned, total_partial_states, cut_tree) -> Tuple[int, int, int, int]:
    cost = score_tour(child_path, edges)
    if len(child_path) == len(edges):
        if cost < global_best_score:
            global_best_score = cost

            cut_tree.cut(child_path)
            total_cut = cut_tree.n_leaves_cut()
            fraction_cut = cut_tree.fraction_leaves_covered()


            stat: SolutionStats = SolutionStats(tour=child_path,
                                               score=cost,
                                               time=timer.time(),
                                               max_queue_size=max_stack_size,
                                               n_nodes_expanded= total_partial_states,
                                               n_nodes_pruned= total_partial_states_pruned,
                                               n_leaves_covered= total_cut,
                                               fraction_leaves_covered=fraction_cut)
            stats.append(stat)
    elif cost < global_best_score:
        stack.append(child_path)
        max_stack_size = max(max_stack_size, len(stack))
    else:
        cut_tree.cut(child_path)
        total_partial_states_pruned += 1
    return global_best_score, max_stack_size, total_partial_states_pruned, total_partial_states