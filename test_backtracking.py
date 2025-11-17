# ---------------------------------- Imports --------------------------------- #
from math import inf

from test_utils import assert_valid_tours
from tsp_solve_backtracking import greedy_tour, backtracking, backtracking_bssf
from utils import Timer, generate_network, score_tour

from byu_pytest_utils import tier

# -------------------------------- Test tiers -------------------------------- #

baseline = tier('baseline', 1)
core = tier('core', 2)
stretch1 = tier('stretch1', 3)

# ------------------------------- Baseline tests ------------------------------ #
@baseline
def test_small_greedy():
    graph = [
        [0, 9, inf, 8, inf],
        [inf, 0, 4, inf, 2],
        [inf, 3, 0, 4, inf],
        [inf, 6, 7, 0, 12],
        [1, inf, inf, 10, 0]
    ]
    timer = Timer(10)
    stats = greedy_tour(graph, timer)
    assert_valid_tours(graph, stats)

    assert stats[0].tour == [1, 4, 0, 3, 2]
    assert stats[0].score == 21

    assert len(stats) == 1


@baseline
def test_multiple_solutions_greedy():
    graph = [
        [0, 1, 2, 3, 4],
        [6, 0, 3, 4, inf],
        [inf, 4, 0, 2, 6],
        [inf, 9, 7, 0, 4],
        [100, 8, 6, 10, 0]
    ]
    timer = Timer(10)
    stats = greedy_tour(graph, timer)
    assert_valid_tours(graph, stats)

    assert len(stats) == 2

    assert stats[0].tour == [0, 1, 2, 3, 4]
    assert stats[0].score == 110

    assert stats[1].tour == [2, 3, 4, 1, 0]
    assert stats[1].score == 22


@baseline
def test_medium_greedy():
    locations, edges = generate_network(
        15,
        euclidean=True,
        reduction=0.2,
        normal=False,
        seed=312,
    )

    timer = Timer(5)
    stats = greedy_tour(edges, timer)
    assert not timer.time_out()
    assert_valid_tours(edges, stats)
    greedy_score = score_tour(stats[-1].tour, edges)

    assert round(greedy_score, 3) == 3.667


@baseline
def test_large_greedy():
    locations, edges = generate_network(
        100,
        euclidean=True,
        reduction=0.2,
        normal=False,
        seed=312,
    )

    timer = Timer(5)
    stats = greedy_tour(edges, timer)
    assert not timer.time_out()
    assert_valid_tours(edges, stats)
    greedy_score = score_tour(stats[-1].tour, edges)

    assert round(greedy_score, 3) == 9.757

@baseline
def test_massive_greedy():
    locations, edges = generate_network(
        200,
        euclidean=True,
        reduction=0.2,
        normal=False,
        seed=312,
    )

    timer = Timer(15)
    stats = greedy_tour(edges, timer)
    assert_valid_tours(edges, stats)

    greedy_score = score_tour(stats[-1].tour, edges)

    assert greedy_score < 16

# ------------------------------- Core tests ------------------------------ #

@core
def test_tiny_backtracking():
    graph = [
        [0, 9, inf, 8, inf],
        [inf, 0, 4, inf, 2],
        [inf, 3, 0, 4, inf],
        [inf, 6, 7, 0, 12],
        [1, inf, inf, 10, 0]
    ]
    timer = Timer(10)
    stats = backtracking(graph, timer)
    assert_valid_tours(graph, stats)

    assert stats[-1].tour == [0, 3, 2, 1, 4]
    assert stats[-1].score == 21


@core
def test_small_backtracking():
    locations, edges = generate_network(
        8,
        euclidean=True,
        reduction=0.2,
        normal=False,
        seed=312,
    )

    timer = Timer(30)
    stats = backtracking(edges, timer)
    assert not timer.time_out()
    assert_valid_tours(edges, stats)
    backtracking_score = score_tour(stats[-1].tour, edges)

    assert round(backtracking_score, 3) == 3.224


@core
def test_medium_backtracking():
    locations, edges = generate_network(
        10,
        euclidean=True,
        reduction=0.5,
        normal=False,
        seed=312
    )

    timer = Timer(30)

    stats = backtracking(edges, timer)

    assert_valid_tours(edges, stats)
    backtracking_score = score_tour(stats[-1].tour, edges)

    assert round(backtracking_score, 3) == 4.416

@core
def test_large_backtracking():
    locations, edges = generate_network(
        14,
        euclidean=True,
        reduction=0,
        normal=False,
        seed=312
    )

    timer = Timer(30)

    stats = backtracking(edges, timer)

    assert_valid_tours(edges, stats)
    backtracking_score = score_tour(stats[-1].tour, edges)

    assert backtracking_score < 15

# -------------------------------- Stretch 1 tests -------------------------------- #


@stretch1
def test_smaller_backtracking_with_bssf():
    locations, edges = generate_network(
        12,
        euclidean=True,
        reduction=0,
        normal=False,
        seed=3,
    )

    backtracking_bssf_timer = Timer(30)
    stats_bssf_backtracking = backtracking_bssf(edges, backtracking_bssf_timer)
    assert backtracking_bssf_timer.time() < 35

    assert_valid_tours(edges, stats_bssf_backtracking)


@stretch1
def test_medium_backtracking_with_bssf():
    locations, edges = generate_network(
        15,
        euclidean=True,
        reduction=0.2,
        normal=False,
        seed=312,
    )

    backtracking_bssf_timer = Timer(30)
    stats_bssf_backtracking = backtracking_bssf(edges, backtracking_bssf_timer)
    assert backtracking_bssf_timer.time() < 35

    assert_valid_tours(edges, stats_bssf_backtracking)
