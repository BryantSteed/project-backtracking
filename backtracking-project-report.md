# Project Report - Backtracking

## Baseline

### Design Experience

Gabriel Pochman on 11/17/2025

For the greedy solution algorithm, I will simply iterate over every starting node to compute its corresponding SolutionStats. Of course, I will add a SolutionStats when it is better then every other SolutionStat that I have. Then, I will keep a list of nodes that I have visited. At a given traversal, I will ensure that the next node taken is the node with the smallest edge.

### Theoretical Analysis - Greedy

#### Time 

*Fill me in*

#### Space

*Fill me in*

### Empirical Data - Greedy

| N   | reduction | time (ms) |
|-----|-----------|-----------|
| 5   | 0         |           |
| 10  | 0         |           |
| 15  | 0         |           |
| 20  | 0         |           |
| 25  | 0         |           |
| 30  | 0         |           |
| 35  | 0         |           |
| 40  | 0         |           |
| 45  | 0         |           |
| 50  | 0         |           |

### Comparison of Theoretical and Empirical Results - Greedy

- Theoretical order of growth: 
- Empirical order of growth (if different from theoretical):

## Core

### Design Experience

Gabriel Pochman on 11/17/2025

To implement the backtracking TSP problem, I will simply do what the pseudocode does. However, my stack will be a list which I will pop and push from using pop and append, respectively. Each partial path will simply be a list. I'll push a path 0 onto the stack and then start a while loop. After popping a path off the stack, if it is not a solution, I will expand all its child paths b iterating through each vertex that has not been explored by the parent and adding it to the parent (making unique copies of the parent). For each of those, I'll check if its a solution, if so, I'll add it to the solutions list if its better than the ones currently in there. If its not a solution, I add it to the stack.

I will then return the list of solutions when the stack runs out. Of course, if the time runs out I will return early.

### Theoretical Analysis - Backtracking

#### Time 

*Fill me in*

#### Space

*Fill me in*

### Empirical Data - Backtracking

| N   | reduction | time (ms) |
|-----|-----------|-----------|
| 5   | 0         |           |
| 10  | 0         |           |
| 15  | 0         |           |
| 20  | 0         |           |
| 25  | 0         |           |
| 30  | 0         |           |
| 35  | 0         |           |
| 40  | 0         |           |
| 45  | 0         |           |
| 50  | 0         |           |

### Comparison of Theoretical and Empirical Results - Backtracking

- Theoretical order of growth: 
- Empirical order of growth (if different from theoretical): 

### Greedy v Backtracking

*Fill me in*

### Water Bottle Scenario 

#### Scenario 1

**Algorithm:** 

*Fill me in*

#### Scenario 2

**Algorithm:** 

*Fill me in*

#### Scenario 2

**Algorithm:** 

*Fill me in*


## Stretch 1

### Design Experience

Gabriel Pochman on 11/17/2025

This will be very similar to my other backtracking implementation. The initial best path will be computed by using the greedy algorithm instead of assuming a default cost of infinity. Also, when I evaluate a child path (that has just been expanded to), if its not a solution I will check to make sure that its current cost isn't already more than (or equal to perhaps) than the best path. If so, I will cease that path and not add it to the stack.

Other than those two differences, this stretch should be a carbon copy of the previous one.

### Demonstrate BSSF Backtracking Works Better than No-BSSF Backtracking 

*Fill me in*

### BSSF Backtracking v Backtracking Complexity Differences

*Fill me in*

### Time v Solution Cost

![Plot]()

*Fill me in*

## Stretch 2

### Design Experience

Gabriel Pochman on 11/17/2025

*Fill me in*

### Cut Tree

*Fill me in*

### Plots 

*Fill me in*

## Project Review

*Fill me in*
