import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np

# ----------------------------
# CSP Solver 
# ----------------------------

def is_complete(grid):
    return all(cell != 0 for row in grid for cell in row)

def select_variable_mrv(grid):
    min_domain = 10
    selected = None
    for i in range(9):
        for j in range(9):
            if grid[i][j] == 0:
                domain = get_legal_values((i, j), grid)
                if len(domain) < min_domain:
                    min_domain = len(domain)
                    selected = (i, j)
    return selected

def get_legal_values(var, grid):
    i, j = var
    used = set(grid[i])
    used.update(grid[r][j] for r in range(9))
    start_i, start_j = 3 * (i // 3), 3 * (j // 3)
    for r in range(start_i, start_i + 3):
        for c in range(start_j, start_j + 3):
            used.add(grid[r][c])
    return [num for num in range(1, 10) if num not in used]

def is_valid_assignment(var, value, grid):
    i, j = var
    if value in grid[i] or value in [grid[r][j] for r in range(9)]:
        return False
    start_i, start_j = 3 * (i // 3), 3 * (j // 3)
    for r in range(start_i, start_i + 3):
        for c in range(start_j, start_j + 3):
            if grid[r][c] == value:
                return False
    return True

def assign(var, value, grid):
    i, j = var
    grid[i][j] = value

def unassign(var, grid):
    i, j = var
    grid[i][j] = 0

def get_neighbors(i, j):
    neighbors = set()
    for k in range(9):
        if k != j:
            neighbors.add((i, k))
        if k != i:
            neighbors.add((k, j))
    start_i, start_j = 3 * (i // 3), 3 * (j // 3)
    for r in range(start_i, start_i + 3):
        for c in range(start_j, start_j + 3):
            if (r, c) != (i, j):
                neighbors.add((r, c))
    return neighbors

def forward_check(var, value, grid):
    i, j = var
    for r, c in get_neighbors(i, j):
        if grid[r][c] == 0:
            domain = get_legal_values((r, c), grid)
            if value in domain:
                domain.remove(value)
                if not domain:
                    return None
    return True

def solve(grid):
    if is_complete(grid):
        return grid
    var = select_variable_mrv(grid)
    if not var:
        return None
    for value in get_legal_values(var, grid):
        if is_valid_assignment(var, value, grid):
            assign(var, value, grid)
            if forward_check(var, value, grid):
                result = solve(grid)
                if result:
                    return result
            unassign(var, grid)
    return None

# ----------------------------
# Helper Functions
# ----------------------------

def str_to_grid(s):
    return [[int(s[i * 9 + j]) for j in range(9)] for i in range(9)]

def evaluate_accuracy(pred, truth):
    return sum(1 for i in range(9) for j in range(9) if pred[i][j] == truth[i][j]) / 81 * 100

def count_zeros(grid):
    return sum(cell == 0 for row in grid for cell in row)

# ----------------------------
# Processing & Analysis
# ----------------------------

results = {"Easy": [], "Medium": [], "Hard": []}
print("Processing in memory-efficient chunks...")

chunksize = 500000  

for chunk in pd.read_csv("sudoku.csv", usecols=["puzzle", "solution"], chunksize=chunksize):
    for _, row in chunk.iterrows():
        puzzle_grid = str_to_grid(row["puzzle"])
        solution_grid = str_to_grid(row["solution"])
        zero_count = count_zeros(puzzle_grid)

        # Classify difficulty
        if zero_count < 30:
            level = "Easy"
        elif zero_count < 42:
            level = "Medium"
        else:
            level = "Hard"

        if len(results[level]) >= 500000:
            continue  

        puzzle = [row[:] for row in puzzle_grid]  
        start = time.time()
        result = solve(puzzle)
        end = time.time()

        acc = evaluate_accuracy(result, solution_grid) if result else 0.0
        results[level].append((acc, end - start))

    
    if all(len(results[lvl]) >= 500000 for lvl in results):
        break

# ----------------------------
# Plotting
# ----------------------------

for level in results:
    accs = [x[0] for x in results[level]]
    times = [x[1] for x in results[level]]

    plt.figure(figsize=(10, 4))
    plt.hist(accs, bins=10, edgecolor="black")
    plt.title(f"{level} Puzzle Accuracy Distribution")
    plt.xlabel("Accuracy (%)")
    plt.ylabel("Number of Puzzles")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(times, marker='o', linestyle='', alpha=0.3)
    plt.title(f"{level} Puzzle Solve Time")
    plt.xlabel("Puzzle #")
    plt.ylabel("Time (seconds)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

