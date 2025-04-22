import pandas as pd
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
    total = 81
    correct = sum(1 for i in range(9) for j in range(9) if pred[i][j] == truth[i][j])
    return correct / total * 100

def count_zeros(grid):
    return np.sum(np.array(grid) == 0)

# ----------------------------
# Preprocess
# ----------------------------

print("Loading and preparing dataset...")
df = pd.read_csv("sudoku.csv", usecols=["puzzle", "solution"])
df = df.head(500000)  # Change this to control how many rows you use

df["puzzle_grid"] = df["puzzle"].apply(str_to_grid)
df["solution_grid"] = df["solution"].apply(str_to_grid)
df["zeros"] = df["puzzle_grid"].apply(count_zeros)

# Difficulty categorization
easy_df = df[df["zeros"] < 30]
medium_df = df[(df["zeros"] >= 30) & (df["zeros"] < 42)]
hard_df = df[df["zeros"] >= 42]

difficulty_sets = {
    "Easy": easy_df,
    "Medium": medium_df,
    "Hard": hard_df
}

# ----------------------------
# Solve & Analyze
# ----------------------------

results = {}

for level, subset in difficulty_sets.items():
    print(f"Solving {level} puzzles ({len(subset)} rows)...")
    accuracies = []
    solve_times = []

    for _, row in subset.iterrows():
        puzzle = [row for row in row["puzzle_grid"]]  # shallow copy
        solution = row["solution_grid"]
        start = time.time()
        result = solve(puzzle)
        end = time.time()

        if result:
            acc = evaluate_accuracy(result, solution)
        else:
            acc = 0.0

        accuracies.append(acc)
        solve_times.append(end - start)

    avg_acc = sum(accuracies) / len(accuracies) if accuracies else 0.0
    avg_time = sum(solve_times) / len(solve_times) if solve_times else 0.0

    results[level] = {
        "average_accuracy": round(avg_acc, 2),
        "average_time": round(avg_time, 2)
    }

# ----------------------------
# Summary Table
# ----------------------------

print("\nSummary Table:")
print(f"{'Difficulty':<10} | {'Accuracy (%)':<14} | {'Avg Time (s)':<12}")
print("-" * 40)
for level in ["Easy", "Medium", "Hard"]:
    acc = results[level]["average_accuracy"]
    t = results[level]["average_time"]
    print(f"{level:<10} | {acc:<14} | {t:<12}")
