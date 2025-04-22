import pandas as pd
import time

# ----------------------------
# CSP Solver Without Backtracking
# ----------------------------

def is_complete(grid):
    return all(cell != 0 for row in grid for cell in row)

def select_variable_mrv(grid):
    min_domain = 10
    selected = None
    for i in range(9):
        for j in range(9):
            if grid[i][j] == 0:
                domain = get_legal_values(i, j, grid)
                if len(domain) < min_domain:
                    min_domain = len(domain)
                    selected = (i, j)
    return selected

def get_legal_values(i, j, grid):
    used = set(grid[i])
    used.update(grid[r][j] for r in range(9))
    start_i, start_j = 3 * (i // 3), 3 * (j // 3)
    used.update(grid[r][c] for r in range(start_i, start_i+3) for c in range(start_j, start_j+3))
    return [n for n in range(1, 10) if n not in used]

def solve_no_backtracking(grid):
    if is_complete(grid):
        return grid
    var = select_variable_mrv(grid)
    if not var:
        return None
    i, j = var
    for value in get_legal_values(i, j, grid):
        grid[i][j] = value
        return solve_no_backtracking(grid)
    return None

# ----------------------------
# Utils
# ----------------------------

def str_to_grid(s):
    return [[int(s[i * 9 + j]) for j in range(9)] for i in range(9)]

def evaluate_accuracy(pred, truth):
    return 100 * sum(pred[i][j] == truth[i][j] for i in range(9) for j in range(9)) / 81

def count_zeros(s):
    return sum(1 for c in s if c == '0')

# ----------------------------
# Load + Run
# ----------------------------

def run_experiment(filepath="sudoku.csv", rows_per_level=100):
    df = pd.read_csv(filepath, usecols=["puzzle", "solution"])
    df["zeros"] = df["puzzle"].apply(count_zeros)

    results = {}

    for level, cond in {
        "Easy": df["zeros"] < 30,
        "Medium": (df["zeros"] >= 30) & (df["zeros"] < 42),
        "Hard": df["zeros"] >= 42
    }.items():
        subset = df[cond].head(rows_per_level)
        accs, times = [], []

        for _, row in subset.iterrows():
            puzzle = str_to_grid(row["puzzle"])
            solution = str_to_grid(row["solution"])

            start = time.time()
            result = solve_no_backtracking(puzzle)
            end = time.time()

            acc = evaluate_accuracy(result, solution) if result else 0.0
            accs.append(acc)
            times.append(end - start)

        results[level] = {
            "average_accuracy": round(sum(accs)/len(accs), 2),
            "average_time": round(sum(times)/len(times), 4)
        }

    # ----------------------------
    # Output
    # ----------------------------
    print(f"\n{'Difficulty':<10} | {'Accuracy (%)':<14} | {'Avg Time (s)':<12}")
    print("-" * 42)
    for level in ["Easy", "Medium", "Hard"]:
        acc = results[level]["average_accuracy"]
        t = results[level]["average_time"]
        print(f"{level:<10} | {acc:<14} | {t:<12}")

# Run  experiment
run_experiment("sudoku.csv", rows_per_level=100)

