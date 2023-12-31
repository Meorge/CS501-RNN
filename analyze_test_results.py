from json import load
from os.path import join, isfile
from os import listdir

from rich.table import Table
from rich.console import Console

import numpy as np

console = Console()
TEST_RESULTS_FOLDER = "test_results_v4"

training_data_files = [
    join(TEST_RESULTS_FOLDER, name)
    for name in listdir(TEST_RESULTS_FOLDER)
    if isfile(join(TEST_RESULTS_FOLDER, name)) and name.lower().endswith(".json")
]

tests = []
for fn in training_data_files:
    with open(fn) as f:
        f_data = load(f)

    meta = f_data["meta"]
    metrics = f_data["epoch_metrics"]
    durations = [i["duration"] for i in metrics]
    losses = [i["loss"] for i in metrics]
    train_accs = [i["train_acc"] for i in metrics]
    test_accs = [i["test_acc"] for i in metrics]

    avg_duration = np.average(durations)
    avg_loss = np.average(losses)
    avg_train_acc = np.average(train_accs)
    avg_test_acc = np.average(test_accs)

    tests.append(
        {
            "n_hidden": meta["n_hidden"],
            "lr": meta["lr"],
            "avg_duration": avg_duration,
            "avg_loss": avg_loss,
            "avg_train_acc": avg_train_acc,
            "avg_test_acc": avg_test_acc,
            "max_train_acc": np.max(train_accs),
            "max_test_acc": np.max(test_accs),
            "last_loss": losses[-1],
            "last_train_acc": train_accs[-1],
            "last_test_acc": test_accs[-1],
            "score": 0,
        }
    )


def add_score(key, reverse=False):
    tests.sort(key=lambda i: i[key], reverse=reverse)
    for i, item in enumerate(tests):
        item["score"] += i


# Lowest duration should receive the highest score
add_score("avg_duration", True)

# Lowest loss should receive the highest score
add_score("avg_loss", True)
add_score("last_loss", True)

# Highest train accuracy should receive the highest score
add_score("avg_train_acc")
add_score("last_train_acc")

# Highest test accuracy should receive the highest score
add_score("avg_test_acc")
add_score("last_test_acc")


# Highest scores first
tests.sort(key=lambda i: i["score"], reverse=True)

cols = [
    "HS",
    "LR",
    "Score",
    "Avg Duration",
    "Avg Loss",
    "Final Loss",
    "Avg Train Acc",
    # "Max Train Acc",
    "Final Train Acc",
    "Avg Test Acc",
    # "Max Test Acc",
    "Final Test Acc",
]

table = Table(*cols)

test_strs = [
    [
        f"{test['n_hidden']}",
        f"{test['lr']:.3f}",
        f"{test['score']}",
        f"{test['avg_duration']:.3f}",
        f"{test['avg_loss']:.5f}",
        f"{test['last_loss']:.5f}",
        f"{test['avg_train_acc']:.2f}",
        # f"{test['max_train_acc']:.2f}",
        f"{test['last_train_acc']:.2f}",
        f"{test['avg_test_acc']:.2f}",
        # f"{test['max_test_acc']:.2f}",
        f"{test['last_test_acc']:.2f}",
    ]
    for test in tests
]

for test in test_strs:
    table.add_row(*test)
    
# console.print(table)


# LaTeX table
header = ' & '.join(cols) + r' \\'
body = ""
for row in test_strs:
    body += ' & '.join(row) + r' \\' +'\n\hline\n'

table = header + '\n\hline\n' + body
console.print(table)