import matplotlib.pyplot as plt
from json import load
from os.path import join, isfile
from os import listdir


def make_plot(content):
    meta = content["meta"]
    title = f"Hidden size {meta['n_hidden']} with LR {meta['lr']}"

    epoch_losses = [i[1] for i in content["epoch_losses"]]
    plt.plot(epoch_losses)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f"hidden-{meta['n_hidden']}-lr-{meta['lr']}.png")


TEST_RESULTS_FOLDER = "test_results"
test_result_files = [
    join(TEST_RESULTS_FOLDER, name)
    for name in listdir(TEST_RESULTS_FOLDER)
    if isfile(join(TEST_RESULTS_FOLDER, name)) and name.lower().endswith(".json")
]

for fn in test_result_files:
    with open(fn) as f:
        make_plot(load(f))
