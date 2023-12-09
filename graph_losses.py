import matplotlib.pyplot as plt
from json import load
from os.path import join, isfile
from os import listdir

TEST_RESULTS_FOLDER = "test_results_v4"


def make_plot(content):
    meta = content["meta"]
    config = f"hidden size {meta['n_hidden']} with LR {meta['lr']}"
    epoch_losses = [i["loss"] for i in content["epoch_metrics"]]

    epoch_train_accs = [i["train_acc"] for i in content["epoch_metrics"]]
    epoch_test_accs = [i["test_acc"] for i in content["epoch_metrics"]]

    # Plot losses
    plt.cla()
    plt.plot(epoch_losses)
    plt.title(f"Loss for {config}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(
        join(TEST_RESULTS_FOLDER, f"loss-{meta['n_hidden']}-lr-{meta['lr']}.png")
    )

    # Plot train accuracy
    plt.cla()
    plt.ylim(0.0, 1.0)
    plt.plot(epoch_train_accs, label="Train")
    plt.plot(epoch_test_accs, label="Test")
    plt.title(f"Accuracy for {config}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.savefig(join(TEST_RESULTS_FOLDER, f"acc-{meta['n_hidden']}-lr{meta['lr']}.png"))


test_result_files = [
    join(TEST_RESULTS_FOLDER, name)
    for name in listdir(TEST_RESULTS_FOLDER)
    if isfile(join(TEST_RESULTS_FOLDER, name)) and name.lower().endswith(".json")
]

for fn in test_result_files:
    with open(fn) as f:
        make_plot(load(f))
