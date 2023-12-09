import numpy as np
from reputation import reputation


def make_features(data: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    all_measurements = []
    all_reputations = []
    for ts in data:
        ts_measurements = np.array(
            [ts["self_measurement"]] + ts["neighbor_measurements"]
        )
        ts_reputations = reputation(ts_measurements)

        all_measurements.append(ts_measurements)
        all_reputations.append(ts_reputations)

    # Now that we have the measurements and reputations, we can get
    # the reputation history for each user.

    most_recent_values = []
    reputations_by_user = []
    for user_i in range(len(all_measurements[0])):
        most_recent_value = all_measurements[-1][user_i]
        reputations = [g[user_i] for g in all_reputations]

        most_recent_values.append(most_recent_value)
        reputations_by_user.append(reputations)

    most_recent_values = np.array(most_recent_values)
    reputations_by_user = np.array(reputations_by_user)
    
    return most_recent_values, reputations_by_user

if __name__ == "__main__":
    from json import load
    with open("outfile", "r") as f:
        d = load(f)
    v, r = make_features(d["data"])
    for i, rep in enumerate(r):
        print(f"user {i} reputations", rep)
