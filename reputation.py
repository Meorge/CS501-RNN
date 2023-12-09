import numpy as np

def reputation(array: np.ndarray, print_verbose=False):
    """Calculates the ReDiSen-style reputation for each value in an array.

    :param array: A 1D array of float values; each one represents the \
        reported value of a single node. The reputations returned by the \
        function will map to these nodes by index.
    :type array: np.ndarray
    """
    # NOTE: I'm confused by the output of this function on test data, such as
    #  the array [92, 93, 91, -40, 59, 83]. I get a negative value for the
    #  reputation of the user at index 3 (with the reported value of -40).
    #  
    #  The ReDiSen paper states that, with this formula, 0 <= R_{j,i} <= 2,
    #  but that isn't the case here.
    average_value = np.average(array)
    distances_from_average = np.abs(array - average_value)
    denominator = np.sum(distances_from_average)
    final_reputations = 2 - ((len(array) * distances_from_average) / denominator)

    if print_verbose:
        print(f"{array=}")
        print(f"{average_value=}")
        print(f"{distances_from_average=}")
        print(f"{denominator=}")
        print(f"{len(array)=}")
        print(f"{final_reputations=}")

    return final_reputations


if __name__ == "__main__":
    reputation(np.array([92, 93, 91, -40, 59, 83]))
    # from json import load
    # with open("outfile", "r") as f:
    #     d = load(f)
    
    # reputation(np.array([d["data"][0]["self_measurement"]] + d["data"][0]["neighbor_measurements"]), print_verbose=True)
