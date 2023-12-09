"""
This is the format that the machine learning model will follow:
  "pu_measurements" - first value is the SU's own measurement; following values
    are the measurements its neighbors reported. These must always be in order.

  "pu_present" - The ground-truth label of whether the primary user is present
    or not. Of course, in practice, secondary users won't know this value; they
    are trying to predict it using the machine learning model.
"""
data_format = [{"pu_measurements": [0, 0, 0, 0, 0], "pu_present": True}]

from ast import arg, parse
from cogsim import Simulator, BaseUser
from cogsim.spatial import User2D
import numpy as np

# rich's print function displays data structures a lot nicer, so if we can
# use it then that's great. but if not, don't crash the program over it
try:
    from rich import print
except ImportError:
    pass

# The transmission power of the PU.
# Units of dB m.
PU_TRANSMIT_POWER: float = 80.0

# The multi-path fading effect.
# Units of dB m?
MULTI_PATH_FADING_EFFECT: float = 0.0

# What honest nodes should report when they do not detect the PU,
# or what malicious nodes should report when they do detect the PU.
# Units of dB m.
NOISE_FLOOR: float = -111.0

# Represented as sigma (σ) in paper.
# Used for calculating the power loss effect.
# Units of dB m.
NORMAL_RNG_SIGMA: float = 3.0

# Represented as alpha (ɑ) in paper.
# Described as "path-loss exponent", but value is not given.
# Online searches suggest a value of 2.0 is good for "free space",
# but a value of 1.0 makes the results line up with the ReDiSen paper the most.
PATH_LOSS_EXPONENT: float = 1.0

rng = np.random.default_rng()


class PrimaryUser(User2D):
    def __init__(
        self,
        x: float,
        y: float,
        time_in_range: tuple[int, int],
        time_out_range: tuple[int, int],
    ):
        super().__init__(x, y)
        self.time_in_range = time_in_range
        self.time_out_range = time_out_range
        self.timer = self.get_new_time_out_of_band()
        self.current_band = None
        self.transmit_power = PU_TRANSMIT_POWER

    def step(self, current_band_contents: list[BaseUser] | None, pass_index: int):
        if pass_index != 0:
            return

        self.timer -= 1
        if self.timer <= 0:
            if self.current_band is None:
                self.current_band = 0
                self.timer = self.get_new_time_in_band()
            else:
                self.current_band = None
                self.timer = self.get_new_time_out_of_band()

    def get_new_time_in_band(self):
        return rng.integers(self.time_in_range[0], self.time_in_range[1])

    def get_new_time_out_of_band(self):
        return rng.integers(self.time_out_range[0], self.time_out_range[1])


class SecondaryUser(User2D):
    def __init__(self, x: float, y: float, attack_probability: float = 1.0):
        super().__init__(x, y)
        self.attack_probability = attack_probability
        self.current_band = 0
        self.reported_value = 0

    def step(self, current_band_contents: list[BaseUser] | None, pass_index: int):
        primary_users = [u for u in current_band_contents if isinstance(u, PrimaryUser)]

        if rng.uniform() <= self.attack_probability:
            """
            Give a dishonest reading of the primary user
            """
            self.reported_value = (
                NOISE_FLOOR if len(primary_users) > 0 else PU_TRANSMIT_POWER
            )
        else:
            """
            Give an honest reading of the primary user
            """
            if len(primary_users) > 0:
                pu = primary_users[0]
                distance_to_pu = self.distance_to(pu)
                power_loss = rng.normal(0.0, NORMAL_RNG_SIGMA)
                self.reported_value = (
                    pu.transmit_power
                    - 10 * PATH_LOSS_EXPONENT * np.log10(distance_to_pu)
                    - power_loss
                    - MULTI_PATH_FADING_EFFECT
                )
            else:
                self.reported_value = NOISE_FLOOR


def simulate(
    time_steps: int = 10,
    time_in_range: tuple[int, int] = (100, 200),
    time_out_range: tuple[int, int] = (50, 100),
    num_good_neighbors: int = 30,
    num_mal_neighbors: int = 5,
    dimensions: tuple[int, int] = (1000, 1000),
    mal_attack_probability: float = 1.0,
):
    """Runs and returns the results from a simulation of neighbors attempting
    to detect the primary user's absence or presence.

    :param time_steps: The number of discrete time steps that the simulation \
        will be run for, defaults to 10
    :type time_steps: int, optional
    :param time_in_range: The minimum and maximum number of time steps that \
        the primary user will stay in the transmitting state for, \
        defaults to (100, 200)
    :type time_in_range: tuple[int, int], optional
    :param time_out_range: The minimum and maximum number of time steps that \
        the primary user will stay in the waiting state for, \
        defaults to (50, 100)
    :type time_out_range: tuple[int, int], optional
    :param num_good_neighbors: The number of good neighbors that the "self" \
        secondary user will have, defaults to 30
    :type num_good_neighbors: int, optional
    :param num_mal_neighbors: The number of malicious neighbors that the \
        "self" secondary user will have, defaults to 5
    :type num_mal_neighbors: int, optional
    :param dimensions: The width and height, in kilometers, of the area \
        that the the users in the network will be placed at random locations \
        within, defaults to (1000, 1000)
    :type dimensions: tuple[int, int], optional
    :return: The measurements from each neighbor, as well as the ground-truth \
        presence of the primary user, at each discrete time step.
    :rtype: list[dict]
    """
    rand_x = lambda: rng.uniform(high=dimensions[0])
    rand_y = lambda: rng.uniform(high=dimensions[1])

    # "The primary user is placed 5km away from the center of the secondary
    # user network", according to the ReDiSen paper. The dimensions are
    # measured in meters, so we move the PU to the midpoint of the network
    # (dividing both width and height by 2) then move it 5000m (5km) to the
    # right.
    primary_user = PrimaryUser(
        x=(dimensions[0] / 2.0) + 5000,
        y=(dimensions[1] / 2.0),
        time_in_range=time_in_range,
        time_out_range=time_out_range,
    )

    self_su = SecondaryUser(x=rand_x(), y=rand_y(), attack_probability=0.0)

    good_neighbors = [
        SecondaryUser(x=rand_x(), y=rand_y(), attack_probability=0.0)
        for _ in range(num_good_neighbors)
    ]
    mal_neighbors = [
        SecondaryUser(x=rand_x(), y=rand_y(), attack_probability=mal_attack_probability)
        for _ in range(num_mal_neighbors)
    ]
    all_neighbors = good_neighbors + mal_neighbors

    all_users = [primary_user, self_su] + all_neighbors

    history = []
    sim = Simulator(num_bands=1, users=all_users, passes=1)
    for _ in range(time_steps):
        sim.step()
        history.append(
            {
                "self_measurement": self_su.reported_value,
                "neighbor_measurements": [s.reported_value for s in all_neighbors],
                "pu_present": primary_user.current_band is not None,
            }
        )

    return history


if __name__ == "__main__":
    import argparse
    from json import dump
    from time import time

    parser = argparse.ArgumentParser(
        prog="CognitiveRadioSimulator",
        description="Generates data sets for cooperative spectrum sensing readings",
    )

    parser.add_argument("-o", "--output")
    parser.add_argument("-d", "--duration", type=int, default=100)
    parser.add_argument("-g", "--good-neighbors", type=int, default=30)
    parser.add_argument("-m", "--mal-neighbors", type=int, default=5)

    parser.add_argument("--min-pu-on", type=int, default=5)
    parser.add_argument("--max-pu-on", type=int, default=10)
    parser.add_argument("--min-pu-off", type=int, default=1)
    parser.add_argument("--max-pu-off", type=int, default=5)

    parser.add_argument("--width", type=float, default=1000)
    parser.add_argument("--height", type=float, default=1000)

    parser.add_argument("--mal-attack-probability", type=float, default=1.0)

    args = parser.parse_args()
    result = simulate(
        time_steps=args.duration,
        time_in_range=(args.min_pu_on, args.max_pu_on),
        time_out_range=(args.min_pu_off, args.max_pu_off),
        num_good_neighbors=args.good_neighbors,
        num_mal_neighbors=args.mal_neighbors,
        dimensions=(args.width, args.height),
        mal_attack_probability=args.mal_attack_probability
    )

    params = {
        "duration": args.duration,
        "pu_in_range": [args.min_pu_on, args.max_pu_on],
        "pu_out_range": [args.min_pu_off, args.max_pu_off],
        "num_good_neighbors": args.good_neighbors,
        "num_mal_neighbors": args.mal_neighbors,
        "dimensions": [args.width, args.height],
        "mal_attack_probability": args.mal_attack_probability
    }

    full_data = {"params": params, "data": result}

    filename = args.output
    if filename is None:
        filename = f"output-{int(time())}.json"

    with open(filename, "w") as f:
        dump(full_data, f, indent=4)
