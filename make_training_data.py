"""
This is the format that the machine learning model will follow:
  "pu_measurements" - first value is the SU's own measurement; following values
    are the measurements its neighbors reported. These must always be in order.

  "pu_present" - The ground-truth label of whether the primary user is present
    or not. Of course, in practice, secondary users won't know this value; they
    are trying to predict it using the machine learning model.
"""
data_format = [{"pu_measurements": [0, 0, 0, 0, 0], "pu_present": True}]

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
        self.transmit_power = 4.0  # watts

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


def simulate(**kwargs):
    time_steps = kwargs.get("time_steps", 10)
    time_in_range = kwargs.get("time_in_range", (100, 200))
    time_out_range = kwargs.get("time_out_range", (50, 100))

    primary_user = PrimaryUser(
        x=0, y=0, time_in_range=time_in_range, time_out_range=time_out_range
    )

    self_su = SecondaryUser(x=0, y=0, attack_probability=0.0)

    other_su: list[SecondaryUser] = []

    all_users = [primary_user, self_su] + other_su

    history = []
    sim = Simulator(num_bands=1, users=all_users, passes=1)
    for _ in range(time_steps):
        sim.step()
        history.append(
            {
                "pu_measurements": [self_su.reported_value]
                + [s.reported_value for s in other_su],
                "pu_present": primary_user.current_band is not None,
            }
        )

    return history

if __name__ == "__main__":
    simulate()
