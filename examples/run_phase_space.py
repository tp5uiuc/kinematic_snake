__doc__ = " Examples for running a phase space  "

import numpy as np
from kinematic_snake import (
    KinematicSnake,
    LiftingKinematicSnake,
    run_phase_space,
    SnakeReader,
)
from collections import OrderedDict


def run_phase_space_for_nonlifting_snake():
    """
    4.1 You can run a phase-space using any of the options seen before (including
    custom functions and so on). However, due to some I/O limitations, it becomes
    hard to store data if custom functions are passed (even with packages like `dill`
    nd `pickle`). So for now let's restrict the list of possible functions to only
    the default ones provided.

    With these functions, here's how you run a phase-space. We create a dict of lists
    as parameters as shown below. The `run_phase_space` function creates a cartesian
    product and then runs all simulations and stores the data for later re-use.

    Note that we need to provide whether the phase-space is for a KinematicSnake or a
    LiftingSnake below, which also gets forwarded to the simulation.

    The order of keys below does not matter, the phase-space function takes care of
    appropriately re-arranging it.

    """
    kwargs = OrderedDict(
        {
            "froude": [1.0, 2.0, 3.0],
            "time_interval": [[0.0, 20.0]],
            "mu_f": [1.0, 2.0],
            "mu_b": [3.0, 4.0],
            "mu_lat": [3.0, 4.0],
            "epsilon": [2.0, 4.0, 7.0],
            "wave_number": [4.0, 5.0, 6.0],
        }
    )

    ps = run_phase_space(snake_type=KinematicSnake, **kwargs)


def run_phase_space_for_lifting_snake():
    """
    4.2 Phase-space for a LiftingSnake
    """
    kwargs = OrderedDict(
        {
            "froude": [1.0, 2.0, 3.0],
            "time_interval": [[0.0, 20.0]],
            "mu_f": [1.0, 2.0],
            "mu_b": [3.0, 4.0],
            "mu_lat": [3.0, 4.0],
            "epsilon": [2.0, 4.0, 7.0],
            "wave_number": [4.0, 5.0, 6.0],
            "lift_wave_number": [2.0, 3.0],
            "lift_amp": [2.0, 3.0],
            "phase": [0.25, 0.5, 0.75],
        }
    )

    ps = run_phase_space(snake_type=LiftingKinematicSnake, **kwargs)


def load_snake():
    """
    5. If you want to load up a saved-snake from file (after running a
    phase-space), you can consult the simulation_id from the `ids.csv` file
    that the simulation creates within the `data` folder in the home directory.

    Once you know the simulation id you can pass it to the SnakeReader as shown
    below, which constructs the snake and the time-history of COM motions. With
    these two parameters, its possible to reconstruct all desired quantitites.
    Look at `run_and_visualize` for a more complete example
    """
    snake, sol_history, time_period = SnakeReader.load_snake_from_disk(1)


if __name__ == "__main__":
    run_phase_space_for_nonlifting_snake()
