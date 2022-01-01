#!/usr/bin/env python3
__doc__ = (
    """Kinematic model of snake-locomotion, with and without friction modulation"""
)

from .kinematic_snake import KinematicSnake, LiftingKinematicSnake, project
from .circle_fit import fit_circle_to_data
from .launch_snake import (
    run_snake,
    run_and_visualize,
    run_phase_space,
    SnakeReader,
    SnakeWriter,
)

Snake = KinematicSnake
LiftingSnake = LiftingKinematicSnake
