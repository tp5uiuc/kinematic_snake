__doc__ = " Examples exploring solution space  "

import numpy as np
from kinematic_snake import KinematicSnake, LiftingKinematicSnake, run_and_visualize


def run_nonlifting_snake_with_default_params():
    """
    1.1 Running a single case with default parameters for (curvature)
    activation (referred to as simply activation) for a non-lifting snake.
    The default activation is a cos wave with
    𝜅 = ε cos (k 𝛑 (s + t))
    where ε = 7.0 and k = 2.0
    """
    snake, sol_history, time_period = run_and_visualize(
        froude=1,
        time_interval=[0.0, 10.0],
        snake_type=KinematicSnake,
        mu_f=1.0,
        mu_b=1.5,
        mu_lat=2.0,
    )


def animate_nonlifting_snake_with_default_params():
    """
    For animating the motion instead of simply visualizing use
    animate = True
    """
    snake, sol_history, time_period = run_and_visualize(
        froude=1,
        time_interval=[0.0, 10.0],
        snake_type=KinematicSnake,
        mu_f=1.0,
        mu_b=1.5,
        mu_lat=2.0,
        # Flag!
        animate=True,
    )


def run_nonlifting_snake_with_custom_params():
    """
    1.2 For running a single case with non-default parameters for
    activation (that is changing ε, k) in the activation below simply
    𝜅 = ε cos (k 𝛑 (s + t))
    please do the following. Note that the order of parameters beyond
    the keyword `snake_type` does not matter.
    """
    epsilon = 5.0
    wave_number = 5.0
    snake, sol_history, time_period = run_and_visualize(
        froude=1,
        time_interval=[0.0, 10.0],
        snake_type=KinematicSnake,
        mu_f=1.0,
        mu_b=1.5,
        mu_lat=2.0,
        epsilon=epsilon,
        wave_number=wave_number,
    )


def run_lifting_snake_with_default_params():
    """
    2.1 If you want a Lifting snake with default parameters (for both
    activation and lifting activation), change the `snake_type` keyword
    as shown below.

    The default curvature activation is 𝜅 = ε cos (k_1 𝛑 (s + t))
    The default lifting activation is A = 1 + lift_amp * cos (k_2 𝛑 (s + t + φ) )
    where φ is the phase-difference between the curvature and lift waves. The
    lift waves is switched on instantaneously after t = 2.0 seconds

    Here ε = 7.0 and k_1 = 2.0 are defaulted while lift_amp = 1.0, k_2 = k_1 and
    φ = 0.26 are set as defaults
    """
    snake, sol_history, time_period = run_and_visualize(
        froude=1,
        time_interval=[0.0, 10.0],
        snake_type=LiftingKinematicSnake,
        mu_f=1.0,
        mu_b=1.5,
        mu_lat=2.0,
    )


def run_lifting_snake_with_custom_params():
    """
    2.2 If you want to specify non-default parameters for either the curvature
    or lifting wave, please do. Once again, order of keywords do not matter
    """
    epsilon = 5.0
    wave_number = 5.0
    lift_amp = 0.4
    lift_wave_number = 2.0
    phase = 0.3

    snake, sol_history, time_period = run_and_visualize(
        froude=1,
        time_interval=[0.0, 10.0],
        snake_type=LiftingKinematicSnake,
        mu_f=1.0,
        mu_b=1.5,
        mu_lat=2.0,
        epsilon=epsilon,
        wave_number=wave_number,
        lift_amp=lift_amp,
        lift_wave_number=lift_wave_number,
        phase=phase,
    )


def run_snake_with_custom_activation():
    """
    3. If you want to rather provide your own activation functions, you can
    do so as shown below. Note that both the activation and lifting activation
    need to be a function of (s,time_v) where s is the centerline and time_v
    is the simulation time. This enables rapid prototyping for quickly changing
    activation (say rampup time etc.)

    Note that both are optional, hence if you want to retain default curvature
    activation, but change lifting activation, simply provide only the lifting
    activation function!

    In case you are running a non-lifting snake, then just set the custom
    curvature activation, the lifting activation will automatically be ignored.
    """

    epsilon = 7.0
    wave_number = 2.0
    lift_amp = 0.0
    phase = 0.26

    # Needs to be differentiated for which we use sympy
    import sympy as sp

    def my_custom_activation(s, time_v):
        return epsilon * sp.cos(wave_number * sp.pi * (s + time_v))

    from scipy.integrate import trapz

    def my_custom_lifting_activation(s, time_v):
        if time_v > 2.0:
            liftwave = (
                lift_amp * np.cos(wave_number * np.pi * (s + phase + time_v)) + 1.0
            )
            np.maximum(0, liftwave, out=liftwave)
            return liftwave / trapz(liftwave, s)
        else:
            return 1.0 + 0.0 * s

    snake, sol_history, time_period = run_and_visualize(
        froude=1,
        time_interval=[0.0, 10.0],
        snake_type=LiftingKinematicSnake,
        mu_f=1.0,
        mu_b=1.5,
        mu_lat=2.0,
        activation=my_custom_activation,
        lifting_activation=my_custom_lifting_activation,
    )


if __name__ == "__main__":
    animate_nonlifting_snake_with_default_params()
