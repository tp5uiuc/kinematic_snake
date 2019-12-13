import numpy as np
from functools import partial
from sympy import sin, cos, pi
from scipy.integrate import solve_ivp

from kinematic_snake.kinematic_snake import KinematicSnake


def run_snake(froude, time_interval=[0.0, 5.0], **kwargs):
    # Set a sane default value, which is updated using kwargs
    friction_coefficients = {"mu_f": 0.11, "mu_b": 0.14, "mu_lat": 0.19}
    friction_coefficients.update(kwargs)

    snake = KinematicSnake(
        froude_number=froude, friction_coefficients=friction_coefficients
    )

    def activation(s, time, epsilon, wave_number):
        return epsilon * cos(wave_number * pi * (s + time))

    bound_activation = kwargs.get(
        "activation",
        partial(
            activation,
            epsilon=kwargs.get("epsilon", 7.0),
            wave_number=kwargs.get("wave_number", 2.0),
        ),
    )
    snake.set_activation(bound_activation)

    sol = solve_ivp(
        snake, time_interval, snake.state.copy().reshape(-1,), method="RK45"
    )
    return snake, sol


def run_and_visualize(*args, **kwargs):
    snake, sol_history = run_snake(*args, **kwargs)

    from matplotlib import pyplot as plt
    from matplotlib.colors import to_rgb

    with plt.style.context("fivethirtyeight"):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        n_steps = sol_history.t.size
        for step, (time, solution) in enumerate(zip(sol_history.t, sol_history.y.T)):
            snake.state = solution.reshape(-1, 1)
            snake._construct(time)
            ax.plot(
                snake.x_s[0, ...],
                snake.x_s[1, ...],
                c=to_rgb("xkcd:bluish"),
                alpha=10 ** (step / n_steps - 1),
                lw=2,
            )

    ax.set_aspect("equal")
    fig.show()


def run_phase_space(**kwargs):
    from p_tqdm import p_map

    # Detect which elements in kwargs are parameterized
    # by multiple values. For only those elements,
    # unpack the list and put them in a special kwargs
    # Possible variations include:
    # 1. Fr
    # 2. mu_f, mu_b, mu_t
    # 3. activations
    # 4. epsilon
    # 5. wavenumbers

    from itertools import product

    def cartesian_product(**kwargs):
        keys = kwargs.keys()
        vals = kwargs.values()
        for instance in product(*vals):
            yield dict(zip(keys, instance))

    phase_space_kwargs = cartesian_product(**kwargs)
    added = p_map(fwd_to_run_snake, list(phase_space_kwargs))


def fwd_to_run_snake(kwargs):
    print(kwargs)
    # run_snake(**kwargs)


if __name__ == "__main__":
    run_and_visualize(froude=1e-1, time_interval=[0.0, 20.0])

    # kwargs = {
    #     "froude": [1.0, 2.0, 3.0],
    #     "time_interval": [[0.0, 20.0]],
    #     "mu_f": [1.0, 2.0],
    #     "mu_b": [3.0, 4.0],
    #     "mu_t": [3.0, 4.0],
    #     "epsilon" : [2.0, 4.0, 7.0],
    #     "wave_numbers" : [4., 5., 6.]
    # }
    # ps = run_phase_space(**kwargs)
