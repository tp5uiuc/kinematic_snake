import numpy as np
from functools import partial
from sympy import sin, cos, pi
from scipy.integrate import solve_ivp, trapz

from kinematic_snake.kinematic_snake import KinematicSnake


def run_snake(froude, time_interval=[0.0, 5.0], **kwargs):
    # Set a sane default value, which is updated using kwargs
    friction_coefficient_ratios = {"a_f": 1, "a_b": 1.4, "a_lat": 2.0}
    friction_coefficient_ratios.update(kwargs)
    forward_friction_coefficient = kwargs.get(
        "friction_coefficient", 1.0 / (1.0 ** 2 * 9.81 * froude)
    )  # for now
    friction_coefficients = {"mu_f": None, "mu_b": None, "mu_lat": None}
    for direction in ["f", "b", "lat"]:
        src_key = "a_" + direction
        tgt_key = "mu_" + direction
        # friction_coefficients[tgt_key] = forward_friction_coefficient * friction_coefficient_ratios[src_key]
        friction_coefficients[tgt_key] = friction_coefficient_ratios[src_key]

    snake = KinematicSnake(
        froude_number=froude, friction_coefficients=friction_coefficients
    )

    def activation(s, time, epsilon, wave_number):
        return epsilon * cos(wave_number * pi * (s + time))

    wave_number = kwargs.get("wave_number", 2.0)
    bound_activation = kwargs.get(
        "activation",
        partial(
            activation, epsilon=kwargs.get("epsilon", 7.0), wave_number=wave_number,
        ),
    )
    snake.set_activation(bound_activation)

    sol = solve_ivp(snake, time_interval, snake.state.copy().reshape(-1, ), method="RK23")
    # omega * time = wave-number * pi * time
    # omega = 2 * pi * freq
    # -> freq = wave_number/2 and T = 1./freq
    return snake, sol, 2.0 / wave_number


def run_and_visualize(*args, **kwargs):
    snake, sol_history, time_period = run_snake(*args, **kwargs)

    from matplotlib import pyplot as plt
    from matplotlib.colors import to_rgb

    with plt.style.context("fivethirtyeight"):
        phys_space_fig = plt.figure(figsize=(10, 8))
        phys_space_ax = phys_space_fig.add_subplot(111)
        n_steps = sol_history.t.size
        skip = int(n_steps / 20)
        # skip = 1
        n_steps = sol_history.t[::skip].size
        for step, (time, solution) in enumerate(
                zip(sol_history.t[::skip], sol_history.y.T[::skip])
        ):
            snake.state = solution.reshape(-1, 1)
            snake._construct(time)
            ext_force = snake.external_force_distribution(time)
            phys_space_ax.plot(
                snake.x_s[0, ...],
                snake.x_s[1, ...],
                c=to_rgb("xkcd:bluish"),
                alpha=10 ** (step / n_steps - 1),
                lw=2,
            )
        quiver_skip = int(snake.centerline.size / 30)
        phys_space_ax.quiver(
            snake.x_s[0, ::quiver_skip],
            snake.x_s[1, ::quiver_skip],
            ext_force[0, ::quiver_skip],
            ext_force[1, ::quiver_skip],
        )
        phys_space_ax.set_aspect("equal")

        velocity_fig = plt.figure(figsize=(10, 8))
        velocity_ax = velocity_fig.add_subplot(111)

        # calculate snake motion axes based on last-three time-periods
        n_past_periods = 3
        final_time = kwargs["time_interval"][1]
        past_period_idx = np.argmin(
            np.abs(sol_history.t - (final_time - n_past_periods * time_period))
        )

        # as a bonus, plot it on the physical axes
        phys_space_ax.plot(
            [sol_history.y[0, past_period_idx], sol_history.y[0, -1]],
            [sol_history.y[1, past_period_idx], sol_history.y[1, -1]],
            c=to_rgb("xkcd:reddish"),
            lw=2,
        )

        travel_axes = sol_history.y[:2, -1] - sol_history.y[:2, past_period_idx]
        travel_axes /= np.linalg.norm(travel_axes, ord=2)
        com_velocity = sol_history.y[3:5, ...]
        # Broadcast shape
        lateral_axes = np.array([-travel_axes[1], travel_axes[0]])
        travel_axes = travel_axes.reshape(-1, 1) + 0.0 * com_velocity
        lateral_axes = lateral_axes.reshape(-1, 1) + 0.0 * com_velocity
        from kinematic_snake.kinematic_snake import project

        mag_projected_velocity_along_travel, _ = project(com_velocity, travel_axes)
        mag_projected_velocity_along_lateral, _ = project(com_velocity, lateral_axes)
        projected_velocity = np.vstack(
            (mag_projected_velocity_along_travel, mag_projected_velocity_along_lateral)
        )

        # calculate average statistics
        avg_projected_velocity = trapz(
            projected_velocity[..., past_period_idx:],
            sol_history.t[past_period_idx:],
            axis=-1,
        ) / (n_past_periods * time_period)

        velocity_ax.plot(sol_history.t, projected_velocity[0], lw=2, label="fwd")
        velocity_ax.plot(sol_history.t, projected_velocity[1], lw=2, label="trans")
        velocity_ax.hlines(
            y=avg_projected_velocity[0],
            xmin=sol_history.t[0],
            xmax=sol_history.t[-1],
            lw=2,
            colors="k",
            linestyles="dashed",
        )
        velocity_ax.hlines(
            y=avg_projected_velocity[1],
            xmin=sol_history.t[0],
            xmax=sol_history.t[-1],
            lw=2,
            colors="k",
            linestyles="dashed",
        )
        velocity_ax.plot(
            sol_history.t, com_velocity[0], lw=1, linestyle="--", label="x"
        )
        velocity_ax.plot(
            sol_history.t, com_velocity[1], lw=1, linestyle="--", label="y"
        )
        velocity_ax.legend()

    phys_space_fig.show()
    velocity_fig.show()
    print(avg_projected_velocity)
    return snake, sol_history


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
    """
    Running a single case
    """
    # snake, sol = run_and_visualize(froude=1e-3, time_interval=[0.0, 10.0], epsilon=7.0)
    snake, sol_history = run_and_visualize(
        froude=0.1, time_interval=[0.0, 20.0],
        # a_f=1.0, a_b=1.27, a_lat=1.81
    )

    """
    Running a phase-space
    """
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
