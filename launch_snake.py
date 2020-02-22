import numpy as np
import pickle
from functools import partial
from sympy import sin, cos, pi
from scipy.integrate import solve_ivp, trapz
from collections import OrderedDict
from kinematic_snake.kinematic_snake import KinematicSnake, LiftingKinematicSnake
from os import path, makedirs


def make_snake(froude, time_interval, snake_type, **kwargs):
    friction_coefficients = {"mu_f": None, "mu_b": None, "mu_lat": None}
    friction_coefficients.update(kwargs)

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

    if snake_type == LiftingKinematicSnake:
        def lifting_activation(s, time_v, phase, lift_amp, lift_wave_number):
            if time_v > 2.0:
                liftwave = lift_amp * np.cos(lift_wave_number * np.pi * (s + phase + time_v)) + 1.0
                np.maximum(0, liftwave, out=liftwave)
                return liftwave / trapz(liftwave, s)
            else:
                return 1.0 + 0.0 * s

        bound_lifting_activation = kwargs.get(
            "lifting_activation",
            partial(
                lifting_activation, phase=kwargs.get("phase", 0.26), lift_amp=kwargs.get("lift_amp", 1.0),
                wave_number=kwargs.get("lift_wave_number", wave_number),
            )
        )
        snake.__class__ = LiftingKinematicSnake
        snake.set_lifting_activation(bound_lifting_activation)

    return snake, wave_number


def run_snake(froude, time_interval=[0.0, 5.0], snake_type=KinematicSnake, **kwargs):
    snake, wave_number = make_snake(froude, time_interval, snake_type, **kwargs)

    # Generate t_eval so that simulation stores data at this point, useful for computing
    # cycle-bases statistics and so on...
    if "activation" not in kwargs:
        # omega * time = wave-number * pi * time
        # omega = 2 * pi * freq
        # -> freq = wave_number/2 and T = 1./freq
        time_period = 2.0 / wave_number
        periods_within_interval = int(time_interval[1] / time_period)
        t_event = np.arange(1.0, periods_within_interval) * time_period
        events = [lambda t, y, x=x: (t - x) for x in t_event]
    else:
        time_period = None
        events = None

    sol = solve_ivp(
        snake,
        time_interval,
        snake.state.copy().reshape(-1, ),
        method="RK23",
        events=events,
    )

    # Append t_events and y_events to the final solution history
    # Monkey patching
    if events is not None:
        insert_idx = np.searchsorted(sol.t, sol.t_events)
        sol.t = np.insert(sol.t, insert_idx[:, 0], np.array(sol.t_events).reshape(-1, ))
        sol.y = np.insert(
            sol.y, insert_idx[:, 0], np.squeeze(np.array(sol.y_events)).T, axis=1
        )

    if time_period is None:
        return snake, sol, 1.0  # Fake time-period in absence of any other data
    else:
        return snake, sol, time_period


class SnakeIO:
    data_folder_name = "data"
    id_file_name = path.join(data_folder_name, "ids.csv")
    pickled_file_name = path.join(data_folder_name, "{id:05d}_results.pkl")


class SnakeReader(SnakeIO):
    @staticmethod
    def load_snake_from_disk(id: int):

        import csv
        import ast

        keys = []
        values = []
        snake_cls = None

        def row_count(filename):
            with open(filename) as in_file:
                return sum(1 for _ in in_file)

        n_lines = row_count(SnakeReader.id_file_name)

        with open(SnakeReader.id_file_name, "r", newline="") as id_file_handler:
            # look for id in first column
            # Read header last
            csvreader = csv.reader(id_file_handler, delimiter=",")
            for row in csvreader:
                if csvreader.line_num == 1:
                    snake_cls = row[0]  # take name of class
                if csvreader.line_num == 2:
                    keys = row[1:]  # take everything except id
                if str(id) == (row[0]):
                    values = [
                        ast.literal_eval(x) for x in row[1:]
                    ]  # take everything except id
                    break
                if csvreader.line_num == n_lines:
                    raise IndexError("End of id file reached, requested id not present")

        if values:
            kwargs = dict(zip(keys, values))
        else:
            raise ValueError("Cannot load keys and values from csv file")

        pickled_file_name = SnakeReader.pickled_file_name.format(id=id)

        with open(pickled_file_name, "rb") as pickled_file:
            sol = pickle.load(pickled_file)

        ### Can use eval, but dangerous
        if snake_cls == KinematicSnake.__name__:
            snake_cls = KinematicSnake
        elif snake_cls == LiftingKinematicSnake.__name__:
            snake_cls = LiftingKinematicSnake
        else:
            raise ValueError("The name of class from disk is invalid! Please raise an"
                             "error report on Github!")

        snake, wave_number = make_snake(
            froude=kwargs.pop("froude"),
            time_interval=kwargs.pop("time_interval"),
            snake_type=snake_cls,
            **kwargs
        )

        return snake, sol


class SnakeWriter(SnakeIO):
    @staticmethod
    def write_ids_to_disk(snake_type, phase_space_ids, phase_space_kwargs):
        makedirs(SnakeWriter.data_folder_name, exist_ok=True)

        import csv

        with open(SnakeWriter.id_file_name, "w", newline="") as id_file_handler:
            # Write header first
            csvwriter = csv.writer(id_file_handler, delimiter=",")
            snake_type_str = [snake_type.__name__]
            csvwriter.writerow(snake_type_str)
            header_row = ["id"] + list(phase_space_kwargs[0].keys())
            csvwriter.writerow(header_row)
            for id, args_dict in zip(phase_space_ids, phase_space_kwargs):
                temp = [id, snake_type]
                temp.extend(args_dict.values())
                csvwriter.writerow(temp)

    @staticmethod
    def write_snake_to_disk(id: int, sol):
        with open(SnakeWriter.pickled_file_name.format(id=id), "wb") as file_handler:
            # # Doesn't work
            # pickle.dump([snake, sol], file_handler)
            # Works
            pickle.dump(sol, file_handler)


def run_and_visualize(*args, **kwargs):
    snake, sol_history, time_period = run_snake(*args, **kwargs)

    from matplotlib import pyplot as plt
    from matplotlib.colors import to_rgb

    with plt.style.context("fivethirtyeight"):
        phys_space_fig = plt.figure(figsize=(10, 8))
        phys_space_ax = phys_space_fig.add_subplot(111)
        n_steps = sol_history.t.size
        skip = max(int(n_steps / 20), 1)
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
    plt.show()
    print(avg_projected_velocity)
    return snake, sol_history


def run_phase_space(snake_type, **kwargs):
    from p_tqdm import p_map
    from psutil import cpu_count

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

    phase_space_kwargs = list(cartesian_product(**kwargs))
    phase_space_ids = list(range(1, len(phase_space_kwargs) + 1))
    # write a file that contains the number-key/pairs mapping

    SnakeWriter.write_ids_to_disk(snake_type, phase_space_ids, phase_space_kwargs)

    # Need this to pass the snake type into fwd_to_run_snake
    updated_phase_space_kwargs = OrderedDict(kwargs)
    updated_phase_space_kwargs.update({'snake_type': snake_type})
    updated_phase_space_kwargs.move_to_end('snake_type', last=False)
    updated_phase_space_kwargs.move_to_end('time_interval', last=False)
    updated_phase_space_kwargs.move_to_end('froude', last=False)
    updated_phase_space_kwargs_list = list(cartesian_product(**updated_phase_space_kwargs))

    _ = p_map(
        fwd_to_run_snake,
        updated_phase_space_kwargs_list,
        phase_space_ids,
        num_cpus=cpu_count(logical=False),
    )


def fwd_to_run_snake(kwargs, id):
    snake, sol, time_period = run_snake(**kwargs)
    SnakeWriter.write_snake_to_disk(id, sol)


def main():
    """

    Returns
    -------

    """

    """

    Exploring solution space 
    ---------------------

    """
    """
    1.1 Running a single case with default parameters for (curvature)
    activation (referred to as simply activation) for a non-lifting snake.
    The default activation is a cos wave with 
    𝜅 = ε cos (k 𝛑 (s + t))
    where ε = 7.0 and k = 2.0
    """
    snake, sol_history = run_and_visualize(
        froude=1, time_interval=[0.0, 10.0], snake_type=KinematicSnake,
        mu_f=1.0, mu_b=1.5, mu_lat=2.0
    )

    """
    1.2 For running a single case with non-default parameters for
    activation (that is changing ε, k) in the activation below simply
    𝜅 = ε cos (k 𝛑 (s + t))
    please do the following. Note that the order of parameters beyond
    the keyword `snake_type` does not matter.
    """
    epsilon = 5.0
    wave_number = 5.0
    snake, sol_history = run_and_visualize(
        froude=1, time_interval=[0.0, 10.0], snake_type=KinematicSnake,
        mu_f=1.0, mu_b=1.5, mu_lat=2.0, epsilon=epsilon, wave_number=wave_number
    )

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
    snake, sol_history = run_and_visualize(
        froude=1, time_interval=[0.0, 10.0], snake_type=LiftingKinematicSnake,
        mu_f=1.0, mu_b=1.5, mu_lat=2.0
    )

    """
    2.2 If you want to specify non-default parameters for either the curvature
    or lifting wave, please do. Once again, order of keywords do not matter
    """
    epsilon = 5.0
    wave_number = 5.0
    lift_amp = 0.4
    lift_wave_number = 2.0
    phase = 0.3

    snake, sol_history = run_and_visualize(
        froude=1, time_interval=[0.0, 10.0], snake_type=LiftingKinematicSnake,
        mu_f=1.0, mu_b=1.5, mu_lat=2.0, epsilon=epsilon, wave_number=wave_number,
        lift_amp=lift_amp, lift_wave_number=lift_wave_number, phase=phase
    )

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

    def my_custom_activation(s, time_v):
        return epsilon * cos(wave_number * pi * (s + time_v))

    def my_custom_lifting_activation(s, time_v):
        if time_v > 2.0:
            liftwave = lift_amp * np.cos(wave_number * np.pi * (s + phase + time_v)) + 1.0
            np.maximum(0, liftwave, out=liftwave)
            return liftwave / trapz(liftwave, s)
        else:
            return 1.0 + 0.0 * s

    snake, sol_history = run_and_visualize(
        froude=1, time_interval=[0.0, 10.0], snake_type=KinematicSnake,
        mu_f=1.0, mu_b=1.5, mu_lat=2.0,
        activation=my_custom_activation, lifting_activation=my_custom_lifting_activation
    )

    """
    
    Running a phase-space
    ---------------------
    
    """

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
            "wave_number": [4., 5., 6.]
        }
    )

    ps = run_phase_space(snake_type=KinematicSnake, **kwargs)

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
            "wave_number": [4., 5., 6.],
            "lift_wave_number": [2.0, 3.0],
            "lift_amp": [2.0, 3.0],
            "phase" : [0.25, 0.5, 0.75]
        }
    )

    """
    5. If you want to load up a saved-snake from file (after running a
    phase-space), you can consult the simulation_id from the `ids.csv` file
    that the simulation creates within the `data` folder in the home directory.
    
    Once you know the simulation id you can pass it to the SnakeReader as shown
    below, which constructs the snake and the time-history of COM motions. With
    these two parameters, its possible to reconstruct all desired quantitites.
    Look at `run_and_visualize` for a more complete example
    """
    snake, sol_history = SnakeReader.load_snake_from_disk(1)


if __name__ == "__main__":
    main()
