import numpy as np
import pickle
from functools import partial
import sympy as sp
from scipy.integrate import solve_ivp, trapz, simps, OdeSolution
from collections import OrderedDict
from kinematic_snake.kinematic_snake import (
    KinematicSnake,
    LiftingKinematicSnake,
    project,
)
from kinematic_snake.circle_fit import fit_circle_to_data
from os import path, makedirs


def make_snake(froude, time_interval, snake_type, **kwargs):
    friction_coefficients = {"mu_f": None, "mu_b": None, "mu_lat": None}
    friction_coefficients.update(kwargs)

    snake = KinematicSnake(
        froude_number=froude, friction_coefficients=friction_coefficients
    )

    def activation(s, time, epsilon, wave_number):
        return epsilon * sp.cos(2.0 * wave_number * sp.pi * (s + time))

    wave_number = kwargs.get("wave_number", 1.0)
    bound_activation = kwargs.get(
        "activation",
        partial(
            activation,
            epsilon=kwargs.get("epsilon", 7.0),
            wave_number=wave_number,
        ),
    )

    snake.set_activation(bound_activation)

    if snake_type == LiftingKinematicSnake:

        def lifting_activation(s, time_v, phase, lift_amp, lift_wave_number):
            if time_v > 2.0:
                liftwave = (
                    lift_amp
                    * np.cos(2.0 * lift_wave_number * np.pi * (s + phase + time_v))
                    + 1.0
                )
                np.maximum(0, liftwave, out=liftwave)
                return liftwave / trapz(liftwave, s)
            else:
                return 1.0 + 0.0 * s

        bound_lifting_activation = kwargs.get(
            "lifting_activation",
            partial(
                lifting_activation,
                phase=kwargs.get("phase", 0.26),
                lift_amp=kwargs.get("lift_amp", 1.0),
                lift_wave_number=kwargs.get("lift_wave_number", wave_number),
            ),
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
        snake.state.copy().reshape(
            -1,
        ),
        method="RK23",
        events=events,
        # t_eval = np.linspace(time_interval[0], time_interval[1], 1e4)
    )

    # Append t_events and y_events to the final solution history
    # Monkey patching
    if events is not None:
        insert_idx = np.searchsorted(sol.t, sol.t_events)
        sol.t = np.insert(
            sol.t,
            insert_idx[:, 0],
            np.array(sol.t_events).reshape(
                -1,
            ),
        )
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
            sol, time_period = pickle.load(pickled_file)

        ### Can use eval, but dangerous
        if snake_cls == KinematicSnake.__name__:
            snake_cls = KinematicSnake
        elif snake_cls == LiftingKinematicSnake.__name__:
            snake_cls = LiftingKinematicSnake
        else:
            raise ValueError(
                "The name of class from disk is invalid! Please raise an"
                "error report on Github!"
            )

        snake, wave_number = make_snake(
            froude=kwargs.pop("froude"),
            time_interval=kwargs.pop("time_interval"),
            snake_type=snake_cls,
            **kwargs
        )

        return snake, sol, time_period


class SnakeWriter(SnakeIO):
    @staticmethod
    def make_data_directory():
        makedirs(SnakeWriter.data_folder_name, exist_ok=True)

    @staticmethod
    def write_ids_to_disk(snake_type, phase_space_ids, phase_space_kwargs):
        SnakeWriter.make_data_directory()

        import csv

        with open(SnakeWriter.id_file_name, "w", newline="") as id_file_handler:
            # Write header first
            csvwriter = csv.writer(id_file_handler, delimiter=",")
            snake_type_str = [snake_type.__name__]
            csvwriter.writerow(snake_type_str)
            header_row = ["id"] + list(phase_space_kwargs[0].keys())
            csvwriter.writerow(header_row)
            for id, args_dict in zip(phase_space_ids, phase_space_kwargs):
                temp = [id]
                temp.extend(args_dict.values())
                csvwriter.writerow(temp)

    @staticmethod
    def write_snake_to_disk(id: int, sol: OdeSolution, time_period: float):
        with open(SnakeWriter.pickled_file_name.format(id=id), "wb") as file_handler:
            # # Doesn't work
            # pickle.dump([snake, sol], file_handler)
            # Works
            pickle.dump((sol, time_period), file_handler)


def calculate_average_force_per_cycle(
    sim_snake,
    sol_his,
    period_start_idx,
    period_stop_idx,
    force_history_in_cycle=None,
    mag_normal_projection_of_force_history_in_cycle=None,
):
    n_steps = period_stop_idx - period_start_idx

    if force_history_in_cycle is None:
        force_history_in_cycle = np.zeros((2, n_steps))

    if mag_normal_projection_of_force_history_in_cycle is None:
        mag_normal_projection_of_force_history_in_cycle = np.zeros((n_steps))

    instantaneous_normal = np.zeros((2, n_steps))

    start_end = slice(period_start_idx, period_stop_idx)

    for step, (time, solution) in enumerate(
        zip(sol_his.t[start_end], sol_his.y[:, start_end].T)
    ):
        sim_snake.state = solution.reshape(-1, 1)
        sim_snake._construct(time)
        # tangent is along (v_x, v_y)
        # normal is along (-v_y, v_x)
        # shape is (2,1) to support project
        instantaneous_normal[:, step] = np.array([-solution[4], solution[3]])

        # Put the spatial average wihin force_history first
        force_history_in_cycle[:, step] = trapz(
            sim_snake.external_force_distribution(time), sim_snake.centerline, axis=-1
        )
        # Can be done all in one go if efficieny needed
        # mag_normal_projection_of_force_history_in_cycle[step], _ = project(
        #     force_history_in_cycle[:, step], instantaneous_normal
        # )

        # force_history_in_cycle[:, step] = trapz(
        #     np.array([[0.0], [1.0]]) + 0.0 * snake.external_force_distribution(time), snake.centerline, axis=-1
        # )

        # force_history_in_cycle[:, step] = np.cos(2.0 * np.pi * time)
    # project does (ij,ij->i) so that we put timeaxis first here
    mag_normal_projection_of_force_history_in_cycle[:], _ = np.abs(
        project(force_history_in_cycle, instantaneous_normal)
    )

    time_elapsed = sol_his.t[start_end]
    avg_force = simps(force_history_in_cycle, time_elapsed, axis=-1)
    # Take mag of force acting at each time-step and then average it out
    avg_mag_force = simps(
        np.linalg.norm(force_history_in_cycle, axis=0), time_elapsed, axis=-1
    )
    # Norm of force in the normal direction
    avg_mag_force_in_normal_direction = simps(
        mag_normal_projection_of_force_history_in_cycle, time_elapsed, axis=-1
    )
    # Should equal the period
    time_elapsed = time_elapsed[-1] - time_elapsed[0]

    return (
        avg_force / time_elapsed,
        avg_mag_force / time_elapsed,
        avg_mag_force_in_normal_direction / time_elapsed,
    )


def calculate_period_start_stop_idx(
    sol_his_time, fin_time, t_period, candidate_n_past_periods
):
    # Give a transitent of at leat 0.4*final_time
    n_past_periods = (
        candidate_n_past_periods
        if ((fin_time - candidate_n_past_periods * t_period) > 0.4 * fin_time)
        else 3
    )
    for i_period in range(n_past_periods):
        start_time = fin_time - (n_past_periods - i_period) * t_period
        # end_time = (fin_time - (n_past_periods - i_period - 1) * t_period)
        end_time = start_time + t_period
        start_period_idx = np.argmin(np.abs(sol_his_time - start_time))
        end_period_idx = np.argmin(np.abs(sol_his_time - end_time))
        # Note that +1 here. This is to accomodate slicing till the next period point
        # and not one before
        yield start_period_idx, end_period_idx + 1


def calculate_statistics_over_n_cycles(
    sim_snake, sol_his, fin_time, time_period, candidate_n_past_periods
):
    """

    Returns
    -------

    """
    # ||\int f(t) dt|| for t from n * T to (n+1) * T
    cumulative_magnitude_of_average_force = 0.0
    # \int ||f(t)|| dt for t from n * T to (n+1) * T
    cumulative_average_force_magnitude = 0.0
    # \int ||f(t) . \hat{n}|| dt for t from n * T to (n+1) * T
    cumulative_average_force_magnitude_in_normal_direction = 0.0
    n_iters = 0
    for start_idx, stop_idx in calculate_period_start_stop_idx(
        sol_his.t, fin_time, time_period, int(candidate_n_past_periods)
    ):
        (
            avg_force,
            avg_force_norm,
            avg_force_norm_in_normal_direction,
        ) = calculate_average_force_per_cycle(sim_snake, sol_his, start_idx, stop_idx)
        cumulative_magnitude_of_average_force += np.linalg.norm(avg_force)
        cumulative_average_force_magnitude += avg_force_norm
        cumulative_average_force_magnitude_in_normal_direction += (
            avg_force_norm_in_normal_direction
        )
        n_iters += 1

    return {
        "magnitude_of_average_force": cumulative_magnitude_of_average_force / n_iters,
        # Commented out as per XZ's request
        # "average_of_force_magnitude": cumulative_average_force_magnitude / n_iters,
        # "average_of_normal_force_magnitude": cumulative_average_force_magnitude_in_normal_direction
        # / n_iters,
    }


def calculate_period_idx(
    fin_time, t_period, sol_his_t, candidate_n_past_periods=8, override=False
):
    # Give a transitent of at leat 0.4*final_time
    n_past_periods = (
        candidate_n_past_periods
        if ((fin_time - candidate_n_past_periods * t_period) > 0.4 * fin_time)
        or override
        else 3
    )
    past_period_idx = np.argmin(
        np.abs(sol_his_t - (fin_time - n_past_periods * t_period))
    )
    return past_period_idx, n_past_periods * t_period


def calculate_cumulative_statistics(
    sim_snake,
    sol_his,
    past_per_index,
    past_time,
    pose_ang_his=None,
    steer_ang_his=None,
    pose_rate_his=None,
    steer_rate_his=None,
):
    """Calculates average pose angle, average pose_angle_rate
    and average turning_rate statistics, cumulated for last many cycles

    Returns
    -------

    """
    n_steps = sol_his.t.size

    if pose_ang_his is None:
        pose_ang_his = np.zeros((n_steps,))
    if steer_ang_his is None:
        steer_ang_his = np.zeros((n_steps,))
    if pose_rate_his is None:
        pose_rate_his = np.zeros((n_steps,))
    if steer_rate_his is None:
        steer_rate_his = np.zeros((n_steps,))

    for step, (time, solution) in enumerate(zip(sol_his.t, sol_his.y.T)):
        sim_snake.state = solution.reshape(-1, 1)
        sim_snake._construct(time)

        pose_ang_his[step] = sim_snake.calculate_instantaneous_pose_angle(time)
        steer_ang_his[step] = sim_snake.calculate_instantaneous_steering_angle(time)

        ## FIXME : Correct rate calculation
        #### Don't compute pose_rate analytically because it's wrong.

    pose_rate_his[0] = 0.0
    steer_rate_his[0] = 0.0
    for step in range(sol_his.t.size - 2):
        pose_rate_his[step + 1] = (pose_ang_his[step + 2] - pose_ang_his[step]) / (
            sol_his.t[step + 2] - sol_his.t[step]
        )
        steer_rate_his[step + 1] = (steer_ang_his[step + 2] - steer_ang_his[step]) / (
            sol_his.t[step + 2] - sol_his.t[step]
        )

    def averager(x):
        # calculate average statistics
        # assert(np.allclose(past_time, sol_his.t[-1] - sol_his.t[past_per_index]))
        # Try and replace with simps to see if there's any change in the results
        avg_val = (
            simps(
                x[..., past_per_index:],
                sol_his.t[past_per_index:],
                axis=-1,
            )
            / past_time
        )
        return avg_val

    # Get \dot{x} and \dot{y} to compute norms
    velocity_com = sol_his.y[3:5, ...]
    average_speed = np.linalg.norm(velocity_com, axis=0)

    # Get x and y from the point that we are considering averages from
    # to pass it onto the circle fitting algorithm
    position_com_over_averaging_window = sol_his.y[:2, past_per_index:]
    xc, yc, rc = fit_circle_to_data(position_com_over_averaging_window, verbose=False)

    # Do not trust the above radius, use the following estimate
    # Teja : In the "good" cases, it doesn't make any difference
    computed_com = np.array([xc, yc])
    integrated_distance = np.linalg.norm(
        sol_his.y[:2].reshape(2, -1) - computed_com.reshape(2, 1), 2, axis=0
    )

    return {
        "average_pose_angle": averager(pose_ang_his),
        "average_steer_angle": averager(steer_ang_his),
        # Doing this seems to accumulate error from floating point precision effects
        # Rather use Leibniz's integral theorem \int_{T_N}^{T_M} df/dt dt = f(T_M) - f(T_N)
        # "average_pose_rate": averager(pose_rate_his),
        # "average_steer_rate": averager(steer_rate_his),
        "average_pose_rate": (pose_ang_his[-1] - pose_ang_his[past_per_index])
        / past_time,
        "average_steer_rate": (steer_ang_his[-1] - steer_ang_his[past_per_index])
        / past_time,
        "average_speed": averager(average_speed),
        "fit_circle_x_center": xc,
        "fit_circle_y_center": yc,
        "fit_circle_radius": averager(integrated_distance),
        # "least_squares_fit_circle_radius": rc,
        "time_elapsed_in_past_periods": past_time,
    }


def calculate_statistics(
    sim_snake, sol_his, final_time, time_period, candidate_n_past_periods=8, **kwargs
):
    # First calculate per period statistics over candidate_n_cycles
    averaged_force_stats = calculate_statistics_over_n_cycles(
        sim_snake, sol_his, final_time, time_period, candidate_n_past_periods
    )

    past_period_idx, time_elapsed_in_past_periods = calculate_period_idx(
        final_time, time_period, sol_his.t
    )

    # The compute cumulative statistics
    averaged_cumulative_stats = calculate_cumulative_statistics(
        sim_snake,
        sol_his,
        past_period_idx,
        time_elapsed_in_past_periods,
        kwargs.get("pose_ang_his", None),
        kwargs.get("steer_ang_his", None),
        kwargs.get("pose_rate_his", None),
        kwargs.get("steer_rate_his", None),
    )

    averaged_cumulative_stats.update(averaged_force_stats)
    return averaged_cumulative_stats


def animate_snake_with_interpolation(snake, sol_history, time_period, snake_id=None):
    from matplotlib import pyplot as plt
    from matplotlib.colors import to_rgb, LinearSegmentedColormap
    from tqdm import tqdm
    import matplotlib.animation as manimation

    plt.rcParams.update({"font.size": 22})

    FFMpegWriter = manimation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    fps = 50
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    dpi = 200
    plt.style.use("seaborn-whitegrid")
    fig = plt.figure(figsize=(10, 8), frameon=True, dpi=dpi)
    ax = fig.add_subplot(111)
    ax.set_aspect("equal", adjustable="box")
    # plt.axis("square")
    time = 0.0
    solution = sol_history.y.T[0]
    snake.state = solution.reshape(-1, 1)
    snake._construct(time)
    (snake_centerline,) = ax.plot(
        snake.x_s[0, ...], snake.x_s[1, ...], lw=2.5, solid_capstyle="round"
    )
    x_com = []
    y_com = []
    # colors = np.zeros((2, 4))
    # colors[:, 3] = 1.0 # alpha channel
    # Only one that works : https://stackoverflow.com/a/48156476
    colors = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.5], [0.0, 0.0, 0.0, 1.0]]
    cmap = LinearSegmentedColormap.from_list("", colors)
    snake_com = ax.scatter(x_com, y_com, c=[], s=16, cmap=cmap, vmin=0, vmax=1)
    time_in_text = plt.text(
        0.5,
        0.9,
        "T = 0.0",
        fontsize=22,
        horizontalalignment="center",
        transform=ax.transAxes,
    )
    ax.set_aspect("equal", adjustable="box")
    if snake_id is None:
        video_name = "snake.mp4"
    else:
        video_name = "snake_{:05d}.mp4".format(snake_id)
    ax.set_xlim([-1.0, 1.0])
    ax.set_ylim([-1.0, 1.0])

    # Should depend on the total size o f
    n_steps_in_total = sol_history.t.shape[0]
    # video_skip = max(int(n_steps_in_total / fps / 20.0), 1)
    # print(n_steps_in_total, video_skip)
    # return

    total_time = sol_history.t[-1]
    # 1 physical second is 2 simulation second
    # fps is 50 => 2 simulation second has 50 frames
    # 1 frame is then 2/50 of second
    # spacing is 2/50
    # dt = 2 / fps
    # total_n_points = T / dt = fps * T / 2
    total_n_points = int(0.5 * total_time * fps)
    t_mesh = np.linspace(0.0, total_time, total_n_points + 1)
    print(
        "Number of simulation samples {0} vs number of interpolated samples {1}".format(
            n_steps_in_total, total_n_points
        )
    )

    # scatter plot recency parameter / exp weight
    recency_weight = 0.25
    # scat_colors = np.zeros((t_mesh.shape[0], 4))
    # scat_colors[:, 3] = 1.0
    # scat_intensity = np.zeros((t_mesh.shape[0], ))

    # We now form cubic interpolants to all data at different points
    from scipy.interpolate import interp1d

    state_interpolant = interp1d(sol_history.t, sol_history.y)

    interpolated_mesh = state_interpolant(t_mesh)

    # if we want average COM, let's calculate that first
    # T, 2T, 3T ... nT
    periods = np.linspace(time_period, total_time, int(total_time / time_period))
    period_indices = [np.argmin(np.abs(t_mesh - period)) for period in periods]
    # Annoying +1 here to account for the fact that the full cycle is from [start,stop] (stop includeed)
    # which doesn't work upon slicing
    period_indices_start_stop_pair = [
        (x, y + 1) for x, y in zip(period_indices[:-1], period_indices[1:])
    ]

    def averager(x, start_idx, stop_idx):
        # calculate average statistics
        # assert(np.allclose(past_time, sol_his.t[-1] - sol_his.t[past_per_index]))
        # Try and replace with simps to see if there's any change in the results
        return (
            simps(
                x[..., start_idx:stop_idx],
                t_mesh[start_idx:stop_idx],
                axis=-1,
            )
            / time_period
        )

    # The first element contains average of first cycle and should be plotting after t = period
    average_com = np.array(
        [
            averager(interpolated_mesh[:2], start, stop)
            for (start, stop) in period_indices_start_stop_pair
        ]
    )
    # The cisualization starts from this point onwards
    start_cycle = 6
    average_com_counter = start_cycle
    average_snake_com = ax.scatter([], [], c="r", s=16)

    with writer.saving(fig, video_name, dpi):
        for t_ind, time in enumerate(tqdm(t_mesh)):
            solution = interpolated_mesh[:, t_ind]
            snake.state = solution.reshape(-1, 1)
            snake._construct(time)
            snake_centerline.set_xdata(snake.x_s[0, ...])
            snake_centerline.set_ydata(snake.x_s[1, ...])
            time_in_text.set_text("T = {:.2f}".format(time))
            # Set x and y valyes
            snake_com.set_offsets(interpolated_mesh[:2, :t_ind].T)
            # past_time = t_mesh[:t_ind]
            # scat_colors[:t_ind, 3] = np.exp(-recency_weight * (past_time - time))
            # scat_intensity[:t_ind] *= recency_weight
            # scat_intensity[t_ind - 1] = 1.0
            # snake_com.set_array(scat_colors[:, 3])
            past_time = t_mesh[:t_ind]
            snake_com.set_array(np.exp(recency_weight * (past_time - time)))
            if t_ind == period_indices[average_com_counter]:
                average_snake_com.set_offsets(
                    average_com[start_cycle:average_com_counter, :2]
                )
                average_com_counter += 1
            writer.grab_frame()


def animate_snake(snake, sol_history, time_period, snake_id=None):
    from matplotlib import pyplot as plt
    from matplotlib.colors import to_rgb, LinearSegmentedColormap
    from tqdm import tqdm
    import matplotlib.animation as manimation

    plt.rcParams.update({"font.size": 22})

    FFMpegWriter = manimation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    fps = 50
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    dpi = 200
    plt.style.use("seaborn-whitegrid")
    fig = plt.figure(figsize=(10, 8), frameon=True, dpi=dpi)
    ax = fig.add_subplot(111)
    ax.set_aspect("equal", adjustable="box")
    # plt.axis("square")
    time = 0.0
    solution = sol_history.y.T[0]
    snake.state = solution.reshape(-1, 1)
    snake._construct(time)
    (snake_centerline,) = ax.plot(
        snake.x_s[0, ...], snake.x_s[1, ...], lw=2.5, solid_capstyle="round"
    )
    x_com = []
    y_com = []
    # colors = np.zeros((2, 4))
    # colors[:, 3] = 1.0 # alpha channel
    # Only one that works : https://stackoverflow.com/a/48156476
    snake_com = ax.scatter(x_com, y_com, c="k", s=16)
    time_in_text = plt.text(
        0.5,
        0.9,
        "T = 0.0",
        fontsize=22,
        horizontalalignment="center",
        transform=ax.transAxes,
    )
    ax.set_aspect("equal", adjustable="box")
    if snake_id is None:
        video_name = "snake_nointerp.mp4"
    else:
        video_name = "snake_nointerp_{:05d}.mp4".format(snake_id)
    ax.set_xlim([-1.0, 1.0])
    ax.set_ylim([-1.0, 1.0])

    # Should depend on the total size o f
    # n_steps_in_total = sol_history.t.shape[0]
    video_skip = 2

    # total_time = sol_history.t[-1]

    with writer.saving(fig, video_name, dpi):
        for time, solution in zip(
            sol_history.t[1::video_skip], sol_history.y.T[1::video_skip]
        ):
            snake.state = solution.reshape(-1, 1)
            snake._construct(time)
            snake_centerline.set_xdata(snake.x_s[0, ...])
            snake_centerline.set_ydata(snake.x_s[1, ...])
            time_in_text.set_text("T = {:.2f}".format(time))
            # Not efficient, but whatever
            x_com.append(solution[0])
            y_com.append(solution[1])
            snake_com.set_offsets(np.c_[x_com, y_com])
            writer.grab_frame()


def visualize_snake(snake, sol_history, time_period, snake_id=None, **kwargs):
    from matplotlib import pyplot as plt
    from matplotlib.colors import to_rgb

    with plt.style.context("fivethirtyeight"):
        phys_space_fig = plt.figure(figsize=(10, 8))
        phys_space_ax = phys_space_fig.add_subplot(111)

        angle_fig = plt.figure(figsize=(10, 8))
        angle_ax = angle_fig.add_subplot(111)

        angle_rate_fig = plt.figure(figsize=(10, 8))
        angle_rate_ax = angle_rate_fig.add_subplot(111)

        n_steps = sol_history.t.size
        skip = max(int(n_steps / 20), 1)
        # skip = 1
        n_steps = sol_history.t[::skip].size

        for step, (time, solution) in enumerate(
            zip(sol_history.t[::skip], sol_history.y.T[::skip])
        ):
            snake.state = solution.reshape(-1, 1)
            snake._construct(time)
            # ext_force = snake.external_force_distribution(time)
            phys_space_ax.plot(
                snake.x_s[0, ...],
                snake.x_s[1, ...],
                c=to_rgb("xkcd:bluish"),
                alpha=10 ** (step / n_steps - 1),
                lw=2,
                solid_capstyle="round",
            )

        # Determine time for average statistics
        # Average statistics
        # calculate snake motion axes based on n_past_periods time-periods
        # final_time = kwargs["time_interval"][1]
        final_time = sol_history.t[-1]

        # Plot centerlines as well starting from some time-perid
        past_idx, _ = calculate_period_idx(
            final_time,
            time_period,
            sol_history.t,
            candidate_n_past_periods=85,
            override=True,
        )

        phys_space_ax.scatter(
            sol_history.y[0, past_idx:],
            sol_history.y[1, past_idx:],
            c="k",
            s=16,
        )

        # quiver_skip = int(snake.centerline.size / 30)
        # # Don't plot a quiver for now
        # phys_space_ax.quiver(
        #     snake.x_s[0, ::quiver_skip],
        #     snake.x_s[1, ::quiver_skip],
        #     ext_force[0, ::quiver_skip],
        #     ext_force[1, ::quiver_skip],
        # )
        phys_space_ax.set_aspect("equal")

        n_steps = sol_history.t.size

        pose_angle_history = np.zeros((n_steps,))
        steering_angle_history = np.zeros((n_steps,))

        pose_rate_history = np.zeros((n_steps,))
        steering_rate_history = np.zeros((n_steps,))

        statistics = calculate_statistics(
            snake,
            sol_history,
            final_time,
            time_period,
            pose_ang_his=pose_angle_history,
            steer_ang_his=steering_angle_history,
            pose_rate_his=pose_rate_history,
            steer_rate_his=steering_rate_history,
        )

        (
            avg_pos_angle,
            avg_steer_angle,
            avg_pos_rate,
            avg_steer_rate,
            avg_speed,
            xc,
            yc,
            avg_radius,
            time_elapsed_in_past_periods,
            mag_avg_force,
            # avg_force_mag,
            # avg_force_mag_in_normal_dir,
        ) = statistics.values()

        angle_ax.plot(
            sol_history.t,
            pose_angle_history,
            marker="o",
            c=to_rgb("xkcd:bluish"),
            label="pose",
            lw=2,
        )
        angle_ax.hlines(
            y=avg_pos_angle,
            xmin=sol_history.t[0],
            xmax=sol_history.t[-1],
            lw=2,
            colors="k",
            linestyles="dashed",
        )
        angle_ax.plot(
            sol_history.t,
            steering_angle_history,
            marker="o",
            c=to_rgb("xkcd:reddish"),
            label="steering",
            lw=2,
        )
        angle_ax.hlines(
            y=avg_steer_angle,
            xmin=sol_history.t[0],
            xmax=sol_history.t[-1],
            lw=2,
            colors="k",
            linestyles="dashed",
        )
        angle_ax.legend()

        angle_rate_ax.plot(
            sol_history.t,
            pose_rate_history,
            marker="o",
            c=to_rgb("xkcd:bluish"),
            label="pose_rate",
            lw=2,
        )
        angle_rate_ax.hlines(
            y=avg_pos_rate,
            xmin=sol_history.t[0],
            xmax=sol_history.t[-1],
            lw=2,
            colors="k",
            linestyles="dashed",
        )
        angle_rate_ax.plot(
            sol_history.t,
            steering_rate_history,
            marker="o",
            c=to_rgb("xkcd:reddish"),
            label="steering_rate",
            lw=2,
        )
        angle_rate_ax.hlines(
            y=avg_steer_rate,
            xmin=sol_history.t[0],
            xmax=sol_history.t[-1],
            lw=2,
            colors="k",
            linestyles="dashed",
        )
        angle_rate_ax.legend()

        velocity_fig = plt.figure(figsize=(10, 8))
        velocity_ax = velocity_fig.add_subplot(111)

        # as a bonus, plot it on the physical axes
        past_period_idx, _ = calculate_period_idx(
            final_time, time_period, sol_history.t
        )
        phys_space_ax.plot(
            [sol_history.y[0, past_period_idx], sol_history.y[0, -1]],
            [sol_history.y[1, past_period_idx], sol_history.y[1, -1]],
            c=to_rgb("xkcd:reddish"),
            lw=2,
        )

        travel_axes = sol_history.y[:2, -1] - sol_history.y[:2, past_period_idx]
        travel_axes /= np.linalg.norm(travel_axes, ord=2)
        com_velocity = sol_history.y[3:5, ...]

        if kwargs.get("snake_type") == KinematicSnake:
            # Broadcast shape
            lateral_axes = np.array([-travel_axes[1], travel_axes[0]])
            travel_axes = travel_axes.reshape(-1, 1) + 0.0 * com_velocity
            lateral_axes = lateral_axes.reshape(-1, 1) + 0.0 * com_velocity
            from kinematic_snake.kinematic_snake import project

            mag_projected_velocity_along_travel, _ = project(com_velocity, travel_axes)
            mag_projected_velocity_along_lateral, _ = project(
                com_velocity, lateral_axes
            )
            projected_velocity = np.vstack(
                (
                    mag_projected_velocity_along_travel,
                    mag_projected_velocity_along_lateral,
                )
            )

            # calculate average statistics
            avg_projected_velocity = (
                simps(
                    projected_velocity[..., past_period_idx:],
                    sol_history.t[past_period_idx:],
                    axis=-1,
                )
                / time_elapsed_in_past_periods
            )

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
        elif kwargs.get("snake_type", LiftingKinematicSnake):
            velocity_ax.plot(sol_history.t, com_velocity[0], lw=2, label="x")
            velocity_ax.plot(sol_history.t, com_velocity[1], lw=2, label="y")

        velocity_ax.hlines(
            y=avg_speed,
            xmin=sol_history.t[0],
            xmax=sol_history.t[-1],
            lw=0.5,
            colors="k",
            linestyles="dashed",
            label="average speed (mag vel)",
        )
        velocity_ax.legend()

        phys_space_fig.show()
        angle_fig.show()
        angle_rate_fig.show()
        velocity_fig.show()
        plt.show()


def run_and_visualize(*args, **kwargs):
    snake, sol_history, time_period = run_snake(*args, **kwargs)
    if kwargs.get("animate", False):
        animate_snake(snake, sol_history, time_period, snake_id=None)
    else:
        visualize_snake(snake, sol_history, time_period, snake_id=None, **kwargs)
    return snake, sol_history, time_period


def fwd_to_run_snake(kwargs, id):
    snake, sol_history, time_period = run_snake(**kwargs)
    SnakeWriter.write_snake_to_disk(id, sol_history, time_period)

    final_time = kwargs["time_interval"][1]

    return calculate_statistics(
        snake, sol_history, final_time, time_period, candidate_n_past_periods=8
    )


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
    # Pospone till after simulations are done, but make the directory immediately
    # SnakeWriter.write_ids_to_disk(snake_type, phase_space_ids, phase_space_kwargs)
    SnakeWriter.make_data_directory()

    # Need this to pass the snake type into fwd_to_run_snake
    updated_phase_space_kwargs = OrderedDict(kwargs)
    updated_phase_space_kwargs.update({"snake_type": [snake_type]})
    updated_phase_space_kwargs.move_to_end("snake_type", last=False)
    updated_phase_space_kwargs.move_to_end("time_interval", last=False)
    updated_phase_space_kwargs.move_to_end("froude", last=False)
    updated_phase_space_kwargs_list = list(
        cartesian_product(**updated_phase_space_kwargs)
    )

    # fwd_to_run_snake(updated_phase_space_kwargs[0], phase_space_ids[0])
    statistics = p_map(
        fwd_to_run_snake,
        updated_phase_space_kwargs_list,
        phase_space_ids,
        num_cpus=cpu_count(logical=False),
    )

    # Append together the statistics and arguments
    for args, stats in zip(phase_space_kwargs, statistics):
        args.update(stats)

    SnakeWriter.write_ids_to_disk(snake_type, phase_space_ids, phase_space_kwargs)
