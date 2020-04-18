import numpy as np
import pickle
from functools import partial
from sympy import sin, cos, pi
from scipy.integrate import solve_ivp, trapz, simps
from collections import OrderedDict
from kinematic_snake.kinematic_snake import (
    KinematicSnake,
    LiftingKinematicSnake,
    project,
)
from kinematic_snake.circle_fit import fit_circle_to_data
from kinematic_snake.compute_circle import findCircle #compute circle from three points

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
                liftwave = (
                    lift_amp * np.cos(lift_wave_number * np.pi * (s + phase + time_v))
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
        snake.state.copy().reshape(-1,),
        method="RK23",
        events=events,
        # t_eval = np.linspace(time_interval[0], time_interval[1], 1e4)
    )

    # Append t_events and y_events to the final solution history
    # Monkey patching
    if events is not None:
        insert_idx = np.searchsorted(sol.t, sol.t_events)
        sol.t = np.insert(sol.t, insert_idx[:, 0], np.array(sol.t_events).reshape(-1,))
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

        return snake, sol


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
    def write_snake_to_disk(id: int, sol):
        with open(SnakeWriter.pickled_file_name.format(id=id), "wb") as file_handler:
            # # Doesn't work
            # pickle.dump([snake, sol], file_handler)
            # Works
            pickle.dump(sol, file_handler)


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
        "average_of_force_magnitude": cumulative_average_force_magnitude / n_iters,
        "average_of_normal_force_magnitude": cumulative_average_force_magnitude_in_normal_direction
        / n_iters,
    }


def calculate_period_idx(fin_time, t_period, sol_his_t, candidate_n_past_periods=8):
    # Give a transitent of at leat 0.4*final_time
    n_past_periods = (
        candidate_n_past_periods
        if ((fin_time - candidate_n_past_periods * t_period) > 0.4 * fin_time)
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
    final_time,
    pose_ang_his=None,
    steer_ang_his=None,
    pose_rate_his=None,
    steer_rate_his=None,
):
    """ Calculates average pose angle, average pose_angle_rate
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
            simps(x[..., past_per_index:], sol_his.t[past_per_index:], axis=-1,)
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


    # Noel's added code to compute the effective speed based on the radius of the circle 
    # Averaging helps smooth things out, likely due to the adaptive time-stepper not alwys
    # giving consistent number of steps, therefor throwing the averaging off. A fix for this
    # would be to having a constant time-step, but this has its own performance issues. 
    # 
    # Adding all the code here to make it easy. Could be better integrated, such as using 
    # calculate_period_start_stop_idx instead of the function below. 
    def calculate_past_periods_idx(fin_time, t_period, sol_his_t, periods_to_average):
        # get the ids of the last 10 periods for computing average values
        past_period_idx_all = []
        for n_past_periods in range(periods_to_average+1):
            past_period_idx_all.append(np.argmin(np.abs(sol_his_t - (fin_time - n_past_periods * t_period))) )
        return past_period_idx_all

    # these values should be better defined. Hard coded for no2.
    periods_to_average = 10
    time_period = 1.0
    period_ids = calculate_past_periods_idx(final_time, time_period, sol_his.t, periods_to_average)

    avg_com_list = []
    # compute the locations of the average COM for each period. Skip the first entry
    # because it is the last index in the list. Could probably be better integrated with trapz of simps
    for i in range(1,len(period_ids)):
        period_com_all = sol_his.y[:2, period_ids[i]:period_ids[i-1]]
        avg_com_list.append(np.average(period_com_all,1))
    radius_collection = []
    speed_effective_collection = []
    steering_rate_collection = []
    #skip the first and last entries because you need three entries to make the calculation
    for i in range(1,len(avg_com_list)-1):
        radius = findCircle(avg_com_list[i-1], avg_com_list[i], avg_com_list[i+1])
        radius_collection.append(radius)
        # compute distance traveled over two periods
        distance = np.linalg.norm(avg_com_list[i-1] - avg_com_list[i+1]) 
        # compute the steering rate along the trajectory of circle defined by the radius. 
        # since distance is over two periods, the effective rate (rads/ period) is half
        steering_rate = 2. * np.arcsin(distance/(2.*radius)) * 1./2. 
        steering_rate_collection.append(steering_rate)
        # compute effective speed as steering rate * radius
        speed_effective_collection.append(radius * steering_rate)
    

    # A few possible extensions:
    # For radii over some large value (say 1e3 or 1e4) we might want to just compute the speed
    # as if it were a straight line, instead of a curve. This would avoid possible floating
    # point errors for very large radii. 

    # Computing the pose angle based on this formulation isn't totally clear to me. We can get 
    # a direction vector for the effective speed, so then compute the average of the snake 
    # position over period and take the angle between the two?

    # One more sanity check we can do is compare the effective force vector and the effective speed 
    # vector. I think they should be orthogonal. 

    # get the average values. 
    speed_effective = np.mean(speed_effective_collection)
    steering_rate_avg = np.mean(steering_rate_collection)
    radius_avg = np.mean(radius_collection)

    return {
        "average_pose_angle": averager(pose_ang_his),
        "average_steer_angle": averager(steer_ang_his),
        "average_pose_rate": averager(pose_rate_his),
        "average_steer_rate": averager(steer_rate_his),
        "average_speed": averager(average_speed),
        "fit_circle_x_center": xc,
        "fit_circle_y_center": yc,
        "fit_circle_radius": rc,
        "average_circle_radius_new": radius_avg,
        "average_steering_rate_new": steering_rate_avg,
        "effective_speed": speed_effective,
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
        final_time,
        kwargs.get("pose_ang_his", None),
        kwargs.get("steer_ang_his", None),
        kwargs.get("pose_rate_his", None),
        kwargs.get("steer_rate_his", None),
    )

    averaged_cumulative_stats.update(averaged_force_stats)
    return averaged_cumulative_stats


def run_and_visualize(*args, **kwargs):
    snake, sol_history, time_period = run_snake(*args, **kwargs)

    from matplotlib import pyplot as plt
    from matplotlib.colors import to_rgb

    # Ugly hack from now
    # FIXME : Abstract this away from the user
    PLOT_VIDEO = False
    if PLOT_VIDEO:
        import matplotlib.animation as manimation

        plt.rcParams.update({"font.size": 22})

        FFMpegWriter = manimation.writers["ffmpeg"]
        metadata = dict(
            title="Movie Test", artist="Matplotlib", comment="Movie support!"
        )
        fps = 60
        writer = FFMpegWriter(fps=fps, metadata=metadata)
        dpi = 200
        fig = plt.figure(figsize=(10, 8), frameon=True, dpi=dpi)
        ax = fig.add_subplot(111)
        ax.set_aspect("equal", adjustable="box")
        # plt.axis("square")
        time = 0.0
        solution = sol_history.y.T[0]
        snake.state = solution.reshape(-1, 1)
        snake._construct(time)
        (snake_centerline,) = ax.plot(snake.x_s[0, ...], snake.x_s[1, ...], lw=2.0)
        ax.set_aspect("equal", adjustable="box")
        video_name = "snake.mp4"
        ax.set_xlim([-1.0, 1.0])
        ax.set_ylim([-1.0, 1.0])
        video_skip = 8
        with writer.saving(fig, video_name, dpi):
            with plt.style.context("fivethirtyeight"):
                for time, solution in zip(
                    sol_history.t[1::video_skip], sol_history.y.T[1::video_skip]
                ):
                    snake.state = solution.reshape(-1, 1)
                    snake._construct(time)
                    snake_centerline.set_xdata(snake.x_s[0, ...])
                    snake_centerline.set_ydata(snake.x_s[1, ...])
                    writer.grab_frame()

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
            ext_force = snake.external_force_distribution(time)
            phys_space_ax.plot(
                snake.x_s[0, ...],
                snake.x_s[1, ...],
                c=to_rgb("xkcd:bluish"),
                alpha=10 ** (step / n_steps - 1),
                lw=2,
            )

        # plot COM for each timestep
        phys_space_ax.plot(
                sol_history.y[0, :],
                sol_history.y[1, :],
                'k.',
                markersize = 1,
                )

        # plot average COM for each period
        final_time = kwargs["time_interval"][1]

        # get locations for each complete period
        period_ids = []
        for n_past_periods in range(int(final_time)):
            period_ids.append(np.argmin(np.abs(sol_history.t - (final_time - n_past_periods * time_period))) )
        period_ids.append(0) # add starting point of series to list

        for i in range(1,len(period_ids)):
            period_com_all = sol_history.y[:2, period_ids[i]:period_ids[i-1]]
            avg_com = np.average(period_com_all,1)
            phys_space_ax.plot(
                avg_com[0],
                avg_com[1],
                'ro',
                markersize = 3,
                # alpha=10**((len(period_ids) - i) / (len(period_ids))-1 ),
            )

        quiver_skip = int(snake.centerline.size / 30)
        phys_space_ax.quiver(
            snake.x_s[0, ::quiver_skip],
            snake.x_s[1, ::quiver_skip],
            ext_force[0, ::quiver_skip],
            ext_force[1, ::quiver_skip],
        )
        phys_space_ax.set_aspect("equal")

        n_steps = sol_history.t.size

        pose_angle_history = np.zeros((n_steps,))
        steering_angle_history = np.zeros((n_steps,))

        pose_rate_history = np.zeros((n_steps,))
        steering_rate_history = np.zeros((n_steps,))

        # Determine time for average statistics
        # Average statistics
        # calculate snake motion axes based on n_past_periods time-periods
        final_time = kwargs["time_interval"][1]

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
            avg_radius_new,
            avg_steer_rate_new,
            effective_speed,
            mag_avg_force,
            avg_force_mag,
            avg_force_mag_in_normal_dir,
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
        # print(avg_projected_velocity)
    return snake, sol_history, time_period


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


def fwd_to_run_snake(kwargs, id):
    snake, sol_history, time_period = run_snake(**kwargs)
    SnakeWriter.write_snake_to_disk(id, sol_history)

    final_time = kwargs["time_interval"][1]

    return calculate_statistics(
        snake, sol_history, final_time, time_period, candidate_n_past_periods=8
    )


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
    # snake, sol_history, time_period = run_and_visualize(
    #     froude=1,
    #     time_interval=[0.0, 10.0],
    #     snake_type=KinematicSnake,
    #     mu_f=1.0,
    #     mu_b=1.5,
    #     mu_lat=2.0,
    # )

    """
    1.2 For running a single case with non-default parameters for
    activation (that is changing ε, k) in the activation below simply
    𝜅 = ε cos (k 𝛑 (s + t))
    please do the following. Note that the order of parameters beyond
    the keyword `snake_type` does not matter.
    """
    # epsilon = 5.0
    # wave_number = 5.0
    # snake, sol_history, time_period = run_and_visualize(
    #     froude=1,
    #     time_interval=[0.0, 10.0],
    #     snake_type=KinematicSnake,
    #     mu_f=1.0,
    #     mu_b=1.5,
    #     mu_lat=2.0,
    #     epsilon=epsilon,
    #     wave_number=wave_number,
    # )

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
    # snake, sol_history, time_period = run_and_visualize(
    #     froude=1,
    #     time_interval=[0.0, 10.0],
    #     snake_type=LiftingKinematicSnake,
    #     mu_f=1.0,
    #     mu_b=1.5,
    #     mu_lat=2.0,
    # )

    """
    2.2 If you want to specify non-default parameters for either the curvature
    or lifting wave, please do. Once again, order of keywords do not matter
    """
    # epsilon = 7.0
    # wave_number = 2.0
    # lift_amp = 1.0
    # lift_wave_number = 2.0
    # phase = 0.3

    # snake, sol_history, time_period = run_and_visualize(
    #     froude=1,
    #     time_interval=[0.0, 20.0],
    #     snake_type=LiftingKinematicSnake,
    #     mu_f=1.0,
    #     mu_b=1.5,
    #     mu_lat=2.0,
    #     epsilon=epsilon,
    #     wave_number=wave_number,
    #     lift_amp=lift_amp,
    #     lift_wave_number=lift_wave_number,
    #     phase=phase,
    # )

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
    # epsilon = 7.0
    # wave_number = 2.0
    # lift_amp = 0.0
    # phase = 0.26

    # def my_custom_activation(s, time_v):
    #     return epsilon * cos(wave_number * pi * (s + time_v))

    # def my_custom_lifting_activation(s, time_v):
    #     if time_v > 2.0:
    #         liftwave = (
    #             lift_amp * np.cos(wave_number * np.pi * (s + phase + time_v)) + 1.0
    #         )
    #         np.maximum(0, liftwave, out=liftwave)
    #         return liftwave / trapz(liftwave, s)
    #     else:
    #         return 1.0 + 0.0 * s

    # snake, sol_history, time_period = run_and_visualize(
    #     froude=1,
    #     time_interval=[0.0, 10.0],
    #     snake_type=LiftingKinematicSnake,
    #     mu_f=1.0,
    #     mu_b=1.5,
    #     mu_lat=2.0,
    #     activation=my_custom_activation,
    #     lifting_activation=my_custom_lifting_activation,
    # )

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
            "wave_number": [4.0, 5.0, 6.0],
        }
    )

    # ps = run_phase_space(snake_type=KinematicSnake, **kwargs)

    """
    4.2 Phase-space for a LiftingSnake
    """
    kwargs = OrderedDict(
        {
            "froude": [0.01],
            "time_interval": [[0.0, 20.0]],
            "mu_f": [1.0],
            "mu_b": [1.5],
            "mu_lat": [2.0],
            "epsilon": [7.0],
            "wave_number": [2.0],
            "lift_wave_number": [2.0],
            "lift_amp": [0.0, 0.5, 1.0],
            "phase": [0.0, 0.125, 0.25],
        }
    )

    ps = run_phase_space(snake_type=LiftingKinematicSnake, **kwargs)

    """
    5. If you want to load up a saved-snake from file (after running a
    phase-space), you can consult the simulation_id from the `ids.csv` file
    that the simulation creates within the `data` folder in the home directory.
    
    Once you know the simulation id you can pass it to the SnakeReader as shown
    below, which constructs the snake and the time-history of COM motions. With
    these two parameters, its possible to reconstruct all desired quantitites.
    Look at `run_and_visualize` for a more complete example
    """
    # snake, sol_history = SnakeReader.load_snake_from_disk(1)


if __name__ == "__main__":
    main()
