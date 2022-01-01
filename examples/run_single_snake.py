from kinematic_snake import run_and_visualize, KinematicSnake, LiftingKinematicSnake

if __name__ == "__main__":
    print("Running a snake with default activation")
    snake, sol_history, time_period = run_and_visualize(
        froude=1,  # The froude number
        time_interval=[0.0, 10.0],  # Time interval of simulation
        snake_type=KinematicSnake,  # Type of snake
        mu_f=1.0,  # Forward friction coefficient ratio, determined from `froude`
        mu_b=1.5,  # Backward friction coefficient
        mu_lat=2.0,  # Lateral friction coefficient
    )

    print("Running a snake that also lifts with default activations")
    snake, sol_history, time_period = run_and_visualize(
        froude=1,  # The froude number
        time_interval=[0.0, 10.0],  # Time interval of simulation
        snake_type=LiftingKinematicSnake,  # Type of snake, can be KinematicSnake/Li
        mu_f=1.0,  # Forward friction coefficient ratio, determined from `froude`
        mu_b=1.5,  # Backward friction coefficient
        mu_lat=2.0,  # Lateral friction coefficient
    )
