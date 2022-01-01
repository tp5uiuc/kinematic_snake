Snake Locomotion Kinematics
&middot;
[![CI](https://github.com/tp5uiuc/kinematic_snake/actions/workflows/ci.yml/badge.svg)](https://github.com/tp5uiuc/kinematic_snake/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT-green)](https://mit-license.org/)
[![pyversion](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue.svg)](https://www.python.org/)
=====

A kinematic model of snake-locomotion, with and without lifting, following our publication[](). 
Can run single specified cases or many independent cases in parallel to generate a phase-space.

> :rocket: Before installing the package, check out an interactive, online version [at this link](https://gazzolalab.github.io/kinematic_snake_sandbox/).

## Installation
For a system wide install, use
```sh
python3 -m pip install kinematic-snake
```
For a local install, please clone this repository and execute the following in the repository directory.
```sh
python3 -m pip install --user .
```
You can then use one of the [examples](examples) for running a single simulation or
a parameter sweep of simulations. For more information see [Usage and examples](#usage-and-examples)

### Extras
If you want to take full advantage of the package (such as running a phase-space of cases in parallel), 
please execute
```sh
python3 -m pip install kinematic-snake[extras]
```

## Usage and examples
The simplest example runs a [single simulation](examples/run_single_snake.py) and produces output 
plots. For convenience, the same code is also listed below.
```python
from kinematic_snake import run_and_visualize, Snake

snake, sol_history, time_period = run_and_visualize(
    froude=1, # The froude number
    time_interval=[0.0, 10.0], # Time interval of simulation
    snake_type=Snake, # Type of snake, can be Snake
    mu_f=1.0, # Forward friction coefficient ratio, determined from `froude`
    mu_b=1.5, # Backward friction coefficient
    mu_lat=2.0, # Lateral friction coefficient
)
```
More examples including the one above are given in that example file. The code is designed to 
encourage the user to play around with the model of the snake using 
any activation function (for both the snake gait, specified by curvature, and the lifting). The documented examples
provided in the `examples` folder should get you started.

## Numerical algorithms
Details on the algorithms employed can be found in our following paper. If you are using this package, 
please cite the work below :)

<strong>Friction modulation in limbless, three-dimensional gaits and heterogeneous terrains</strong>, doi : [10.1038/s41467-021-26276-x](https://doi.org/10.1038/s41467-021-26276-x)

```
@Article{Zhang2021,
author={Zhang, Xiaotian
and Naughton, Noel
and Parthasarathy, Tejaswin
and Gazzola, Mattia},
title={Friction modulation in limbless, three-dimensional gaits and heterogeneous terrains},
journal={Nature Communications},
year={2021},
month={Oct},
day={19},
volume={12},
number={1},
pages={6076},
issn={2041-1723},
doi={10.1038/s41467-021-26276-x},
url={https://doi.org/10.1038/s41467-021-26276-x}
}
```
