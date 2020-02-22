# Snake Kinematics

Kinematic model of snake-locomotion, with and without lifting, following Hu & Shelley (2012). Can run single
specified cases or many independent cases in parallel to generate a phase-space

## Prerequisites
To install required prerequisities, please do
```shell script
pip install -r requirements.txt
```
More specifically some features require the latest version of `scipy.integrate` to detect
events while integrating the non-linear ODEs.

Some prerequisites are optional, unless you are running a phase-space in parallel. In which case please 
execute
```shell script
pip install -r optional-requirements.txt
```

## Design
The code is designed to encourage the user to play around with the model of the snake using 
any activation function (for both the snake gait, specified by curvature, and the lifting). Few examples
are provided in the `examples` folder which should get you started.
