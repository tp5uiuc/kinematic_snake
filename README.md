# Snake Kinematics

Kinematic model of snake-locomotion, with and without lifting, following Hu & Shelley (2012). Can run single
specified cases or many independent cases in parallel to generate a phase-space

## Prerequisites
To install required prerequisities, please do
```shell script
pip install -r requirements.txt
```
More specifically some features require latest version of `scipy.integrate` to detect
events while integrating the non-linear ODEs.