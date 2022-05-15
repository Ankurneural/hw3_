# hw3_

## How to register custom env:

Go the folder a3_gym_env where the setup.py file is kept.
Run:

pip install -e .


Make changes in the script where ever you want to gym.make the cutom env:

1. import a3_gym_env

2. gym.make('Pendulum-v1-custom')


