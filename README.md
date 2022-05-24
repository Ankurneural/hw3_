# hw3_

## How to register custom env:

Go the folder a3_gym_env where the setup.py file is kept.
Run:

pip install -e .


Make changes in the script where ever you want to gym.make the cutom env:

1. import a3_gym_env

2. gym.make('Pendulum-v1-custom')


The file for :

PPO+without clip: PPO_Noclip.py 
PPO implementation : PPO_2.py
Vanilla policy:VPG.py 
All the grpahs and PPO implementation can be viewed in test.ipynb file.

How to run :
python <file name>

  

