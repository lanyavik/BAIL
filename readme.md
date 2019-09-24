## Anonymized code for BAIL paper

code is currently anonymized for review. 

Most important code can be found in the `spinup/algos` folder. The ue folder contains the code for generating the upper envelope.
The BAIL folder contains the code for training the BAIL algorithm. The sac_pytorch folder contains code for generating SAC buffer. 

Code runs with pytorch 1.2, mujoco 150

## Run experiment
The SAC implementation can be found under `spinup/algos/sac_pytorch/`

Run experiments with pytorch sac: 

In the sac_pytorch folder, run the SAC code with `python sac_pytorch.py`

Note: currently there is no parallel running for SAC (also not supported by spinup), so you should always set number of cpu to 1 when you use experiment grid.

The program structure, though in Pytorch has been made to be as close to spinup tensorflow code as possible so readers who are familiar with other algorithm code in spinup will find this one easier to work with. I also referenced rlkit's SAC pytorch implementation, especially for the policy and value models part, but did a lot of simplification. 

Consult Spinup documentation for output and plotting:

https://spinningup.openai.com/en/latest/user/saving_and_loading.html

https://spinningup.openai.com/en/latest/user/plotting.html

## Reference: 

Original SAC paper: https://arxiv.org/abs/1801.01290

OpenAI Spinup docs on SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html

rlkit sac implementation: https://github.com/vitchyr/rlkit

Code will be released publicly on github with a refined documentation after the review perioed. 

