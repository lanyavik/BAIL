## Anonymized code for BAIL paper

The code is currently anonymized for review. 

Most important code can be found in the `spinup/algos` folder. The BAIL implementation can be found under `spinup/algos/BAIL/`.
Code runs with pytorch >= 1.2, mujoco 150

## Run experiment
Create a folder of the path "spinup/algos/BAIL/buffers" and put all the RL batches there. 

Then you run BAIL in the following procedure:

Under the BAIL folder, first do the returns calculation with `python main_get_mcret.py`, which stores the calculated returns in "./results".

Then run Static BAIL with `python main_static_bail.py` or run Progressive BAIL with `python main_prog_bail.py`. 
We use the logger of Spinup to record algorithmic performances. You may Consult Spinup documentation for output and plotting:
(https://spinningup.openai.com/en/latest/user/saving_and_loading.html, https://spinningup.openai.com/en/latest/user/plotting.html). Or you could modify the code and save it in your style.



## Reference: 


Implementation of the BCQ algorithm: https://github.com/sfujim/BCQ

Code will be released publicly on github with a refined documentation after the review perioed. 

