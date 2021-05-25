## BAIL

Code for the paper "BAIL: Best-Action Imitation Learning for Batch Deep Reinforcement Learning"(link: https://arxiv.org/abs/1910.12179), published at NeurIPS 2020.

The BAIL implementation can be found under `spinup/algos/BAIL/`.
Code runs with pytorch >= 1.2, mujoco 150

## Run experiment
Create a folder of the path "spinup/algos/BAIL/buffers" and put the RL batches there. 

Then you run BAIL in the following procedure:

Under the BAIL folder, first do the returns calculation with `python main_get_mcret.py`, which stores the calculated returns in "./results".

Then run BAIL with `python main_static_bail.py` or run Progressive BAIL with `python main_prog_bail.py`. 
We use the logger of Spinup to record algorithmic performances. You may consult Spinup documentation for output and plotting:
(https://spinningup.openai.com/en/latest/user/saving_and_loading.html, https://spinningup.openai.com/en/latest/user/plotting.html). Or you could modify the code and save it in your style.



## Reference: 


Implementation of the BCQ algorithm: https://github.com/sfujim/BCQ


## BAIL performance on 62 batches

|ENVIRONMENT|BAIL MEAN|BAIL STD|
|---|---|---|
|sigma=0.1 Hopper B1|2173|291|
|sigma=0.1 Hopper B2|2078|180|
|sigma=0.1 Walker B1|1125|113|
|sigma=0.1 Walker B2|3141|300|
|sigma=0.1 HC B1|5746|29|
|sigma=0.1 HC B2|7212|43|
|sigma=0.5 Hopper B1|2054|158|
|sigma=0.5 Hopper B2|2623|282|
|sigma=0.5 Walker B1|2522|51|
|sigma=0.5 Walker B2|3115|133|
|sigma=0.5 HC B1|1055|9|
|sigma=0.5 HC B2|7173|120|
|SAC Hopper B1|3296|105|
|SAC Hopper B2|1831|915|
|SAC Walker B1|2455|211|
|SAC Walker B2|4767|130|
|SAC HC B1|10143|77|
|SAC HC B2|10772|59|
|SAC Ant B1|4284|64|
|SAC Ant B2|4946|148|
|SAC Humanoid B1|3852|430|
|SAC Humanoid B2|3565|153|
|M sigma=0 Hopper B1|1026|0|
|M sigma=0 Hopper B2|696|233|
|M sigma=0 Walker B1|437|20|
|M sigma=0 Walker B2|500|12|
|M sigma=0 HC B1|4057|69|
|M sigma=0 HC B2|4013|12|
|M sigma=0 Ant B1|753|9|
|M sigma=0 Ant B2|738|4|
|M sigma=0 Humanoid B1|4313|139|
|M sigma=0 Humanoid B2|4053|252|
|M sigma=sigma(s) Hopper B1|375|52|
|M sigma=sigma(s) Hopper B2|254|102|
|M sigma=sigma(s) Walker B1|384|21|
|M sigma=sigma(s) Walker B2|512|24|
|M sigma=sigma(s) HC B1|4744|19|
|M sigma=sigma(s) HC B2|4123|19|
|M sigma=sigma(s) Ant B1|790|9|
|M sigma=sigma(s) Ant B2|781|6|
|M sigma=sigma(s) Humanoid B1|1375|387|
|M sigma=sigma(s) Humanoid B2|1309|372|
|O sigma=0 Hopper B1|2602|5|
|O sigma=0 Hopper B2|3046|34|
|O sigma=0 Walker B1|2735|26|
|O sigma=0 Walker B2|3019|6|
|O sigma=0 HC B1|11265|243|
|O sigma=0 HC B2|11360|265|
|O sigma=0 Ant B1|4901|65|
|O sigma=0 Ant B2|4975|108|
|O sigma=0 Humanoid B1|4872|895|
|O sigma=0 Humanoid B2|5320|125|
|O sigma=sigma(s) Hopper B1|2359|153|
|O sigma=sigma(s) Hopper B2|2035|217|
|O sigma=sigma(s) Walker B1|2834|120|
|O sigma=sigma(s) Walker B2|3200|16|
|O sigma=sigma(s) HC B1|10258|1255|
|O sigma=sigma(s) HC B2|10882|634|
|O sigma=sigma(s) Ant B1|4981|91|
|O sigma=sigma(s) Ant B2|5067|83|
|O sigma=sigma(s) Humanoid B1|2129|381|
|O sigma=sigma(s) Humanoid B2|4328|569|


## Citation
If you use our code, please cite our paper:
```
@inproceedings{chen2020bail,
 author = {Chen, Xinyue and Zhou, Zijian and Wang, Zheng and Wang, Che and Wu, Yanqiu and Ross, Keith},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {H. Larochelle and M. Ranzato and R. Hadsell and M. F. Balcan and H. Lin},
 pages = {18353--18363},
 publisher = {Curran Associates, Inc.},
 title = {BAIL: Best-Action Imitation Learning for Batch Deep Reinforcement Learning},
 url = {https://proceedings.neurips.cc/paper/2020/file/d55cbf210f175f4a37916eafe6c04f0d-Paper.pdf},
 volume = {33},
 year = {2020}
}
```
